"""
Target Defense Environment in VMAS
Agents control only heading (direction) via first action dimension and always move at maximum speed
Variable number of attackers and defenders with sensing-based observations
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import math

from vmas import render_interactively
from vmas.simulator.core import Agent, World, Landmark, Sphere
from vmas.simulator.scenario import BaseScenario
from vmas.simulator.utils import Y, X

# Try importing analytical solver
try:
    from apollonius_solver import solve_apollonius_optimization
    APOLLONIUS_AVAILABLE = True
except ImportError:
    APOLLONIUS_AVAILABLE = False
    print("Warning: apollonius_solver not available. Using fallback rewards.")


class Scenario(BaseScenario):
    """
    Target Defense Scenario for VMAS
    Defenders try to sense/intercept attackers before they reach the target line
    
    Action Format:
    - Actions are 2D tensors with shape (batch_size, 2)
    - First dimension: heading angle NORMALIZED to [-1, 1] (maps to [-π, π] radians)
      * 1.0 = π radians (180°, West)
      * 0.5 = π/2 radians (90°, North)
      * 0.0 = 0 radians (0°, East)
      * -0.5 = -π/2 radians (-90°, South)
      * -1.0 = -π radians (-180°)
    - Second dimension: ignored (required by VMAS but not used)
    - Agents always move at maximum speed in the specified heading direction
    """
    
    def make_world(self, batch_dim: int, device: torch.device, **kwargs) -> World:
        """
        Create the world with agents and parameters
        
        Args:
            batch_dim: Number of parallel environments
            device: Device to run on (cpu/cuda)
            **kwargs: Additional scenario parameters
        """
        # Extract parameters from kwargs with defaults
        num_defenders = kwargs.get('num_defenders', 3)
        num_attackers = kwargs.get('num_attackers', 1)
        sensing_radius = kwargs.get('sensing_radius', 0.15)
        attacker_sensing_radius = kwargs.get('attacker_sensing_radius', 0.2)
        speed_ratio = kwargs.get('speed_ratio', 0.7)
        target_distance = kwargs.get('target_distance', 0.05)
        defender_color = kwargs.get('defender_color', (0.0, 0.0, 1.0))
        attacker_color = kwargs.get('attacker_color', (1.0, 0.0, 0.0))
        randomize_attacker_x = kwargs.get('randomize_attacker_x', False)
        fixed_attacker_policy = kwargs.get('fixed_attacker_policy', True)
        num_spawn_positions = kwargs.get('num_spawn_positions', 3)  # Generalized spawn positions
        # Store scenario parameters
        self.batch_dim = batch_dim
        self.device = device
        self.num_defenders = num_defenders
        self.num_attackers = num_attackers
        self.sensing_radius = sensing_radius
        self.attacker_sensing_radius = attacker_sensing_radius
        self.speed_ratio = speed_ratio
        self.target_distance = target_distance
        self.randomize_attacker_x = randomize_attacker_x
        self.fixed_attacker_policy = fixed_attacker_policy
        self.num_spawn_positions = num_spawn_positions
        
        # Speed settings
        self.defender_max_speed = 0.05
        self.attacker_max_speed = self.defender_max_speed * self.speed_ratio
        
        # Create world
        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=0.5,  # World goes from -0.5 to 0.5
            y_semidim=0.5,
            collision_force=0,  # No collisions
            substeps=1,
            dt=1.0  # Unit timestep
        )
        
        # Create defender agents
        for i in range(num_defenders):
            agent = Agent(
                name=f"defender_{i}",
                shape=Sphere(radius=0.02),
                color=defender_color,
                max_speed=self.defender_max_speed,
                rotatable=False,  # No rotation needed for point agents
                silent=True
                # Note: We keep default action_size (2) and handle heading in process_action
            )
            agent.is_defender = True
            agent.sensing_radius = sensing_radius
            world.add_agent(agent)
        
        # Create attacker agents
        for i in range(num_attackers):
            agent = Agent(
                name=f"attacker_{i}",
                shape=Sphere(radius=0.02),
                color=attacker_color,
                max_speed=self.attacker_max_speed,
                rotatable=False,
                silent=True
                # Note: We keep default action_size (2) and handle heading in process_action
            )
            agent.is_defender = False
            agent.sensing_radius = attacker_sensing_radius
            world.add_agent(agent)
        
        # Store episode tracking variables (will be initialized in reset_world_at)
        # Track sensing for EACH attacker separately
        self.attacker_sensed = None  # Will be shape (batch_dim, num_attackers)
        self.attacker_reached_target = None  # Will be shape (batch_dim, num_attackers)
        self.attacker_sensing_rewards = None  # Will be shape (batch_dim, num_attackers)
        
        # Initialize them here for immediate use
        self.attacker_sensed = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_reached_target = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_sensing_rewards = torch.zeros((batch_dim, num_attackers), device=device)
        self.defender_has_sensed = torch.zeros((batch_dim, num_defenders), dtype=torch.bool, device=device)
        
        return world
    
    def reset_world_at(self, env_index: Optional[int] = None):
        """
        Reset world to initial positions
        
        Args:
            env_index: Index of the environment to reset (for vectorized envs)
        """
        # Get defenders and attackers
        defenders = [a for a in self.world.agents if a.is_defender]
        attackers = [a for a in self.world.agents if not a.is_defender]
        
        # Position defenders evenly along bottom
        defender_spacing = 1.0 / (self.num_defenders + 1)
        for i, defender in enumerate(defenders):
            x_pos = (i + 1) * defender_spacing - 0.5  # Convert to [-0.5, 0.5]
            
            if env_index is None:
                defender.state.pos[:, X] = x_pos
                defender.state.pos[:, Y] = -0.5
                defender.state.vel[:, :] = 0
            else:
                defender.state.pos[env_index, X] = x_pos
                defender.state.pos[env_index, Y] = -0.5
                defender.state.vel[env_index, :] = 0
        
        # Create spawn positions once for all attackers
        if self.randomize_attacker_x:
            # Create equally spaced spawn positions
            # For K positions, space them evenly across [-0.4, 0.4]
            if self.num_spawn_positions == 1:
                spawn_positions = torch.tensor([0.0], device=self.device)
            else:
                # Equally spaced positions
                spacing = 0.8 / (self.num_spawn_positions - 1)  # Total range is 0.8
                spawn_positions = torch.tensor(
                    [-0.4 + i * spacing for i in range(self.num_spawn_positions)],
                    device=self.device
                )
            
            # Ensure attackers spawn at different positions
            if env_index is None:
                batch_size = attackers[0].state.pos.shape[0] if attackers else self.batch_dim
                
                # For each environment, select unique positions for attackers
                for env_idx in range(batch_size):
                    # Get available positions
                    available_positions = spawn_positions.clone()
                    
                    # If we have more attackers than spawn positions, some will have to share
                    if self.num_attackers > self.num_spawn_positions:
                        # Repeat positions to have at least as many as attackers
                        repeats = (self.num_attackers // self.num_spawn_positions) + 1
                        available_positions = available_positions.repeat(repeats)[:self.num_attackers]
                        # Shuffle to randomize which positions get repeated
                        perm = torch.randperm(len(available_positions))
                        available_positions = available_positions[perm]
                    else:
                        # Randomly select unique positions for each attacker
                        perm = torch.randperm(len(available_positions))[:self.num_attackers]
                        available_positions = available_positions[perm]
                    
                    # Assign positions to attackers
                    for i, attacker in enumerate(attackers):
                        if i < len(available_positions):
                            attacker.state.pos[env_idx, X] = available_positions[i]
                        else:
                            # Fallback (shouldn't happen)
                            attacker.state.pos[env_idx, X] = 0.0
            else:
                # Single environment reset
                available_positions = spawn_positions.clone()
                
                if self.num_attackers > self.num_spawn_positions:
                    # Similar logic for single environment
                    repeats = (self.num_attackers // self.num_spawn_positions) + 1
                    available_positions = available_positions.repeat(repeats)[:self.num_attackers]
                    perm = torch.randperm(len(available_positions))
                    available_positions = available_positions[perm]
                else:
                    perm = torch.randperm(len(available_positions))[:self.num_attackers]
                    available_positions = available_positions[perm]
                
                for i, attacker in enumerate(attackers):
                    if i < len(available_positions):
                        attacker.state.pos[env_index, X] = available_positions[i]
                    else:
                        attacker.state.pos[env_index, X] = 0.0
        else:
            # Fixed center position for all attackers (original behavior)
            for i, attacker in enumerate(attackers):
                x_pos = 0.0
                if env_index is None:
                    attacker.state.pos[:, X] = x_pos
                else:
                    attacker.state.pos[env_index, X] = x_pos
        
        # Set Y position and velocity for all attackers
        for attacker in attackers:
            if env_index is None:
                attacker.state.pos[:, Y] = 0.5
                attacker.state.vel[:, :] = 0
            else:
                attacker.state.pos[env_index, Y] = 0.5
                attacker.state.vel[env_index, :] = 0
        
        # Reset episode tracking - track each attacker separately
        if env_index is None:
            batch_size = self.batch_dim if hasattr(self, 'batch_dim') else self.world.batch_dim
            device = self.device if hasattr(self, 'device') else self.world.device
            self.attacker_sensed = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_reached_target = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_sensing_rewards = torch.zeros((batch_size, self.num_attackers), device=device)
            self.defender_has_sensed = torch.zeros((batch_size, self.num_defenders), dtype=torch.bool, device=device)
            self.distances = torch.zeros((batch_size, self.num_defenders, self.num_attackers), device=device)
        else:
            self.attacker_sensed[env_index, :] = False
            self.attacker_reached_target[env_index, :] = False
            self.attacker_sensing_rewards[env_index, :] = 0.0
            if hasattr(self, 'defender_has_sensed'):
                self.defender_has_sensed[env_index, :] = False
            if hasattr(self, 'distances'):
                self.distances[env_index, :, :] = 0.0
    
    def reward(self, agent: Agent) -> torch.Tensor:
        """
        Compute reward for an agent - ONLY at episode end to avoid double counting
        
        Args:
            agent: The agent to compute reward for
            
        Returns:
            Reward tensor for the agent
        """
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        # Initialize tracking variables if not already done
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            self.attacker_sensed = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_reached_target = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_sensing_rewards = torch.zeros((batch_size, self.num_attackers), device=device)
            self.defender_has_sensed = torch.zeros((batch_size, self.num_defenders), dtype=torch.bool, device=device)
            self.distances = torch.zeros((batch_size, self.num_defenders, self.num_attackers), device=device)
        
        # Check for sensing events and track distances
        defenders = [a for a in self.world.agents if a.is_defender]
        attackers = [a for a in self.world.agents if not a.is_defender]
        
        # Update sensing status and track distances for EACH attacker-defender pair
        for attacker_idx, attacker in enumerate(attackers):
            for defender_idx, defender in enumerate(defenders):
                dist = torch.norm(attacker.state.pos - defender.state.pos, dim=-1)
                # Store current distance for monitoring
                self.distances[:, defender_idx, attacker_idx] = dist
                
                newly_sensed = (dist <= self.sensing_radius) & ~self.attacker_sensed[:, attacker_idx]
                
                # When sensing occurs, snap attacker to boundary and compute rewards
                if newly_sensed.any():
                    # Mark this defender as having sensed an attacker
                    self.defender_has_sensed[:, defender_idx] |= newly_sensed
                    # Snap attacker position to the sensing boundary
                    for env_idx in torch.where(newly_sensed)[0]:
                        def_pos = defender.state.pos[env_idx]
                        att_pos = attacker.state.pos[env_idx]
                        
                        # Calculate direction from defender to attacker
                        direction = att_pos - def_pos
                        direction_norm = torch.norm(direction)
                        
                        # If inside sensing radius, move to boundary
                        if direction_norm < self.sensing_radius and direction_norm > 0:
                            # Normalize direction and place at boundary
                            direction_normalized = direction / direction_norm
                            attacker.state.pos[env_idx] = def_pos + direction_normalized * self.sensing_radius
                        
                        # Also set velocity to zero
                        attacker.state.vel[env_idx] = torch.zeros_like(attacker.state.vel[env_idx])
                    
                    if APOLLONIUS_AVAILABLE:
                        # Compute how "good" the interception was based on attacker position
                        # Higher Y position (closer to start) = better interception
                        # Y ranges from 0.5 (start) to -0.5 (target)
                        attacker_y = attacker.state.pos[:, Y]
                        
                        # Normalize: 0.5 (best) to -0.5 (worst) → 1.0 to 0.0
                        quality = (attacker_y + 0.5)  # Now 0 to 1
                        
                        # Scale the reward: early interception = higher reward
                        # Range: 0.5 (late interception) to 1.5 (early interception)
                        self.attacker_sensing_rewards[newly_sensed, attacker_idx] = 0.5 + quality[newly_sensed]
                    else:
                        # Default reward if Apollonius not available
                        self.attacker_sensing_rewards[newly_sensed, attacker_idx] = 1.0
                
                self.attacker_sensed[:, attacker_idx] |= newly_sensed
        
        # Check for target reached for EACH attacker
        for attacker_idx, attacker in enumerate(attackers):
            reached = (attacker.state.pos[:, Y] <= -0.5 + self.target_distance) & ~self.attacker_sensed[:, attacker_idx]
            self.attacker_reached_target[:, attacker_idx] |= reached
        
        # Check if episode is done for any environment
        episode_done = self.done()
        
        # Compute rewards ONLY at episode end to avoid double counting
        reward = torch.zeros(batch_size, device=device)
        
        if agent.is_defender and episode_done.any():
            # For environments where episode ended, compute final rewards
            for env_idx in torch.where(episode_done)[0]:
                total_reward = 0.0
                
                # Sum rewards for all attackers in this environment
                for attacker_idx in range(self.num_attackers):
                    if self.attacker_sensed[env_idx, attacker_idx]:
                        # Positive reward for sensing
                        total_reward += self.attacker_sensing_rewards[env_idx, attacker_idx]
                    elif self.attacker_reached_target[env_idx, attacker_idx]:
                        # Negative reward for attacker reaching target
                        total_reward -= 10.0 / self.num_defenders  # Split penalty among defenders
                
                reward[env_idx] = total_reward
        elif not agent.is_defender:
            # Attackers get no rewards (fixed policy)
            pass
        
        return reward
    
    def observation(self, agent: Agent) -> torch.Tensor:
        """
        Get observation for an agent
        Observation includes positions of all agents, but opponents only if within sensing radius
        
        Args:
            agent: The agent to get observation for
            
        Returns:
            Observation tensor of shape (batch_size, obs_dim)
        """
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        # Total observation size: (num_defenders + num_attackers) * 2
        obs_dim = (self.num_defenders + self.num_attackers) * 2
        obs = torch.zeros(batch_size, obs_dim, device=device)
        
        # Get all agents sorted by type and index
        defenders = sorted([a for a in self.world.agents if a.is_defender], 
                          key=lambda x: x.name)
        attackers = sorted([a for a in self.world.agents if not a.is_defender], 
                          key=lambda x: x.name)
        all_agents_ordered = defenders + attackers
        
        # Fill observation
        idx = 0
        for other_agent in all_agents_ordered:
            if other_agent == agent:
                # Always observe own position
                obs[:, idx:idx+2] = other_agent.state.pos
            elif (agent.is_defender and other_agent.is_defender) or \
                 (not agent.is_defender and not other_agent.is_defender):
                # Same team - always visible
                obs[:, idx:idx+2] = other_agent.state.pos
            else:
                # Opponent - only visible if within sensing radius
                dist = torch.norm(agent.state.pos - other_agent.state.pos, dim=-1)
                within_range = dist <= agent.sensing_radius
                
                # Set position only for agents within range, zeros otherwise
                obs[within_range, idx:idx+2] = other_agent.state.pos[within_range]
                # Agents outside range remain as zeros (unobserved)
            
            idx += 2
        
        return obs
    
    def extra_render(self, env_index: int = 0) -> Optional[str]:
        """Called by VMAS to get extra render information."""
        # Reset the current step rewards when moving to next step
        if hasattr(self, 'current_step_rewards'):
            self.current_step_rewards = None
        return None
    
    def process_action(self, agent: Agent):
        """
        Process agent action (heading control only)
        We use the first action dimension as heading angle and ignore the second
        Actions come in normalized to [-1, 1], we scale to [-π, π]
        Agents always move at maximum speed in the direction specified
        
        Args:
            agent: The agent whose action to process
        """
        # Get batch size and device
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        # Check if this attacker is already sensed or reached target (becomes inactive)
        if not agent.is_defender:
            # Find which attacker this is
            attacker_idx = -1
            attackers = [a for a in self.world.agents if not a.is_defender]
            for idx, a in enumerate(attackers):
                if a == agent:
                    attacker_idx = idx
                    break
            
            # If this attacker is sensed or reached target, it stops moving
            if attacker_idx >= 0 and hasattr(self, 'attacker_sensed'):
                is_inactive = self.attacker_sensed[:, attacker_idx] | self.attacker_reached_target[:, attacker_idx]
                
                if self.fixed_attacker_policy:
                    # Active attackers move down, inactive ones keep their heading but speed will be 0
                    heading = torch.full((batch_size,), -math.pi/2, device=device)  # Always down heading
                else:
                    # For non-fixed policy, use action heading (speed will be set to 0 below)
                    if agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
                        normalized_heading = agent.action.u[:, 0]
                        heading = normalized_heading * math.pi  # Use action heading
                    else:
                        heading = torch.zeros(batch_size, device=device)
            else:
                # Fallback if tracking not initialized
                if self.fixed_attacker_policy:
                    heading = torch.full((batch_size,), -math.pi/2, device=device)
                else:
                    heading = torch.zeros(batch_size, device=device)
        else:
            # Defender processing
            # Use the first action dimension as heading angle, ignore second dimension
            # Action shape is (batch_size, 2) but we only use first column
            # Actions are normalized to [-1, 1], scale to [-π, π]
            if agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
                normalized_heading = agent.action.u[:, 0]  # In range [-1, 1]
                heading = normalized_heading * math.pi  # Scale to [-π, π]
            else:
                # Default to no movement
                heading = torch.zeros(batch_size, device=device)
        
        # Convert heading to velocity
        max_speed = self.attacker_max_speed if not agent.is_defender else self.defender_max_speed
        
        # For defenders, check if they should be inactive
        if agent.is_defender and hasattr(self, 'defender_has_sensed'):
            # Find defender index again (could optimize by storing earlier)
            defender_idx = -1
            defenders = [a for a in self.world.agents if a.is_defender]
            for idx, a in enumerate(defenders):
                if a == agent:
                    defender_idx = idx
                    break
            
            if defender_idx >= 0:
                is_inactive = self.defender_has_sensed[:, defender_idx]
                # Set speed to 0 for inactive defenders
                max_speed = torch.where(
                    is_inactive,
                    torch.zeros(batch_size, device=device),
                    torch.full((batch_size,), max_speed, device=device)
                )
        
        # For attackers, check if they should be inactive
        elif not agent.is_defender and hasattr(self, 'attacker_sensed'):
            # Find attacker index again (could optimize by storing earlier)
            attacker_idx = -1
            attackers = [a for a in self.world.agents if not a.is_defender]
            for idx, a in enumerate(attackers):
                if a == agent:
                    attacker_idx = idx
                    break
            
            if attacker_idx >= 0:
                is_inactive = self.attacker_sensed[:, attacker_idx] | self.attacker_reached_target[:, attacker_idx]
                # Set speed to 0 for inactive attackers
                max_speed = torch.where(
                    is_inactive, 
                    torch.zeros(batch_size, device=device), 
                    torch.full((batch_size,), max_speed, device=device)
                )
        
        # Override the action with computed velocity
        # This bypasses the default dynamics processing
        if agent.action is not None and hasattr(agent.action, 'u'):
            agent.action.u[:, 0] = max_speed * torch.cos(heading)
            agent.action.u[:, 1] = max_speed * torch.sin(heading)
    
    def done(self) -> torch.Tensor:
        """
        Check if episodes are done
        
        Returns:
            Boolean tensor indicating which episodes are complete
        """
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
            device = self.world.device if hasattr(self.world, 'device') else self.device
            return torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Episode is done when ALL attackers are either sensed or reached target
        # Each attacker must reach a terminal state (sensed OR reached target)
        all_attackers_done = (self.attacker_sensed | self.attacker_reached_target).all(dim=1)
        return all_attackers_done
    
    def info(self, agent: Agent) -> Dict:
        """
        Get info dictionary for an agent
        
        Args:
            agent: The agent to get info for
            
        Returns:
            Dictionary with episode information
        """
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
            device = self.world.device if hasattr(self.world, 'device') else self.device
            return {
                "attackers_sensed": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attackers_reached_target": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attacker_rewards": torch.zeros((batch_size, self.num_attackers), device=device),
                # Aggregate info for backward compatibility
                "sensing_occurred": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "target_reached": torch.zeros(batch_size, dtype=torch.bool, device=device)
            }
        
        return {
            "attackers_sensed": self.attacker_sensed.clone(),
            "attackers_reached_target": self.attacker_reached_target.clone(),
            "attacker_rewards": self.attacker_sensing_rewards.clone(),
            # Aggregate info for backward compatibility
            "sensing_occurred": self.attacker_sensed.any(dim=1),
            "target_reached": self.attacker_reached_target.any(dim=1)
        }


if __name__ == "__main__":
    # Test the scenario
    import vmas
    
    # Create environment with variable agent numbers
    scenario = Scenario()
    
    # Test with default configuration (3 defenders, 1 attacker)
    env = vmas.make_env(
        scenario=scenario,
        num_envs=4,  # Vectorized batch of 4 environments
        device="cpu",
        continuous_actions=True,
        num_defenders=3,
        num_attackers=1,
        randomize_attacker_x=True
    )
    
    print(f"Environment created with {env.n_agents} agents")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("\nAction format:")
    print("  - Shape: (batch_size, 2)")
    print("  - Dim 0: heading angle NORMALIZED to [-1, 1] (maps to [-π, π] radians)")
    print("  - Dim 1: ignored (set to 0)")
    print("  - Example: 0.5 = π/2 radians (90°), -0.5 = -π/2 radians (-90°)")
    
    # Reset and test
    obs = env.reset()
    print(f"\nInitial observations shape: {[o.shape for o in obs]}")
    
    # Test with different agent numbers
    print("\n" + "="*50)
    print("Testing with 5 defenders and 2 attackers:")
    
    env2 = vmas.make_env(
        scenario=scenario,
        num_envs=2,
        device="cpu",
        continuous_actions=True,
        num_defenders=5,
        num_attackers=2
    )
    
    print(f"Environment created with {env2.n_agents} agents")
    obs2 = env2.reset()
    print(f"Observations shape: {[o.shape for o in obs2]}")
    
    # Test step with heading control
    actions = []
    for i, agent in enumerate(env2.agents):
        if agent.is_defender:
            # Defenders move up (0.5 normalized = π/2 radians)
            action = torch.tensor([[0.5, 0], [0.5, 0]])  # Shape: (num_envs, 2)
        else:
            # Attackers: action provided but overridden by fixed policy
            action = torch.tensor([[-0.5, 0], [-0.5, 0]])  # -0.5 = -π/2 radians
        actions.append(action)
    
    obs, rewards, dones, info = env2.step(actions)
    print(f"\nAfter step:")
    print(f"Rewards shape: {[r.shape for r in rewards]}")
    print(f"Done: {dones}")
    print("\n✓ Environment working correctly with heading-only control!")
    
    # Optional: Interactive rendering (requires pygame)
    # render_interactively(__file__, control_two_agents=False)