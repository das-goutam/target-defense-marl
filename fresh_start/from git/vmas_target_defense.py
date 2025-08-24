"""
Target Defense Environment in VMAS
Agents control only heading (direction) via first action dimension and always move at maximum speed
Variable number of attackers and defenders with sensing-based observations
"""

import torch
import numpy as np
from typing import Dict, List, Optional
import math
from dataclasses import dataclass, MISSING

@dataclass
class TaskConfig:
    max_steps: int = 200
    num_defenders: int = 3
    num_attackers: int = 1
    sensing_radius: float = 0.1
    attacker_sensing_radius: float = 0.1
    speed_ratio: float = 0.7
    target_distance: float = 0.05
    randomize_attacker_x: bool = True
    num_spawn_positions: int = 3
    enable_wall_constraints: bool = True
    wall_epsilon: float = 0.03
    fixed_attacker_policy: bool = True
    use_apollonius: bool = True

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
        num_spawn_positions = kwargs.get('num_spawn_positions', 3)  # Default 3 spawn positions on top edge
        max_steps = kwargs.get('max_steps', 200)  # Default max steps
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
        self.max_steps = max_steps
        
        # Speed settings
        self.defender_max_speed = 0.05
        self.attacker_max_speed = self.defender_max_speed * self.speed_ratio
        
        # Near-wall constraint controls
        self.enable_wall_constraints = bool(kwargs.get("enable_wall_constraints", True))
        self.wall_epsilon = float(kwargs.get("wall_epsilon", 0.03))  # distance from wall to start clamping
        
        # Apollonius solver controls
        self.use_apollonius = bool(kwargs.get("use_apollonius", True)) and APOLLONIUS_AVAILABLE
        
        # Create world - 1x1 space from 0 to 1
        world = World(
            batch_dim=batch_dim,
            device=device,
            x_semidim=0.5,  # World goes from -0.5 to 0.5 (will offset positions)
            y_semidim=0.5,  # World goes from -0.5 to 0.5 (will offset positions)
            collision_force=0,  # No collisions
            substeps=1,
            dt=1.0  # Unit timestep
        )
        
        # Store world bounds for coordinate transformation
        self.world_min = 0.0
        self.world_max = 1.0
        
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
        self.attacker_intercepted = None  # Will be shape (batch_dim, num_attackers) - based on Apollonius payoff
        self.attacker_reached_target = None  # Will be shape (batch_dim, num_attackers)
        self.attacker_sensing_rewards = None  # Will be shape (batch_dim, num_attackers)
        
        # Initialize them here for immediate use
        self.attacker_sensed = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_intercepted = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_reached_target = torch.zeros((batch_dim, num_attackers), dtype=torch.bool, device=device)
        self.attacker_sensing_rewards = torch.zeros((batch_dim, num_attackers), device=device)
        self.defender_has_sensed = torch.zeros((batch_dim, num_defenders), dtype=torch.bool, device=device)
        
        # Step tracking for max_steps termination
        self.step_count = torch.zeros(batch_dim, dtype=torch.long, device=device)
        
        return world
    
    def _world_to_vmas(self, coord):
        """Convert world coordinates [0,1] to VMAS coordinates [-0.5,0.5]"""
        if isinstance(coord, np.ndarray):
            return coord - 0.5
        return coord - 0.5
    
    def _vmas_to_world(self, coord):
        """Convert VMAS coordinates [-0.5,0.5] to world coordinates [0,1]"""
        if isinstance(coord, np.ndarray):
            return coord + 0.5
        return coord + 0.5
    
    def reset_world_at(self, env_index: Optional[int] = None):
        """
        Reset world to initial positions
        
        Args:
            env_index: Index of the environment to reset (for vectorized envs)
        """
        # Get defenders and attackers
        defenders = [a for a in self.world.agents if a.is_defender]
        attackers = [a for a in self.world.agents if not a.is_defender]
        
        # Position defenders evenly along bottom edge (world coordinates [0,1])
        defender_spacing = 1.0 / (self.num_defenders + 1)
        for i, defender in enumerate(defenders):
            world_x = (i + 1) * defender_spacing  # x position in [0,1] world coordinates
            vmas_x = self._world_to_vmas(world_x)  # Convert to VMAS [-0.5, 0.5]
            vmas_y = self._world_to_vmas(0.0)      # Bottom edge = y=0 in world = y=-0.5 in VMAS
            
            if env_index is None:
                defender.state.pos[:, X] = vmas_x
                defender.state.pos[:, Y] = vmas_y
                defender.state.vel[:, :] = 0
            else:
                defender.state.pos[env_index, X] = vmas_x
                defender.state.pos[env_index, Y] = vmas_y
                defender.state.vel[env_index, :] = 0
        
        # Create spawn positions once for all attackers (top edge y=1)
        if self.randomize_attacker_x:
            # Create equally spaced spawn positions along top edge
            # For K positions, space them evenly across [0.1, 0.9] in world coordinates
            if self.num_spawn_positions == 1:
                world_spawn_positions = torch.tensor([0.5], device=self.device)  # Center
            else:
                # Equally spaced positions with 0.1 margin from edges
                spacing = 0.8 / (self.num_spawn_positions - 1)  # Total range is 0.8
                world_spawn_positions = torch.tensor(
                    [0.1 + i * spacing for i in range(self.num_spawn_positions)],
                    device=self.device
                )
            # Convert to VMAS coordinates
            spawn_positions = self._world_to_vmas(world_spawn_positions)
            
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
                world_x = 0.5  # Center in world coordinates
                vmas_x = self._world_to_vmas(world_x)
                if env_index is None:
                    attacker.state.pos[:, X] = vmas_x
                else:
                    attacker.state.pos[env_index, X] = vmas_x
        
        # Set Y position and velocity for all attackers (top edge y=1)
        world_y = 1.0  # Top edge in world coordinates
        vmas_y = self._world_to_vmas(world_y)
        for attacker in attackers:
            if env_index is None:
                attacker.state.pos[:, Y] = vmas_y
                attacker.state.vel[:, :] = 0
            else:
                attacker.state.pos[env_index, Y] = vmas_y
                attacker.state.vel[env_index, :] = 0
        
        # Reset episode tracking - track each attacker separately
        if env_index is None:
            batch_size = self.batch_dim if hasattr(self, 'batch_dim') else self.world.batch_dim
            device = self.device if hasattr(self, 'device') else self.world.device
            self.attacker_sensed = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_intercepted = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_reached_target = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_sensing_rewards = torch.zeros((batch_size, self.num_attackers), device=device)
            self.defender_has_sensed = torch.zeros((batch_size, self.num_defenders), dtype=torch.bool, device=device)
            self.distances = torch.zeros((batch_size, self.num_defenders, self.num_attackers), device=device)
            self.step_count = torch.zeros(batch_size, dtype=torch.long, device=device)
        else:
            self.attacker_sensed[env_index, :] = False
            self.attacker_intercepted[env_index, :] = False
            self.attacker_reached_target[env_index, :] = False
            self.attacker_sensing_rewards[env_index, :] = 0.0
            if hasattr(self, 'defender_has_sensed'):
                self.defender_has_sensed[env_index, :] = False
            if hasattr(self, 'distances'):
                self.distances[env_index, :, :] = 0.0
            if hasattr(self, 'step_count'):
                self.step_count[env_index] = 0
        
        # Reset step-based flags
        self._events_updated_this_step = False
        self._step_incremented_this_step = False
    
    def reward(self, agent: Agent) -> torch.Tensor:
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device

        # ensure events are up-to-date; does NOT mutate if already updated
        self.update_events()

        # Only terminal rewards for defenders
        done_mask = self.done()
        r = torch.zeros(batch_size, device=device)
        
        if agent.is_defender and done_mask.any():
            for env_idx in torch.where(done_mask)[0]:
                total_reward = 0.0
                for a_idx in range(self.num_attackers):
                    if self.attacker_sensed[env_idx, a_idx]:
                        # Defender gets reward for sensing attacker
                        total_reward += self.attacker_sensing_rewards[env_idx, a_idx]
                    # If attacker reached target, defender gets 0 (no penalty, just no reward)
                r[env_idx] = total_reward
        
        # Attackers get no rewards (we're only training defenders)
        return r
    
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
        # Reset step-based flags for next step
        self._events_updated_this_step = False
        self._step_incremented_this_step = False
        return None
    
    def _clamp_interval(self, theta, lo, hi):
        """Clamp angle theta (radians in [0, 2π)) into [lo, hi] elementwise."""
        lo_t = torch.full_like(theta, lo)
        hi_t = torch.full_like(theta, hi)
        theta = torch.where(theta < lo_t, lo_t, theta)
        theta = torch.where(theta > hi_t, hi_t, theta)
        return theta

    def _clamp_union(self, theta, segments):
        """
        Clamp theta (radians in [0, 2π)) to the closest boundary of a union of segments.
        segments: list of (lo, hi) with 0 <= lo < hi <= 2π.
        """
        B = theta.shape[0]
        inside = torch.zeros(B, dtype=torch.bool, device=theta.device)
        # Track nearest boundary across segments
        best_dist = torch.full_like(theta, float("inf"))
        best_proj = theta.clone()
        for lo, hi in segments:
            in_seg = (theta >= lo) & (theta <= hi)
            inside |= in_seg
            d_lo = torch.abs(theta - lo)
            d_hi = torch.abs(theta - hi)
            proj = torch.where(d_lo <= d_hi, torch.full_like(theta, lo), torch.full_like(theta, hi))
            dist = torch.minimum(d_lo, d_hi)
            better = dist < best_dist
            best_proj = torch.where(better, proj, best_proj)
            best_dist = torch.where(better, dist, best_dist)
        # Keep theta if inside any segment; otherwise project to nearest boundary
        return torch.where(inside, theta, best_proj)

    def _apply_wall_constraints(self, agent, theta_0_2pi):
        """
        Apply state-dependent heading constraints near walls/corners for DEFENDERS.
        theta_0_2pi: [B] angle in [0, 2π).
        Returns: clamped theta in [0, 2π).
        """
        if not self.enable_wall_constraints:
            return theta_0_2pi

        x = agent.state.pos[:, 0]  # X in VMAS coordinates [-0.5, 0.5]
        y = agent.state.pos[:, 1]  # Y in VMAS coordinates [-0.5, 0.5]
        wx = getattr(self.world, "x_semidim", 0.5)  # VMAS world bounds
        wy = getattr(self.world, "y_semidim", 0.5)  # VMAS world bounds
        eps = float(self.wall_epsilon)

        near_right  = (wx - x) <= eps
        near_left   = (x + wx) <= eps
        near_top    = (wy - y) <= eps
        near_bottom = (y + wy) <= eps

        theta = theta_0_2pi.clone()

        pi = math.pi
        two_pi = 2 * pi
        # Corner masks (handle first)
        tr = near_top & near_right     # top-right: 180°–270°
        tl = near_top & near_left      # top-left : 270°–360°
        br = near_bottom & near_right  # bot-right:  90°–180°
        bl = near_bottom & near_left   # bot-left :   0°– 90°

        theta = torch.where(tr, self._clamp_interval(theta, pi, 1.5 * pi), theta)
        theta = torch.where(tl, self._clamp_interval(theta, 1.5 * pi, two_pi), theta)
        theta = torch.where(br, self._clamp_interval(theta, 0.5 * pi, pi), theta)
        theta = torch.where(bl, self._clamp_interval(theta, 0.0, 0.5 * pi), theta)

        # Wall-only masks (exclude corners)
        right_only  = near_right  & ~(near_top | near_bottom)   # 90°–270°
        left_only   = near_left   & ~(near_top | near_bottom)   # 270°–360° ∪ 0°–90°
        top_only    = near_top    & ~(near_left | near_right)   # 180°–360°
        bottom_only = near_bottom & ~(near_left | near_right)   # 0°–180°

        # Right wall: single segment
        if right_only.any():
            theta[right_only] = self._clamp_interval(theta[right_only], 0.5 * pi, 1.5 * pi)

        # Left wall: union of two segments
        if left_only.any():
            segs = [(0.0, 0.5 * pi), (1.5 * pi, two_pi)]
            theta[left_only] = self._clamp_union(theta[left_only], segs)

        # Top wall: single segment
        if top_only.any():
            theta[top_only] = self._clamp_interval(theta[top_only], pi, two_pi)

        # Bottom wall: single segment
        if bottom_only.any():
            theta[bottom_only] = self._clamp_interval(theta[bottom_only], 0.0, pi)

        return theta

    def update_events(self):
        """
        Update sensing events and handle position snapping.
        Called once per step to avoid double processing.
        """
        if hasattr(self, '_events_updated_this_step') and self._events_updated_this_step:
            return  # Already updated this step
        
        batch_size = self.world.batch_dim if hasattr(self.world, 'batch_dim') else self.batch_dim
        device = self.world.device if hasattr(self.world, 'device') else self.device
        
        # Initialize tracking variables if not already done
        if not hasattr(self, 'attacker_sensed') or self.attacker_sensed is None:
            self.attacker_sensed = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
            self.attacker_intercepted = torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device)
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
                
                # When sensing occurs, snap attacker to boundary and compute rewards/interception
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
                    
                    # Compute payoff using Apollonius solver and determine interception
                    if self.use_apollonius:
                        for env_idx in torch.where(newly_sensed)[0]:
                            # Get current positions for this environment in VMAS coords
                            attacker_pos_vmas = attacker.state.pos[env_idx].cpu().numpy()
                            # Convert to global frame [0,1] for Apollonius solver
                            attacker_pos = self._vmas_to_world(attacker_pos_vmas)
                            
                            # Get all defender positions for this environment
                            defender_positions = []
                            for def_agent in self.world.agents:
                                if def_agent.is_defender:
                                    defender_pos_vmas = def_agent.state.pos[env_idx].cpu().numpy()
                                    # Convert to global frame [0,1] for Apollonius solver
                                    defender_pos_global = self._vmas_to_world(defender_pos_vmas)
                                    defender_positions.append(defender_pos_global)
                            
                            # Solve Apollonius optimization
                            result = solve_apollonius_optimization(
                                attacker_pos=attacker_pos,
                                defender_positions=defender_positions,
                                nu=1.0 / self.speed_ratio  # nu = defender_speed / attacker_speed
                            )
                            
                            if result['success']:
                                # Use defender payoff from Apollonius solver
                                defender_payoff = result['defender_payoff']
                                
                                # If payoff is positive, mark as intercepted
                                if defender_payoff > 0:
                                    self.attacker_intercepted[env_idx, attacker_idx] = True
                                    # Use the payoff as reward (scaled by 100)
                                    self.attacker_sensing_rewards[env_idx, attacker_idx] = defender_payoff * 100.0
                                else:
                                    # Negative payoff means attacker can still reach target
                                    self.attacker_sensing_rewards[env_idx, attacker_idx] = 0.1  # Small sensing reward
                            else:
                                # Fallback if solver fails - use y-position in world coordinates
                                attacker_y_vmas = attacker.state.pos[env_idx, Y].item()
                                attacker_y_world = self._vmas_to_world(attacker_y_vmas)  # Convert to [0,1]
                                # Use y-position as reward (higher y = closer to spawn = higher reward)
                                self.attacker_sensing_rewards[env_idx, attacker_idx] = attacker_y_world
                    else:
                        # Default reward if Apollonius not available - use y-position
                        for env_idx in torch.where(newly_sensed)[0]:
                            attacker_y_vmas = attacker.state.pos[env_idx, Y].item()
                            attacker_y_world = self._vmas_to_world(attacker_y_vmas)  # Convert to [0,1]
                            # Use y-position as reward (higher y = closer to spawn = higher reward)
                            self.attacker_sensing_rewards[env_idx, attacker_idx] = attacker_y_world
                
                self.attacker_sensed[:, attacker_idx] |= newly_sensed
        
        # Check for target reached for EACH attacker (only if not sensed)
        # Target is at bottom edge (y=0 in world, y=-0.5 in VMAS)
        target_y_vmas = self._world_to_vmas(0.0)  # Bottom edge
        for attacker_idx, attacker in enumerate(attackers):
            reached = (attacker.state.pos[:, Y] <= target_y_vmas + self.target_distance) & ~self.attacker_sensed[:, attacker_idx]
            self.attacker_reached_target[:, attacker_idx] |= reached
        
        # Mark that events have been updated this step
        self._events_updated_this_step = True
        
        # Reset flag for next step (in case extra_render is not called)
        # We'll reset this flag when the first agent processes its action next step
        self._reset_events_flag_next_step = True

    def process_action(self, agent: Agent):
        """
        Process agent action (heading control only)
        We use the first action dimension as heading angle and ignore the second
        Actions come in normalized to [-1, 1], we scale to [-π, π]
        Agents always move at maximum speed in the direction specified
        
        Args:
            agent: The agent whose action to process
        """
        # Reset events flag for new step if needed
        if hasattr(self, '_reset_events_flag_next_step') and self._reset_events_flag_next_step:
            self._events_updated_this_step = False
            self._step_incremented_this_step = False
            self._reset_events_flag_next_step = False
        
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
            # Actions are normalized to [-1, 1], scale to [-π, π]
            if agent.action is not None and hasattr(agent.action, 'u') and agent.action.u is not None:
                normalized_heading = agent.action.u[:, 0]  # In range [-1, 1]
                heading = normalized_heading * math.pi      # [-π, π]
            else:
                heading = torch.zeros(batch_size, device=device)

        # Convert to [0, 2π) for clamping
        theta = torch.remainder(heading, 2 * math.pi)

        # >>> ADD THIS: state-dependent angle clamping for defenders <<<
        if agent.is_defender:
            theta = self._apply_wall_constraints(agent, theta)
        # <<< END ADD >>>
        
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
        
        # Update sensing events once per step (before velocity computation)
        self.update_events()
        
        # Increment step count (only once per step for the first agent processed)
        if not hasattr(self, '_step_incremented_this_step') or not self._step_incremented_this_step:
            if hasattr(self, 'step_count'):
                self.step_count += 1
            self._step_incremented_this_step = True
        
        # Override the action with computed velocity (use theta, not heading)
        # This bypasses the default dynamics processing
        if agent.action is not None and hasattr(agent.action, 'u'):
            agent.action.u[:, 0] = max_speed * torch.cos(theta)
            agent.action.u[:, 1] = max_speed * torch.sin(theta)
    
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
        
        # OR when max_steps is reached
        max_steps_reached = torch.zeros_like(all_attackers_done, dtype=torch.bool)
        if hasattr(self, 'step_count'):
            max_steps_reached = self.step_count >= self.max_steps
        
        return all_attackers_done | max_steps_reached
    
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
                "attackers_intercepted": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attackers_reached_target": torch.zeros((batch_size, self.num_attackers), dtype=torch.bool, device=device),
                "attacker_rewards": torch.zeros((batch_size, self.num_attackers), device=device),
                # Aggregate info for backward compatibility
                "sensing_occurred": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "interception_occurred": torch.zeros(batch_size, dtype=torch.bool, device=device),
                "target_reached": torch.zeros(batch_size, dtype=torch.bool, device=device)
            }
        
        return {
            "attackers_sensed": self.attacker_sensed.clone(),
            "attackers_intercepted": self.attacker_intercepted.clone(),
            "attackers_reached_target": self.attacker_reached_target.clone(),
            "attacker_rewards": self.attacker_sensing_rewards.clone(),
            # Aggregate info for backward compatibility
            "sensing_occurred": self.attacker_sensed.any(dim=1),
            "interception_occurred": self.attacker_intercepted.any(dim=1),
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
    
    # Debug positions in world coordinates [0,1]
    print("\nDebug: Agent positions in world coordinates [0,1]:")
    for i, agent in enumerate(env.agents):
        vmas_pos = agent.state.pos[0]  # First environment
        world_pos = scenario._vmas_to_world(vmas_pos)
        print(f"  {agent.name}: VMAS {vmas_pos.tolist()} -> World {world_pos.tolist()}")
    
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