"""
Complete RL Training Setup for Target Defense Environment
File: rl_train_target_defense.py

This provides integration points for real RL algorithms (PPO, SAC, QMIX, etc.)
and shows how to structure your training for research/production use.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# VMAS imports
import vmas
from vmas_target_defense import Scenario


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # Environment
    num_defenders: int = 3
    num_attackers: int = 1
    sensing_radius: float = 0.15
    speed_ratio: float = 0.7
    max_steps: int = 200
    
    # Training
    num_envs: int = 32
    num_episodes: int = 1000
    learning_rate: float = 3e-4
    gamma: float = 0.99
    device: str = "cpu"
    
    # Algorithm specific
    algorithm: str = "ppo"  # Options: "ppo", "sac", "qmix", "simple"
    hidden_size: int = 256
    num_layers: int = 3
    
    # Logging
    log_interval: int = 10
    save_interval: int = 100
    save_path: str = "./checkpoints"


class SimpleDefenderPolicy(nn.Module):
    """Simple neural network policy for defenders."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int = 256, num_layers: int = 2):
        super().__init__()
        
        layers = []
        input_size = obs_dim
        
        for i in range(num_layers):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size
        
        layers.append(nn.Linear(hidden_size, action_dim))
        layers.append(nn.Tanh())  # Actions in [-1, 1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(obs)


class TargetDefenseRLTrainer:
    """RL Trainer for Target Defense environment."""
    
    def __init__(self, config: TrainingConfig):
        """Initialize trainer with configuration."""
        self.config = config
        self.device = torch.device(config.device)
        
        # Create environment
        self.env = self.create_environment()
        
        # Get dimensions
        obs = self.env.reset()
        self.obs_dim = obs[0].shape[-1]  # Observation dimension
        self.action_dim = 1  # Only using first action dimension (heading)
        self.num_agents = self.env.n_agents
        
        # Identify defenders and attackers
        self.defenders = [a for a in self.env.agents if a.is_defender]
        self.attackers = [a for a in self.env.agents if not a.is_defender]
        
        # Create policies for defenders
        self.policies = {}
        self.optimizers = {}
        
        for defender in self.defenders:
            policy = SimpleDefenderPolicy(
                self.obs_dim, 
                self.action_dim,
                config.hidden_size,
                config.num_layers
            ).to(self.device)
            
            self.policies[defender.name] = policy
            self.optimizers[defender.name] = optim.Adam(
                policy.parameters(), 
                lr=config.learning_rate
            )
        
        # Training statistics
        self.episode_rewards = []
        self.sensing_rates = []
        self.target_rates = []
    
    def create_environment(self):
        """Create VMAS environment."""
        return vmas.make_env(
            scenario=Scenario(),
            num_envs=self.config.num_envs,
            device=self.config.device,
            continuous_actions=True,
            num_defenders=self.config.num_defenders,
            num_attackers=self.config.num_attackers,
            sensing_radius=self.config.sensing_radius,
            speed_ratio=self.config.speed_ratio,
            max_steps=self.config.max_steps,
            fixed_attacker_policy=True
        )
    
    def get_actions(self, observations: List[torch.Tensor], explore: bool = True) -> List[torch.Tensor]:
        """Get actions from policies."""
        actions = []
        
        for i, agent in enumerate(self.env.agents):
            if agent.is_defender:
                # Use neural network policy
                obs = observations[i]
                
                with torch.no_grad():
                    action_values = self.policies[agent.name](obs)
                
                # Add exploration noise if training
                if explore:
                    noise = torch.randn_like(action_values) * 0.1
                    action_values = action_values + noise
                    action_values = torch.clamp(action_values, -1, 1)
                
                # Create 2D action (second dimension ignored by environment)
                action = torch.zeros((self.config.num_envs, 2), device=self.device)
                action[:, 0] = action_values.squeeze()
                action[:, 1] = 0
            else:
                # Attacker uses fixed policy
                action = torch.ones((self.config.num_envs, 2), device=self.device) * -0.5
                action[:, 1] = 0
            
            actions.append(action)
        
        return actions
    
    def train_episode(self) -> Tuple[float, float, float]:
        """Run one training episode."""
        obs = self.env.reset()
        done = torch.zeros(self.config.num_envs, dtype=torch.bool, device=self.device)
        
        episode_rewards = {agent.name: [] for agent in self.env.agents}
        trajectory = {agent.name: {"obs": [], "actions": [], "rewards": []} 
                     for agent in self.defenders}
        
        steps = 0
        
        while not done.all() and steps < self.config.max_steps:
            # Get actions
            actions = self.get_actions(obs, explore=True)
            
            # Store observations and actions for defenders
            for i, agent in enumerate(self.env.agents):
                if agent.is_defender:
                    trajectory[agent.name]["obs"].append(obs[i])
                    trajectory[agent.name]["actions"].append(actions[i][:, 0])
            
            # Step environment
            obs, rewards, dones, info = self.env.step(actions)
            
            # Store rewards
            for i, agent in enumerate(self.env.agents):
                episode_rewards[agent.name].append(rewards[i])
                if agent.is_defender:
                    trajectory[agent.name]["rewards"].append(rewards[i])
            
            done = done | dones
            steps += 1
        
        # Compute returns and train
        self.update_policies(trajectory)
        
        # Calculate statistics
        defender_total_rewards = []
        for name, rewards_list in episode_rewards.items():
            if "defender" in name and rewards_list:
                total = torch.stack(rewards_list).sum(0).mean().item()
                defender_total_rewards.append(total)
        
        avg_reward = np.mean(defender_total_rewards) if defender_total_rewards else 0
        
        # Get termination statistics
        sensing_rate = 0
        target_rate = 0
        
        if hasattr(self.env.scenario, 'sensing_occurred'):
            sensing_rate = self.env.scenario.sensing_occurred.float().mean().item()
        
        if hasattr(self.env.scenario, 'target_reached'):
            target_rate = self.env.scenario.target_reached.float().mean().item()
        
        return avg_reward, sensing_rate, target_rate
    
    def update_policies(self, trajectory: Dict):
        """Update policies using collected trajectory (simplified PPO-style update)."""
        
        for defender_name in self.policies.keys():
            if defender_name not in trajectory:
                continue
            
            traj = trajectory[defender_name]
            
            if not traj["rewards"]:
                continue
            
            # Compute returns (discounted cumulative rewards)
            returns = []
            running_return = torch.zeros(self.config.num_envs, device=self.device)
            
            for reward in reversed(traj["rewards"]):
                running_return = reward + self.config.gamma * running_return
                returns.insert(0, running_return.clone())
            
            if not returns:
                continue
            
            # Convert to tensors
            obs_batch = torch.stack(traj["obs"])
            action_batch = torch.stack(traj["actions"])
            returns_batch = torch.stack(returns)
            
            # Normalize returns
            returns_batch = (returns_batch - returns_batch.mean()) / (returns_batch.std() + 1e-8)
            
            # Compute loss (simplified policy gradient)
            predicted_actions = self.policies[defender_name](obs_batch.reshape(-1, self.obs_dim))
            predicted_actions = predicted_actions.reshape(obs_batch.shape[0], self.config.num_envs, -1)
            
            # MSE loss between predicted and taken actions, weighted by returns
            action_loss = ((predicted_actions.squeeze(-1) - action_batch) ** 2)
            weighted_loss = (action_loss * returns_batch.detach()).mean()
            
            # Update policy
            self.optimizers[defender_name].zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policies[defender_name].parameters(), 0.5)
            self.optimizers[defender_name].step()
    
    def train(self):
        """Main training loop."""
        print("\n" + "="*60)
        print(f"TRAINING WITH {self.config.algorithm.upper()}")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  - Defenders: {self.config.num_defenders}")
        print(f"  - Attackers: {self.config.num_attackers}")
        print(f"  - Parallel envs: {self.config.num_envs}")
        print(f"  - Episodes: {self.config.num_episodes}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Network: {self.config.num_layers} layers × {self.config.hidden_size} units")
        
        print("\nStarting training...")
        print("-"*40)
        
        for episode in range(self.config.num_episodes):
            # Train one episode
            avg_reward, sensing_rate, target_rate = self.train_episode()
            
            # Store statistics
            self.episode_rewards.append(avg_reward)
            self.sensing_rates.append(sensing_rate)
            self.target_rates.append(target_rate)
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                recent_rewards = np.mean(self.episode_rewards[-self.config.log_interval:])
                recent_sensing = np.mean(self.sensing_rates[-self.config.log_interval:])
                recent_target = np.mean(self.target_rates[-self.config.log_interval:])
                
                print(f"Episode {episode+1}/{self.config.num_episodes}:")
                print(f"  Avg Reward: {recent_rewards:.3f}")
                print(f"  Sensing Rate: {recent_sensing:.2%}")
                print(f"  Target Rate: {recent_target:.2%}")
            
            # Save checkpoints
            if (episode + 1) % self.config.save_interval == 0:
                self.save_checkpoint(episode + 1)
        
        # Training complete
        print("\n" + "="*60)
        print("TRAINING COMPLETE")
        print("="*60)
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print final training statistics."""
        window = min(50, len(self.episode_rewards) // 5)
        
        print(f"\nFinal Statistics (last {window} episodes):")
        print(f"  Average Reward: {np.mean(self.episode_rewards[-window:]):.3f}")
        print(f"  Sensing Success Rate: {np.mean(self.sensing_rates[-window:]):.2%}")
        print(f"  Target Reached Rate: {np.mean(self.target_rates[-window:]):.2%}")
        
        # Performance assessment
        final_sensing = np.mean(self.sensing_rates[-window:])
        
        print("\nPerformance Assessment:")
        if final_sensing > 0.8:
            print("  🏆 EXCELLENT: Defenders intercept >80% of attackers!")
        elif final_sensing > 0.6:
            print("  ✅ GOOD: Defenders intercept >60% of attackers")
        elif final_sensing > 0.4:
            print("  ⚠️  MODERATE: Defenders intercept >40% of attackers")
        else:
            print("  ❌ NEEDS IMPROVEMENT: Defenders intercept <40% of attackers")
        
        # Improvement analysis
        early_reward = np.mean(self.episode_rewards[:window])
        late_reward = np.mean(self.episode_rewards[-window:])
        improvement = ((late_reward - early_reward) / abs(early_reward)) * 100 if early_reward != 0 else 0
        
        print(f"\nLearning Progress:")
        print(f"  Early performance: {early_reward:.3f}")
        print(f"  Final performance: {late_reward:.3f}")
        print(f"  Improvement: {improvement:+.1f}%")
    
    def save_checkpoint(self, episode: int):
        """Save model checkpoint."""
        import os
        os.makedirs(self.config.save_path, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'config': self.config,
            'policies': {name: policy.state_dict() 
                        for name, policy in self.policies.items()},
            'optimizers': {name: opt.state_dict() 
                          for name, opt in self.optimizers.items()},
            'statistics': {
                'rewards': self.episode_rewards,
                'sensing_rates': self.sensing_rates,
                'target_rates': self.target_rates
            }
        }
        
        path = f"{self.config.save_path}/checkpoint_ep{episode}.pt"
        torch.save(checkpoint, path)
        print(f"    💾 Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        for name, state_dict in checkpoint['policies'].items():
            if name in self.policies:
                self.policies[name].load_state_dict(state_dict)
        
        for name, state_dict in checkpoint['optimizers'].items():
            if name in self.optimizers:
                self.optimizers[name].load_state_dict(state_dict)
        
        self.episode_rewards = checkpoint['statistics']['rewards']
        self.sensing_rates = checkpoint['statistics']['sensing_rates']
        self.target_rates = checkpoint['statistics']['target_rates']
        
        print(f"✓ Checkpoint loaded from episode {checkpoint['episode']}")
    
    def evaluate(self, num_episodes: int = 10) -> Dict[str, float]:
        """Evaluate current policies."""
        print("\nEvaluating...")
        
        eval_rewards = []
        eval_sensing = []
        eval_target = []
        
        for _ in range(num_episodes):
            obs = self.env.reset()
            done = torch.zeros(self.config.num_envs, dtype=torch.bool, device=self.device)
            episode_reward = 0
            steps = 0
            
            while not done.all() and steps < self.config.max_steps:
                # Get actions without exploration
                actions = self.get_actions(obs, explore=False)
                
                # Step
                obs, rewards, dones, info = self.env.step(actions)
                
                # Track rewards
                for i, agent in enumerate(self.env.agents):
                    if agent.is_defender:
                        episode_reward += rewards[i].mean().item()
                
                done = done | dones
                steps += 1
            
            eval_rewards.append(episode_reward / len(self.defenders))
            
            if hasattr(self.env.scenario, 'sensing_occurred'):
                eval_sensing.append(self.env.scenario.sensing_occurred.float().mean().item())
            
            if hasattr(self.env.scenario, 'target_reached'):
                eval_target.append(self.env.scenario.target_reached.float().mean().item())
        
        results = {
            'avg_reward': np.mean(eval_rewards),
            'sensing_rate': np.mean(eval_sensing) if eval_sensing else 0,
            'target_rate': np.mean(eval_target) if eval_target else 0
        }
        
        print(f"Evaluation Results ({num_episodes} episodes):")
        print(f"  Average Reward: {results['avg_reward']:.3f}")
        print(f"  Sensing Rate: {results['sensing_rate']:.2%}")
        print(f"  Target Rate: {results['target_rate']:.2%}")
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Training for Target Defense")
    
    # Environment arguments
    parser.add_argument("--defenders", type=int, default=3, help="Number of defenders")
    parser.add_argument("--attackers", type=int, default=1, help="Number of attackers")
    parser.add_argument("--sensing-radius", type=float, default=0.15, help="Sensing radius")
    parser.add_argument("--speed-ratio", type=float, default=0.7, help="Speed ratio")
    
    # Training arguments
    parser.add_argument("--episodes", type=int, default=500, help="Number of episodes")
    parser.add_argument("--envs", type=int, default=32, help="Parallel environments")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=3, help="Number of hidden layers")
    
    # Other arguments
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--algorithm", type=str, default="ppo", help="Algorithm")
    parser.add_argument("--load", type=str, default=None, help="Load checkpoint")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        sensing_radius=args.sensing_radius,
        speed_ratio=args.speed_ratio,
        num_envs=args.envs,
        num_episodes=args.episodes,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        device=args.device,
        algorithm=args.algorithm
    )
    
    # Create trainer
    trainer = TargetDefenseRLTrainer(config)
    
    # Load checkpoint if specified
    if args.load:
        trainer.load_checkpoint(args.load)
    
    # Evaluate or train
    if args.evaluate:
        trainer.evaluate(num_episodes=20)
    else:
        trainer.train()
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        trainer.evaluate(num_episodes=20)
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
    