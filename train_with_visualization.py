"""
Enhanced Training Script with Visualization for Target Defense
File: train_with_visualization.py

This script provides extended training with:
- Periodic trajectory visualization (GIF/PNG)
- Training metrics plotting
- Progress saving
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, Rectangle, FancyArrowPatch
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# VMAS imports
import vmas
from vmas_target_defense import Scenario

# Import the RL trainer
from rl_train_target_defense import TargetDefenseRLTrainer, TrainingConfig, SimpleDefenderPolicy
# Note: The parent class TargetDefenseRLTrainer now uses randomize_attacker_x=True


class VisualizationTrainer(TargetDefenseRLTrainer):
    """Extended trainer with visualization capabilities."""
    
    def __init__(self, config: TrainingConfig, save_dir: str = "./visualizations"):
        """Initialize with visualization support."""
        super().__init__(config)
        
        # Create save directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_dir = f"{save_dir}/run_{timestamp}"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(f"{self.save_dir}/trajectories", exist_ok=True)
        os.makedirs(f"{self.save_dir}/metrics", exist_ok=True)
        
        # For storing trajectory data
        self.trajectory_buffer = []
        self.metrics_history = {
            'episode': [],
            'reward': [],
            'interception_rate': [],
            'sensing_rate': [],
            'avg_episode_length': []
        }
        
        print(f"\n📁 Saving visualizations to: {self.save_dir}")
    
    def collect_trajectory(self, episode: int) -> Dict:
        """Collect a single trajectory for visualization."""
        obs = self.env.reset()
        done = torch.zeros(self.config.num_envs, dtype=torch.bool, device=self.device)
        
        trajectory = {
            'episode': episode,
            'positions': {'defenders': [], 'attackers': []},
            'sensing_events': [],
            'target_events': []
        }
        
        steps = 0
        while not done[0] and steps < self.config.max_steps:  # Check only first env
            # Store positions (only first env)
            defender_pos = []
            attacker_pos = []
            
            for i, agent in enumerate(self.env.agents):
                pos = agent.state.pos[0].cpu().numpy()  # First env only
                if agent.is_defender:
                    defender_pos.append(pos)
                else:
                    attacker_pos.append(pos)
            
            trajectory['positions']['defenders'].append(defender_pos)
            trajectory['positions']['attackers'].append(attacker_pos)
            
            # Get actions for all environments (but we'll only visualize the first)
            actions = []
            for i, agent in enumerate(self.env.agents):
                if agent.is_defender:
                    with torch.no_grad():
                        action_values = self.policies[agent.name](obs[i])
                    action = torch.zeros((self.config.num_envs, 2), device=self.device)
                    # Handle different output shapes
                    if action_values.dim() == 1:
                        action[:, 0] = action_values
                    else:
                        action[:, 0] = action_values.squeeze(-1)
                else:
                    action = torch.zeros((self.config.num_envs, 2), device=self.device)
                    action[:, 0] = -0.5
                action[:, 1] = 0
                actions.append(action)
            
            obs, rewards, dones, info = self.env.step(actions)
            
            # Check for sensing/target events (first env only)
            if hasattr(self.env.scenario, 'sensing_occurred'):
                if self.env.scenario.sensing_occurred[0]:
                    trajectory['sensing_events'].append(steps)
            
            if hasattr(self.env.scenario, 'target_reached'):
                if self.env.scenario.target_reached[0]:
                    trajectory['target_events'].append(steps)
            
            done = done | dones
            steps += 1
        
        return trajectory
    
    def create_trajectory_animation(self, trajectory: Dict, save_path: str):
        """Create an animated GIF of the trajectory."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Set up the plot
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.6, 0.6)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_title(f'Episode {trajectory["episode"]} - Target Defense Trajectory')
        
        # Add target line
        ax.axhline(y=-0.5, color='red', linestyle='--', linewidth=2, label='Target Line')
        
        # Add potential attacker spawn positions (generalized)
        num_spawn = self.config.num_spawn_positions  # Direct access, no default
        if num_spawn == 1:
            spawn_positions = [0.0]
        else:
            spacing = 0.8 / (num_spawn - 1)
            spawn_positions = [-0.4 + i * spacing for i in range(num_spawn)]
        
        spawn_y = 0.5
        for i, x_pos in enumerate(spawn_positions):
            ax.plot(x_pos, spawn_y, 'r^', markersize=15, alpha=0.3, 
                   label='Potential Spawn' if i == 0 else "")
            ax.text(x_pos, spawn_y + 0.05, f'S{i+1}', 
                   ha='center', fontsize=8, color='red', alpha=0.7)
        
        # Initialize sensing radius circles (will be updated each frame)
        sensing_circles = []
        for _ in range(self.config.num_defenders):
            circle = Circle((0, 0), self.config.sensing_radius, 
                          fill=False, edgecolor='blue', alpha=0.3, linestyle='--')
            ax.add_patch(circle)
            sensing_circles.append(circle)
        
        # Initialize agent markers
        defender_markers = []
        attacker_markers = []
        
        for i in range(self.config.num_defenders):
            marker, = ax.plot([], [], 'bo', markersize=10, label=f'Defender {i+1}' if i == 0 else "")
            defender_markers.append(marker)
        
        for i in range(self.config.num_attackers):
            marker, = ax.plot([], [], 'r^', markersize=12, label=f'Attacker {i+1}' if i == 0 else "")
            attacker_markers.append(marker)
        
        # Add trails
        defender_trails = [ax.plot([], [], 'b-', alpha=0.3, linewidth=1)[0] 
                          for _ in range(self.config.num_defenders)]
        attacker_trails = [ax.plot([], [], 'r-', alpha=0.3, linewidth=1)[0] 
                          for _ in range(self.config.num_attackers)]
        
        # Text for events
        event_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, 
                            verticalalignment='top', fontsize=10,
                            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        def animate(frame):
            # Update defender positions and sensing circles
            for i, (marker, trail, circle) in enumerate(zip(defender_markers, defender_trails, sensing_circles)):
                if frame < len(trajectory['positions']['defenders']):
                    pos = trajectory['positions']['defenders'][frame][i]
                    marker.set_data([pos[0]], [pos[1]])
                    
                    # Update sensing circle position
                    circle.center = (pos[0], pos[1])
                    
                    # Update trail
                    trail_x = [p[i][0] for p in trajectory['positions']['defenders'][:frame+1]]
                    trail_y = [p[i][1] for p in trajectory['positions']['defenders'][:frame+1]]
                    trail.set_data(trail_x, trail_y)
            
            # Update attacker positions
            for i, (marker, trail) in enumerate(zip(attacker_markers, attacker_trails)):
                if frame < len(trajectory['positions']['attackers']) and i < len(trajectory['positions']['attackers'][frame]):
                    pos = trajectory['positions']['attackers'][frame][i]
                    marker.set_data([pos[0]], [pos[1]])
                    
                    # Update trail
                    trail_x = [p[i][0] for p in trajectory['positions']['attackers'][:frame+1] if i < len(p)]
                    trail_y = [p[i][1] for p in trajectory['positions']['attackers'][:frame+1] if i < len(p)]
                    trail.set_data(trail_x, trail_y)
            
            # Update event text
            event_str = f"Step: {frame}/{len(trajectory['positions']['defenders'])}"
            if frame in trajectory['sensing_events']:
                event_str += "\n🎯 SENSING OCCURRED!"
            if frame in trajectory['target_events']:
                event_str += "\n❌ TARGET REACHED!"
            event_text.set_text(event_str)
            
            return defender_markers + attacker_markers + defender_trails + attacker_trails + [event_text]
        
        # Create animation
        anim = animation.FuncAnimation(fig, animate, 
                                     frames=len(trajectory['positions']['defenders']),
                                     interval=50, blit=True)
        
        # Save as GIF
        anim.save(save_path, writer='pillow', fps=20)
        plt.close()
        
        print(f"  💾 Saved trajectory GIF: {save_path}")
    
    def create_trajectory_snapshot(self, trajectory: Dict, save_path: str):
        """Create a static PNG showing the full trajectory."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Left plot: Full trajectory
        ax1.set_xlim(-0.6, 0.6)
        ax1.set_ylim(-0.6, 0.6)
        ax1.set_aspect('equal')
        ax1.set_xlabel('X Position')
        ax1.set_ylabel('Y Position')
        ax1.set_title(f'Episode {trajectory["episode"]} - Complete Trajectory')
        
        # Add target line
        ax1.axhline(y=-0.5, color='red', linestyle='--', linewidth=2, label='Target Line')
        
        # Add potential attacker spawn positions with labels (generalized)
        num_spawn = self.config.num_spawn_positions  # Direct access, no default
        if num_spawn == 1:
            spawn_positions = [0.0]
        else:
            spacing = 0.8 / (num_spawn - 1)
            spawn_positions = [-0.4 + i * spacing for i in range(num_spawn)]
        
        for i, x_pos in enumerate(spawn_positions):
            ax1.plot(x_pos, 0.5, 'r^', markersize=12, alpha=0.3)
            label = f'S{i+1} (x={x_pos:.2f})'
            ax1.annotate(label, xy=(x_pos, 0.5), xytext=(x_pos, 0.55),
                        ha='center', fontsize=9, color='red', alpha=0.8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        # Plot defender trajectories
        for i in range(self.config.num_defenders):
            x = [p[i][0] for p in trajectory['positions']['defenders']]
            y = [p[i][1] for p in trajectory['positions']['defenders']]
            ax1.plot(x, y, 'b-', alpha=0.6, linewidth=2, label=f'Defender {i+1}')
            ax1.plot(x[0], y[0], 'bo', markersize=8)  # Start position
            ax1.plot(x[-1], y[-1], 'bs', markersize=10)  # End position
            
            # Add sensing radius at final position
            circle = Circle((x[-1], y[-1]), self.config.sensing_radius, 
                          fill=False, edgecolor='blue', alpha=0.3, linestyle='--')
            ax1.add_patch(circle)
        
        # Plot attacker trajectories
        for i in range(self.config.num_attackers):
            x = [p[i][0] for p in trajectory['positions']['attackers'] if i < len(p)]
            y = [p[i][1] for p in trajectory['positions']['attackers'] if i < len(p)]
            if x and y:  # Only plot if we have data
                ax1.plot(x, y, 'r-', alpha=0.6, linewidth=2, label=f'Attacker {i+1}')
                ax1.plot(x[0], y[0], 'r^', markersize=10)  # Start position
                ax1.plot(x[-1], y[-1], 'rv', markersize=12)  # End position
        
        # Mark sensing events
        for event in trajectory['sensing_events']:
            if event < len(trajectory['positions']['attackers']):
                pos = trajectory['positions']['attackers'][event][0]
                ax1.plot(pos[0], pos[1], 'y*', markersize=15, label='Sensing' if event == trajectory['sensing_events'][0] else "")
        
        ax1.legend(loc='best')
        ax1.grid(True, alpha=0.3)
        
        # Right plot: Key moments
        ax2.set_xlim(-0.6, 0.6)
        ax2.set_ylim(-0.6, 0.6)
        ax2.set_aspect('equal')
        ax2.set_xlabel('X Position')
        ax2.set_ylabel('Y Position')
        ax2.set_title('Key Moments')
        
        # Add potential spawn positions on right plot too (generalized)
        for x_pos in spawn_positions:  # Use the same spawn_positions from above
            ax2.plot(x_pos, 0.5, 'r^', markersize=12, alpha=0.2)
            ax2.plot([x_pos, x_pos], [0.5, -0.5], 'r:', alpha=0.2)  # Vertical guide lines
        
        # Show initial, mid, and final positions
        moments = [0, len(trajectory['positions']['defenders'])//2, -1]
        colors = ['green', 'orange', 'red']
        labels = ['Start', 'Middle', 'End']
        
        for idx, (moment, color, label) in enumerate(zip(moments, colors, labels)):
            # Defenders
            for i in range(self.config.num_defenders):
                pos = trajectory['positions']['defenders'][moment][i]
                ax2.plot(pos[0], pos[1], 'o', color=color, markersize=8, 
                        label=f'{label} - Defenders' if i == 0 else "")
            
            # Attackers
            for i in range(self.config.num_attackers):
                pos = trajectory['positions']['attackers'][moment][i]
                ax2.plot(pos[0], pos[1], '^', color=color, markersize=10,
                        label=f'{label} - Attackers' if i == 0 else "")
        
        ax2.axhline(y=-0.5, color='red', linestyle='--', linewidth=2, label='Target Line')
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
        
        # Add summary text
        sensing_occurred = len(trajectory['sensing_events']) > 0
        target_reached = len(trajectory['target_events']) > 0
        outcome = "✅ DEFENDED" if sensing_occurred else "❌ BREACHED" if target_reached else "⏱️ TIMEOUT"
        
        fig.suptitle(f'Target Defense - Episode {trajectory["episode"]} - Outcome: {outcome}', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved trajectory PNG: {save_path}")
    
    def plot_training_metrics(self, save_path: str):
        """Create and save training metrics plot."""
        if len(self.metrics_history['episode']) < 2:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Average Reward
        ax = axes[0, 0]
        ax.plot(self.metrics_history['episode'], self.metrics_history['reward'], 
               'b-', linewidth=2, label='Episode Reward')
        
        # Add rolling average
        window = min(20, len(self.metrics_history['reward']) // 4)
        if window > 1:
            rolling_avg = np.convolve(self.metrics_history['reward'], 
                                     np.ones(window)/window, mode='valid')
            roll_episodes = self.metrics_history['episode'][window-1:]
            ax.plot(roll_episodes, rolling_avg, 'r--', linewidth=2, 
                   label=f'{window}-Episode Average')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Average Reward')
        ax.set_title('Training Reward Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Success Rates
        ax = axes[0, 1]
        ax.plot(self.metrics_history['episode'], 
               np.array(self.metrics_history['interception_rate']) * 100,
               'g-', linewidth=2, label='Complete Interception')
        ax.plot(self.metrics_history['episode'], 
               np.array(self.metrics_history['sensing_rate']) * 100,
               'b-', linewidth=2, label='Avg Sensing Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Rate (%)')
        ax.set_title('Defense Success Metrics')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 105])
        
        # Plot 3: Episode Length
        ax = axes[1, 0]
        if self.metrics_history['avg_episode_length']:
            ax.plot(self.metrics_history['episode'], 
                   self.metrics_history['avg_episode_length'],
                   'purple', linewidth=2)
            ax.set_xlabel('Episode')
            ax.set_ylabel('Average Episode Length')
            ax.set_title('Episode Duration')
            ax.grid(True, alpha=0.3)
        
        # Plot 4: Performance Summary
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate summary statistics
        recent_episodes = min(50, len(self.metrics_history['reward']) // 5)
        if recent_episodes > 0:
            recent_reward = np.mean(self.metrics_history['reward'][-recent_episodes:])
            recent_interception = np.mean(self.metrics_history['interception_rate'][-recent_episodes:]) * 100
            recent_sensing = np.mean(self.metrics_history['sensing_rate'][-recent_episodes:]) * 100
            
            best_reward = max(self.metrics_history['reward'])
            best_sensing = max(self.metrics_history['sensing_rate']) * 100
            
            summary_text = f"""
TRAINING SUMMARY (Last {recent_episodes} Episodes)
{'='*40}

Average Reward:        {recent_reward:.3f}
Complete Interception: {recent_interception:.1f}%
Avg Sensing Rate:      {recent_sensing:.1f}%

Best Performance:
  Peak Reward:         {best_reward:.3f}
  Peak Sensing Rate:   {best_sensing:.1f}%

Configuration:
  Defenders:           {self.config.num_defenders}
  Attackers:           {self.config.num_attackers}
  Learning Rate:       {self.config.learning_rate}
  Network:             {self.config.num_layers} layers × {self.config.hidden_size} units
  Parallel Envs:       {self.config.num_envs}
"""
            ax.text(0.1, 0.9, summary_text, transform=ax.transAxes,
                   fontsize=11, verticalalignment='top',
                   fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'Training Metrics - Episode {self.metrics_history["episode"][-1]}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved metrics plot: {save_path}")
    
    def train(self):
        """Extended training loop with visualization."""
        print("\n" + "="*60)
        print(f"TRAINING WITH VISUALIZATION")
        print("="*60)
        print(f"\nConfiguration:")
        print(f"  - Defenders: {self.config.num_defenders}")
        print(f"  - Attackers: {self.config.num_attackers}")
        print(f"  - Parallel envs: {self.config.num_envs}")
        print(f"  - Episodes: {self.config.num_episodes}")
        print(f"  - Learning rate: {self.config.learning_rate}")
        print(f"  - Network: {self.config.num_layers} layers × {self.config.hidden_size} units")
        print(f"  - Visualization dir: {self.save_dir}")
        
        print("\nStarting training with visualization...")
        print("-"*40)
        
        for episode in range(self.config.num_episodes):
            # Train one episode
            avg_reward, interception_rate, sensing_rate = self.train_episode()
            
            # Store metrics
            self.episode_rewards.append(avg_reward)
            self.interception_rates.append(interception_rate)
            self.sensing_rates.append(sensing_rate)
            
            self.metrics_history['episode'].append(episode + 1)
            self.metrics_history['reward'].append(avg_reward)
            self.metrics_history['interception_rate'].append(interception_rate)
            self.metrics_history['sensing_rate'].append(sensing_rate)
            
            # Logging
            if (episode + 1) % self.config.log_interval == 0:
                recent_rewards = np.mean(self.episode_rewards[-self.config.log_interval:])
                recent_interception = np.mean(self.interception_rates[-self.config.log_interval:])
                recent_sensing = np.mean(self.sensing_rates[-self.config.log_interval:])
                
                print(f"Episode {episode+1}/{self.config.num_episodes}:")
                print(f"  Avg Reward: {recent_rewards:.3f}")
                print(f"  Complete Interception: {recent_interception:.2%}")
                print(f"  Avg Sensing Rate: {recent_sensing:.2%}")
            
            # Visualization at intervals
            viz_interval = max(10, self.config.num_episodes // 20)  # ~20 visualizations total
            if (episode + 1) % viz_interval == 0 or episode == 0:
                print(f"\n📊 Creating visualizations for episode {episode+1}...")
                
                # Collect and visualize trajectory
                trajectory = self.collect_trajectory(episode + 1)
                
                # Save trajectory as GIF
                gif_path = f"{self.save_dir}/trajectories/trajectory_ep{episode+1:04d}.gif"
                self.create_trajectory_animation(trajectory, gif_path)
                
                # Save trajectory as PNG
                png_path = f"{self.save_dir}/trajectories/trajectory_ep{episode+1:04d}.png"
                self.create_trajectory_snapshot(trajectory, png_path)
                
                # Update metrics plot
                metrics_path = f"{self.save_dir}/metrics/training_metrics_ep{episode+1:04d}.png"
                self.plot_training_metrics(metrics_path)
            
            # Save checkpoints
            if (episode + 1) % self.config.save_interval == 0:
                self.save_checkpoint(episode + 1)
        
        # Final visualizations
        print("\n📊 Creating final visualizations...")
        
        # Final trajectory
        trajectory = self.collect_trajectory(self.config.num_episodes)
        gif_path = f"{self.save_dir}/trajectories/trajectory_final.gif"
        png_path = f"{self.save_dir}/trajectories/trajectory_final.png"
        self.create_trajectory_animation(trajectory, gif_path)
        self.create_trajectory_snapshot(trajectory, png_path)
        
        # Final metrics
        metrics_path = f"{self.save_dir}/metrics/training_metrics_final.png"
        self.plot_training_metrics(metrics_path)
        
        # Create combined summary plot
        self.create_summary_visualization()
        
        # Training complete
        print("\n" + "="*60)
        print("TRAINING COMPLETE WITH VISUALIZATIONS")
        print("="*60)
        self.print_final_statistics()
        print(f"\n📁 All visualizations saved to: {self.save_dir}")
    
    def create_summary_visualization(self):
        """Create a comprehensive summary visualization."""
        fig = plt.figure(figsize=(20, 12))
        
        # Create grid
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Main reward plot
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(self.metrics_history['episode'], self.metrics_history['reward'], 
                'b-', alpha=0.6, linewidth=1, label='Episode Reward')
        
        # Rolling average
        window = min(20, len(self.metrics_history['reward']) // 4)
        if window > 1:
            rolling_avg = np.convolve(self.metrics_history['reward'], 
                                     np.ones(window)/window, mode='valid')
            roll_episodes = self.metrics_history['episode'][window-1:]
            ax1.plot(roll_episodes, rolling_avg, 'r-', linewidth=3, 
                    label=f'{window}-Episode Average')
        
        ax1.set_xlabel('Episode', fontsize=12)
        ax1.set_ylabel('Average Reward', fontsize=12)
        ax1.set_title('Learning Progress', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Success rates plot
        ax2 = fig.add_subplot(gs[1, :2])
        ax2.fill_between(self.metrics_history['episode'], 0,
                        np.array(self.metrics_history['sensing_rate']) * 100,
                        alpha=0.3, color='green', label='Sensing Rate')
        ax2.fill_between(self.metrics_history['episode'],
                        np.array(self.metrics_history['sensing_rate']) * 100,
                        100, alpha=0.3, color='red', label='Target Reached Rate')
        ax2.set_xlabel('Episode', fontsize=12)
        ax2.set_ylabel('Rate (%)', fontsize=12)
        ax2.set_title('Defense Success Over Time', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 105])
        
        # Performance heatmap
        ax3 = fig.add_subplot(gs[2, :2])
        
        # Create bins for heatmap
        n_bins = min(20, len(self.metrics_history['episode']) // 5)
        if n_bins > 1:
            bin_size = len(self.metrics_history['episode']) // n_bins
            binned_sensing = []
            binned_reward = []
            
            for i in range(n_bins):
                start_idx = i * bin_size
                end_idx = min((i + 1) * bin_size, len(self.metrics_history['sensing_rate']))
                binned_sensing.append(np.mean(self.metrics_history['sensing_rate'][start_idx:end_idx]) * 100)
                binned_reward.append(np.mean(self.metrics_history['reward'][start_idx:end_idx]))
            
            # Create 2D array for heatmap
            heatmap_data = np.array([binned_sensing, binned_reward])
            
            im = ax3.imshow(heatmap_data, aspect='auto', cmap='RdYlGn', 
                          interpolation='nearest', vmin=0, vmax=100)
            ax3.set_yticks([0, 1])
            ax3.set_yticklabels(['Sensing Rate (%)', 'Normalized Reward'])
            ax3.set_xlabel('Training Progress →', fontsize=12)
            ax3.set_title('Performance Heatmap', fontsize=14, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax3, orientation='horizontal', pad=0.1)
            cbar.set_label('Performance', fontsize=10)
        
        # Statistics panel
        ax4 = fig.add_subplot(gs[:, 2])
        ax4.axis('off')
        
        # Calculate statistics
        total_episodes = len(self.metrics_history['episode'])
        recent_n = min(50, total_episodes // 5)
        
        stats_text = f"""
TRAINING SUMMARY
{'='*30}

Total Episodes: {total_episodes}
Parallel Environments: {self.config.num_envs}

FINAL PERFORMANCE (Last {recent_n} ep)
{'='*30}
Avg Reward: {np.mean(self.metrics_history['reward'][-recent_n:]):.3f}
Complete Interception: {np.mean(self.metrics_history['interception_rate'][-recent_n:]) * 100:.1f}%
Avg Sensing Rate: {np.mean(self.metrics_history['sensing_rate'][-recent_n:]) * 100:.1f}%

BEST PERFORMANCE
{'='*30}
Peak Reward: {max(self.metrics_history['reward']):.3f}
Peak Sensing: {max(self.metrics_history['sensing_rate']) * 100:.1f}%
Best Episode: {self.metrics_history['episode'][np.argmax(self.metrics_history['reward'])]}

LEARNING METRICS
{'='*30}
Episodes to 90% success: {self._find_convergence_episode(0.9)}
Episodes to 95% success: {self._find_convergence_episode(0.95)}
Final Stability: {self._calculate_stability():.2f}

CONFIGURATION
{'='*30}
Defenders: {self.config.num_defenders}
Attackers: {self.config.num_attackers}
Sensing Radius: {self.config.sensing_radius}
Speed Ratio: {self.config.speed_ratio}
Learning Rate: {self.config.learning_rate}
Network: {self.config.num_layers}×{self.config.hidden_size}
Algorithm: PPO
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes,
                fontsize=9, verticalalignment='top',
                fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        # Overall title
        outcome = "🏆 EXCELLENT" if np.mean(self.metrics_history['sensing_rate'][-recent_n:]) > 0.8 else "✅ GOOD" if np.mean(self.metrics_history['sensing_rate'][-recent_n:]) > 0.6 else "⚠️ NEEDS IMPROVEMENT"
        fig.suptitle(f'Target Defense Training Summary - {outcome}', 
                    fontsize=16, fontweight='bold')
        
        # Save
        summary_path = f"{self.save_dir}/training_summary.png"
        plt.savefig(summary_path, dpi=200, bbox_inches='tight')
        plt.close()
        
        print(f"  💾 Saved training summary: {summary_path}")
    
    def _find_convergence_episode(self, threshold: float) -> str:
        """Find episode where success rate first exceeds threshold."""
        for i, rate in enumerate(self.metrics_history['sensing_rate']):
            if rate >= threshold:
                return str(self.metrics_history['episode'][i])
        return "N/A"
    
    def _calculate_stability(self) -> float:
        """Calculate stability metric (lower is better)."""
        if len(self.metrics_history['sensing_rate']) < 20:
            return 0.0
        recent = self.metrics_history['sensing_rate'][-20:]
        return np.std(recent)


def main():
    """Main entry point for visualization training."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Target Defense Training with Visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with visualizations
  python train_with_visualization.py --episodes 100 --envs 32
  
  # Extended training session
  python train_with_visualization.py --episodes 1000 --envs 64 --lr 0.001
  
  # Custom configuration
  python train_with_visualization.py --episodes 500 --defenders 4 --attackers 2 --sensing-radius 0.2
        """
    )
    
    # Training arguments
    parser.add_argument("--episodes", type=int, default=3000,
                       help="Number of training episodes")
    parser.add_argument("--envs", type=int, default=32,
                       help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device to use (cpu/cuda)")
    
    # Environment arguments
    parser.add_argument("--defenders", type=int, default=3,
                       help="Number of defenders")
    parser.add_argument("--attackers", type=int, default=1,
                       help="Number of attackers")
    parser.add_argument("--spawn-positions", type=int, default=3,
                       help="Number of spawn positions for attackers")
    parser.add_argument("--sensing-radius", type=float, default=0.15,
                       help="Defender sensing radius")
    parser.add_argument("--speed-ratio", type=float, default=0.7,
                       help="Speed ratio (attacker/defender)")
    
    # Network arguments
    parser.add_argument("--lr", type=float, default=0.0007,
                       help="Learning rate")
    parser.add_argument("--hidden-size", type=int, default=256,
                       help="Hidden layer size")
    parser.add_argument("--num-layers", type=int, default=3,
                       help="Number of hidden layers")
    
    # Visualization arguments
    parser.add_argument("--save-dir", type=str, default="./visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--log-interval", type=int, default=100,
                       help="Episodes between logging")
    parser.add_argument("--save-interval", type=int, default=250,
                       help="Episodes between checkpoint saves")
    
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig(
        num_defenders=args.defenders,
        num_attackers=args.attackers,
        num_spawn_positions=args.spawn_positions,
        sensing_radius=args.sensing_radius,
        speed_ratio=args.speed_ratio,
        num_envs=args.envs,
        num_episodes=args.episodes,
        learning_rate=args.lr,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        device=args.device,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )
    
    # Create and run trainer with visualization
    trainer = VisualizationTrainer(config, save_dir=args.save_dir)
    trainer.train()
    
    # Final evaluation
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    trainer.evaluate(num_episodes=20)
    
    print(f"\n✅ Training complete!")
    print(f"📁 Visualizations saved to: {trainer.save_dir}")
    print(f"\nView results:")
    print(f"  - Trajectories: {trainer.save_dir}/trajectories/")
    print(f"  - Metrics: {trainer.save_dir}/metrics/")
    print(f"  - Summary: {trainer.save_dir}/training_summary.png")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())