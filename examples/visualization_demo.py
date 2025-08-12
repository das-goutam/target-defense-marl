#!/usr/bin/env python3
"""
Demonstration of visualization capabilities.
Generates GIFs and plots for a short training session.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_with_visualization import VisualizationTrainer
from rl_train_target_defense import TrainingConfig
import argparse

def run_visualization_demo():
    """Run a short training session with full visualization."""
    
    # Configuration for demo
    config = TrainingConfig(
        num_defenders=3,
        num_attackers=2,
        num_spawn_positions=4,
        num_episodes=100,  # Short demo
        num_envs=16,  # Fewer environments for demo
        learning_rate=0.001,
        log_interval=10,  # Frequent logging
        save_interval=25  # Save visualizations every 25 episodes
    )
    
    print("="*60)
    print("VISUALIZATION DEMO")
    print("="*60)
    print(f"Configuration: {config.num_defenders}v{config.num_attackers}")
    print(f"Spawn positions: {config.num_spawn_positions}")
    print(f"Episodes: {config.num_episodes}")
    print("="*60)
    
    # Create trainer with visualization
    trainer = VisualizationTrainer(config, save_dir="./demo_visualizations")
    
    print(f"\nVisualizations will be saved to: {trainer.save_dir}")
    print("Files generated will include:")
    print("  - Trajectory GIFs showing agent movements")
    print("  - Trajectory PNGs with complete paths")
    print("  - Training metrics plots")
    print("  - Summary statistics\n")
    
    # Run training
    trainer.train()
    
    # Final evaluation
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    trainer.evaluate(num_episodes=10)
    
    print(f"\nVisualization files saved to: {trainer.save_dir}")
    print("\nGenerated files:")
    print(f"  - Trajectories: {trainer.save_dir}/trajectories/")
    print(f"  - Metrics: {trainer.save_dir}/metrics/")
    print(f"  - Summary: {trainer.save_dir}/training_summary.png")
    
    return trainer.save_dir

def create_comparison_visualization():
    """Create visualizations comparing different sensing radii."""
    
    sensing_radii = [0.1, 0.15, 0.2]
    
    for radius in sensing_radii:
        config = TrainingConfig(
            num_defenders=3,
            num_attackers=1,
            num_spawn_positions=3,
            sensing_radius=radius,
            num_episodes=200,
            log_interval=20,
            save_interval=50
        )
        
        save_dir = f"./comparison_r{radius}"
        print(f"\nTraining with sensing radius = {radius}")
        
        trainer = VisualizationTrainer(config, save_dir=save_dir)
        trainer.train()
        
        eval_results = trainer.evaluate(num_episodes=20)
        print(f"Results for r={radius}:")
        print(f"  CIR: {eval_results['interception_rate']:.2%}")
        print(f"  Visualizations: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualization demonstrations")
    parser.add_argument("--mode", choices=['demo', 'compare'], default='demo',
                       help="Which visualization to generate")
    
    args = parser.parse_args()
    
    if args.mode == 'demo':
        save_dir = run_visualization_demo()
        print(f"\nTo view the generated visualizations:")
        print(f"  open {save_dir}/trajectories/trajectory_final.gif")
        print(f"  open {save_dir}/metrics/training_metrics_final.png")
    else:
        create_comparison_visualization()
        print("\nComparison visualizations created in ./comparison_r* directories")