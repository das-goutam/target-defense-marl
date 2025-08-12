#!/usr/bin/env python3
"""
Basic training example for Target Defense environment.
Demonstrates simple 3v1 configuration with default parameters.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_train_target_defense import TrainingConfig, TargetDefenseRLTrainer

def main():
    # Basic configuration: 3 defenders vs 1 attacker
    config = TrainingConfig(
        num_defenders=3,
        num_attackers=1,
        num_spawn_positions=3,
        num_episodes=1000,
        num_envs=32,
        learning_rate=0.001,
        log_interval=50
    )
    
    print("Starting basic training...")
    print(f"Configuration: {config.num_defenders} defenders vs {config.num_attackers} attacker(s)")
    print(f"Spawn positions: {config.num_spawn_positions}")
    print(f"Training episodes: {config.num_episodes}")
    
    # Create trainer and run
    trainer = TargetDefenseRLTrainer(config)
    trainer.train()
    
    # Evaluate final performance
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    results = trainer.evaluate(num_episodes=50)
    
    print("\nTraining complete!")
    print(f"Final Complete Interception Rate: {results['interception_rate']:.2%}")
    print(f"Final Average Sensing Rate: {results['sensing_rate']:.2%}")
    print(f"Final Average Reward: {results['avg_reward']:.3f}")

if __name__ == "__main__":
    main()