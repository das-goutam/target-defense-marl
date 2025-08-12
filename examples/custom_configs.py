#!/usr/bin/env python3
"""
Examples of different training configurations for various scenarios.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rl_train_target_defense import TrainingConfig, TargetDefenseRLTrainer

def train_outnumbered():
    """Train defenders when outnumbered by attackers."""
    config = TrainingConfig(
        num_defenders=2,
        num_attackers=3,
        num_spawn_positions=4,
        num_episodes=2000,  # More episodes needed for harder task
        learning_rate=0.0007,  # Lower learning rate for stability
        sensing_radius=0.15
    )
    
    print("Training OUTNUMBERED scenario: 2 defenders vs 3 attackers")
    trainer = TargetDefenseRLTrainer(config)
    trainer.train()
    return trainer

def train_many_spawns():
    """Train with many possible spawn positions."""
    config = TrainingConfig(
        num_defenders=3,
        num_attackers=1,
        num_spawn_positions=10,  # Many spawn positions
        num_episodes=1500,
        learning_rate=0.001,
        sensing_radius=0.1  # Smaller sensing radius for challenge
    )
    
    print("Training MANY SPAWNS scenario: 10 possible spawn positions")
    trainer = TargetDefenseRLTrainer(config)
    trainer.train()
    return trainer

def train_balanced_large():
    """Train balanced configuration with many agents."""
    config = TrainingConfig(
        num_defenders=5,
        num_attackers=5,
        num_spawn_positions=5,
        num_episodes=3000,  # Longer training for complex coordination
        num_envs=64,  # More parallel environments
        learning_rate=0.0005,
        hidden_size=512,  # Larger network for complex task
        num_layers=4
    )
    
    print("Training BALANCED LARGE scenario: 5v5 configuration")
    trainer = TargetDefenseRLTrainer(config)
    trainer.train()
    return trainer

def train_speed_disadvantage():
    """Train with defenders slower than attackers."""
    config = TrainingConfig(
        num_defenders=3,
        num_attackers=2,
        num_spawn_positions=3,
        speed_ratio=1.2,  # Attackers faster than defenders!
        num_episodes=2000,
        learning_rate=0.001
    )
    
    print("Training SPEED DISADVANTAGE: Attackers 20% faster")
    trainer = TargetDefenseRLTrainer(config)
    trainer.train()
    return trainer

def compare_configurations():
    """Compare different configurations."""
    configurations = [
        ("Easy (3v1)", 3, 1, 3, 500),
        ("Balanced (3v3)", 3, 3, 3, 1000),
        ("Outnumbered (2v3)", 2, 3, 3, 1500),
        ("Advantage (4v3)", 4, 3, 3, 1000),
    ]
    
    results = {}
    for name, defenders, attackers, spawns, episodes in configurations:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        config = TrainingConfig(
            num_defenders=defenders,
            num_attackers=attackers,
            num_spawn_positions=spawns,
            num_episodes=episodes,
            log_interval=100
        )
        
        trainer = TargetDefenseRLTrainer(config)
        trainer.train()
        eval_results = trainer.evaluate(num_episodes=50)
        
        results[name] = {
            'CIR': eval_results['interception_rate'],
            'ASR': eval_results['sensing_rate'],
            'Reward': eval_results['avg_reward']
        }
    
    # Print comparison table
    print("\n" + "="*70)
    print("CONFIGURATION COMPARISON")
    print("="*70)
    print(f"{'Configuration':<20} {'CIR':<10} {'ASR':<10} {'Avg Reward':<10}")
    print("-"*50)
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['CIR']*100:>6.1f}%    "
              f"{metrics['ASR']*100:>6.1f}%    {metrics['Reward']:>8.2f}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Custom configuration examples")
    parser.add_argument("--scenario", choices=['outnumbered', 'spawns', 'large', 'speed', 'compare'],
                       default='compare', help="Which scenario to run")
    
    args = parser.parse_args()
    
    if args.scenario == 'outnumbered':
        train_outnumbered()
    elif args.scenario == 'spawns':
        train_many_spawns()
    elif args.scenario == 'large':
        train_balanced_large()
    elif args.scenario == 'speed':
        train_speed_disadvantage()
    else:
        compare_configurations()