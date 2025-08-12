#!/usr/bin/env python3
"""
Generate experimental results and plots for the Target Defense paper
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import os
import json
from datetime import datetime

# Import your training modules
from rl_train_target_defense import TrainingConfig, TargetDefenseRLTrainer

def run_experiment(config_name: str, defenders: int, attackers: int, 
                  spawn_positions: int, episodes: int = 500) -> Dict:
    """Run a single experiment configuration."""
    print(f"\n{'='*60}")
    print(f"Running Experiment: {config_name}")
    print(f"Configuration: {defenders} defenders vs {attackers} attackers, {spawn_positions} spawn positions")
    print(f"{'='*60}")
    
    config = TrainingConfig(
        num_defenders=defenders,
        num_attackers=attackers,
        num_spawn_positions=spawn_positions,
        num_episodes=episodes,
        num_envs=32,
        learning_rate=0.001,
        log_interval=50
    )
    
    trainer = TargetDefenseRLTrainer(config)
    trainer.train()
    
    # Evaluate final performance
    eval_results = trainer.evaluate(num_episodes=50)
    
    # Store results
    results = {
        'config_name': config_name,
        'defenders': defenders,
        'attackers': attackers,
        'spawn_positions': spawn_positions,
        'episodes': episodes,
        'final_reward': eval_results['avg_reward'],
        'final_interception_rate': eval_results['interception_rate'],
        'final_sensing_rate': eval_results['sensing_rate'],
        'reward_history': trainer.episode_rewards,
        'interception_history': trainer.interception_rates,
        'sensing_history': trainer.sensing_rates
    }
    
    return results

def generate_comparison_table(results_list: List[Dict]) -> pd.DataFrame:
    """Generate comparison table for the paper."""
    data = []
    for r in results_list:
        data.append({
            'Defenders': r['defenders'],
            'Attackers': r['attackers'],
            'Spawn Pos': r['spawn_positions'],
            'CIR (%)': f"{r['final_interception_rate']*100:.1f}",
            'ASR (%)': f"{r['final_sensing_rate']*100:.1f}",
            'Avg Reward': f"{r['final_reward']:.2f}"
        })
    
    df = pd.DataFrame(data)
    return df

def plot_learning_curves(results_list: List[Dict], save_path: str = './paper_figures/'):
    """Generate learning curve plots for the paper."""
    os.makedirs(save_path, exist_ok=True)
    
    # Plot 1: Complete Interception Rate over time
    plt.figure(figsize=(10, 6))
    for r in results_list:
        if len(r['interception_history']) > 0:
            episodes = np.arange(1, len(r['interception_history']) + 1)
            plt.plot(episodes, np.array(r['interception_history']) * 100, 
                    label=r['config_name'], linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Complete Interception Rate (%)', fontsize=12)
    plt.title('Learning Progress: Complete Interception Rate', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 500])
    plt.ylim([0, 105])
    plt.savefig(f'{save_path}/learning_curves_cir.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/learning_curves_cir.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Average Reward over time
    plt.figure(figsize=(10, 6))
    for r in results_list:
        if len(r['reward_history']) > 0:
            episodes = np.arange(1, len(r['reward_history']) + 1)
            # Smooth the rewards with rolling average
            window = 20
            rewards = np.array(r['reward_history'])
            if len(rewards) > window:
                rewards_smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
                episodes_smooth = episodes[window-1:]
                plt.plot(episodes_smooth, rewards_smooth, 
                        label=r['config_name'], linewidth=2)
    
    plt.xlabel('Episode', fontsize=12)
    plt.ylabel('Average Reward', fontsize=12)
    plt.title('Learning Progress: Average Reward', fontsize=14)
    plt.legend(loc='lower right')
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 500])
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.savefig(f'{save_path}/learning_curves_reward.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/learning_curves_reward.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Bar chart comparing final performance
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    configs = [r['config_name'] for r in results_list]
    cir_values = [r['final_interception_rate'] * 100 for r in results_list]
    asr_values = [r['final_sensing_rate'] * 100 for r in results_list]
    reward_values = [r['final_reward'] for r in results_list]
    
    # Complete Interception Rate
    bars1 = ax1.bar(configs, cir_values, color='green', alpha=0.7)
    ax1.set_ylabel('Complete Interception Rate (%)', fontsize=12)
    ax1.set_title('Final CIR Comparison', fontsize=14)
    ax1.set_ylim([0, 105])
    ax1.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(cir_values):
        ax1.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    
    # Average Sensing Rate
    bars2 = ax2.bar(configs, asr_values, color='blue', alpha=0.7)
    ax2.set_ylabel('Average Sensing Rate (%)', fontsize=12)
    ax2.set_title('Final ASR Comparison', fontsize=14)
    ax2.set_ylim([0, 105])
    ax2.grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(asr_values):
        ax2.text(i, v + 2, f'{v:.1f}%', ha='center', fontsize=10)
    
    # Average Reward
    bars3 = ax3.bar(configs, reward_values, color='orange', alpha=0.7)
    ax3.set_ylabel('Average Reward', fontsize=12)
    ax3.set_title('Final Reward Comparison', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    for i, v in enumerate(reward_values):
        ax3.text(i, v + 0.1 if v > 0 else v - 0.3, f'{v:.2f}', ha='center', fontsize=10)
    
    # Rotate x-axis labels
    for ax in [ax1, ax2, ax3]:
        ax.set_xticklabels(configs, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/final_performance_comparison.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{save_path}/final_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {save_path}")

def generate_latex_table(df: pd.DataFrame) -> str:
    """Convert DataFrame to LaTeX table format."""
    latex = df.to_latex(index=False, escape=False)
    # Make it look nicer
    latex = latex.replace('\\toprule', '\\toprule\n\\midrule')
    latex = latex.replace('\\bottomrule', '\\midrule\n\\bottomrule')
    return latex

def main():
    """Run all experiments and generate results for the paper."""
    
    # Define experiment configurations
    experiments = [
        ("3v1 Baseline", 3, 1, 3),
        ("3v2 Moderate", 3, 2, 3),
        ("3v3 Balanced", 3, 3, 3),
        ("4v3 Advantage", 4, 3, 3),
        ("2v3 Challenge", 2, 3, 3),
        ("3v3 4-Spawn", 3, 3, 4),
        ("3v4 Outnumbered", 3, 4, 4),
    ]
    
    # Run experiments (or load existing results)
    results_file = './paper_results.json'
    
    if os.path.exists(results_file):
        print(f"Loading existing results from {results_file}")
        with open(results_file, 'r') as f:
            all_results = json.load(f)
    else:
        all_results = []
        for name, defenders, attackers, spawns in experiments:
            try:
                result = run_experiment(name, defenders, attackers, spawns, episodes=500)
                all_results.append(result)
                
                # Save intermediate results
                with open(results_file, 'w') as f:
                    # Convert numpy arrays to lists for JSON serialization
                    results_to_save = []
                    for r in all_results:
                        r_copy = r.copy()
                        r_copy['reward_history'] = list(r_copy['reward_history'])
                        r_copy['interception_history'] = list(r_copy['interception_history'])
                        r_copy['sensing_history'] = list(r_copy['sensing_history'])
                        results_to_save.append(r_copy)
                    json.dump(results_to_save, f, indent=2)
                    
            except Exception as e:
                print(f"Error in experiment {name}: {e}")
                continue
    
    # Generate comparison table
    df = generate_comparison_table(all_results)
    print("\n" + "="*60)
    print("EXPERIMENTAL RESULTS TABLE")
    print("="*60)
    print(df.to_string())
    
    # Save as LaTeX
    latex_table = generate_latex_table(df)
    with open('./paper_tables.tex', 'w') as f:
        f.write(latex_table)
    print("\nLaTeX table saved to paper_tables.tex")
    
    # Generate plots
    plot_learning_curves(all_results)
    
    # Generate summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    for r in all_results:
        print(f"\n{r['config_name']}:")
        print(f"  Final CIR: {r['final_interception_rate']*100:.1f}%")
        print(f"  Final ASR: {r['final_sensing_rate']*100:.1f}%")
        print(f"  Final Reward: {r['final_reward']:.2f}")
        
        # Calculate improvement
        if len(r['reward_history']) > 20:
            early = np.mean(r['reward_history'][:20])
            late = np.mean(r['reward_history'][-20:])
            improvement = ((late - early) / abs(early) * 100) if early != 0 else 0
            print(f"  Improvement: {improvement:+.1f}%")

if __name__ == "__main__":
    main()