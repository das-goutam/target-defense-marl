# Target Defense Multi-Agent Reinforcement Learning

A multi-agent reinforcement learning environment for the target defense problem, where multiple defenders must cooperate to intercept attackers before they reach a target line. Built on [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) framework.

## Key Features

- **Sparse Reward System**: Apollonius circle-based terminal rewards for optimal interception quality
- **Generalized Configuration**: Support for N defenders, M attackers, and K spawn positions
- **Parallel Training**: Vectorized environments for efficient PPO training
- **Complete Interception Metric**: Track episodes where ALL attackers are successfully intercepted
- **Comprehensive Visualization**: Training curves, trajectory animations, and performance metrics

## Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA (optional, for GPU acceleration)

### Setup

```bash
# Clone the repository
git clone https://github.com/das-goutam/target-defense-marl.git
cd target-defense-marl

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Training with Visualization

Train agents and generate GIFs and performance plots:

```bash
# Basic training with default parameters (3 defenders vs 1 attacker, 3000 episodes)
python train_with_visualization.py

# Quick test run with fewer episodes
python train_with_visualization.py --episodes 500 --save-interval 50
```

### Advanced Configuration

Train with multiple attackers and custom parameters:

```bash
python train_with_visualization.py \
    --episodes 5000 \
    --defenders 3 \
    --attackers 2 \
    --spawn-positions 5 \
    --sensing-radius 0.1 \
    --lr 0.0007 \
    --save-interval 500 \
    --log-interval 100 \
    --save-dir ./visualizations
```

## Environment Details

### State Space
- **Defenders**: Continuous position (x, y) ∈ [-0.5, 0.5]²
- **Attackers**: Spawn at y=0.5 from K equally-spaced positions
- **Target Line**: Horizontal line at y=-0.5

### Action Space
- **Continuous heading control**: θ ∈ [-π, π]
- **Fixed speed**: Defenders move at max_speed, attackers at 0.7 × max_speed

### Reward Structure
Sparse rewards given only at episode termination:
- **Sensing Success**: 0.5 to 1.5 (based on interception y-position)
- **Target Reached**: -10.0 (split among defenders)

### Episode Termination
Episode ends when ALL attackers have either:
- Been sensed by a defender (within radius 0.15)
- Reached the target line
- Maximum steps (200) exceeded

## Performance Metrics

### Complete Interception Rate (CIR)
Percentage of episodes where ALL attackers are intercepted:
```
CIR = (Episodes with all attackers sensed) / (Total episodes)
```

### Average Sensing Rate (ASR)
Overall interception performance across all attackers:
```
ASR = (Total attackers sensed) / (Total attackers spawned)
```

## Project Structure

```
target-defense-marl/
├── vmas_target_defense.py      # Core VMAS environment implementation
├── train_with_visualization.py # PPO training with comprehensive visualization
├── apollonius_solver.py        # Apollonius circle solver for rewards
├── requirements.txt            # Python dependencies
├── LICENSE                    # MIT license
├── README.md                  # This file
├── sample_results/            # Sample results for different training configurations
│   ├── trajectory_ep2000.gif/png
│   ├── trajectory_ep3500.gif/png
│   ├── trajectory_ep3750.gif/png
│   ├── trajectory_ep5000.gif/png
│   ├── 324_trajectory_ep2000.gif/png
│   └── training_metrics_ep3750.png
└── docs/                      # Documentation
    └── paper/                # LaTeX files for academic paper
        ├── target_defense_paper.tex  # Main paper document
        └── references.bib            # Bibliography
```

**During Training**: The following directories will be created automatically:
- `visualizations/` - Contains GIFs and PNGs from each training run
- `models/` - Saved model checkpoints
- `results/` - Training logs and metrics

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 3000 | Number of training episodes |
| `--defenders` | 3 | Number of defender agents |
| `--attackers` | 1 | Number of attacker agents |
| `--spawn-positions` | 3 | Number of possible attacker spawn positions |
| `--sensing-radius` | 0.15 | Defender sensing radius |
| `--speed-ratio` | 0.7 | Attacker speed relative to defenders |
| `--lr` | 0.0007 | Learning rate for PPO |
| `--envs` | 32 | Number of parallel environments |
| `--save-interval` | 250 | Episodes between saving visualizations |
| `--log-interval` | 100 | Episodes between logging metrics |



## Citation

If you use this code in your research, please cite:

```bibtex
@software{target_defense_marl,
  title={Target Defense Multi-Agent Reinforcement Learning},
  author={Goutam Das},
  year={2024},
  url={https://github.com/das-goutam/target-defense-marl}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on [VMAS](https://github.com/proroklab/VectorizedMultiAgentSimulator) framework
- Uses [BenchMARL](https://github.com/facebookresearch/BenchMARL) concepts
- Apollonius circle theory for optimal reward calculation

## Contact

For questions or collaborations, please open an issue on the [GitHub repository](https://github.com/das-goutam/target-defense-marl/issues).