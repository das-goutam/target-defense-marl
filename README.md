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

### Basic Training

Train 3 defenders against 1 attacker with 3 spawn positions:

```bash
python rl_train_target_defense.py
```

### Training with Visualization

Generate GIFs and performance plots during training:

```bash
python train_with_visualization.py --episodes 1000 --defenders 3 --attackers 1 --spawn-positions 3 --lr 0.001
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
├── rl_train_target_defense.py  # PPO training with direct VMAS integration
├── train_with_visualization.py # Extended training with visualization
├── apollonius_solver.py        # Apollonius circle solver for rewards
├── requirements.txt            # Python dependencies
├── LICENSE                    # MIT license
├── README.md                  # This file
├── examples/                  # Example scripts
│   ├── basic_training.py     # Simple training demonstration
│   ├── custom_configs.py     # Various configuration examples
│   └── visualization_demo.py # Visualization features demo
├── sample_results/            # Example outputs from training
│   ├── trajectories/         # Sample trajectory visualizations
│   │   ├── sample_trajectory.gif
│   │   └── sample_trajectory.png
│   └── metrics/              # Sample training metrics
│       └── sample_training_metrics.png
└── docs/                      # Documentation
    └── paper/                # LaTeX files for paper
        ├── target_defense_paper.tex
        ├── references.bib
        └── paper_generation_README.md
```

**During Training**: The following directories will be created automatically:
- `visualizations/` - Contains GIFs and PNGs from each training run
- `models/` - Saved model checkpoints
- `results/` - Training logs and metrics

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--episodes` | 1000 | Number of training episodes |
| `--defenders` | 3 | Number of defender agents |
| `--attackers` | 1 | Number of attacker agents |
| `--spawn-positions` | 3 | Number of possible attacker spawn positions |
| `--sensing-radius` | 0.15 | Defender sensing radius |
| `--speed-ratio` | 0.7 | Attacker speed relative to defenders |
| `--lr` | 0.001 | Learning rate for PPO |
| `--envs` | 32 | Number of parallel environments |
| `--save-interval` | 250 | Episodes between saving visualizations |
| `--log-interval` | 50 | Episodes between logging metrics |

## Example Results

### Training Performance
Typical performance with 3 defenders vs 1 attacker:
- **100% CIR** achieved after ~500 episodes
- **Average reward**: 1.0-1.5 per defender

### Challenging Configurations
- **2v3 (outnumbered)**: ~45% CIR, demonstrates coordination limits
- **3v3 (balanced)**: ~97% CIR with proper training
- **4v3 (advantage)**: ~99% CIR, faster convergence

## Visualization Examples

### Trajectory Animation
The system generates animated GIFs showing:
- Agent movements over time
- Sensing boundaries (blue circles)
- Spawn positions (red triangles)
- Interception events (yellow stars)

### Training Metrics
Automatically generated plots include:
- Complete Interception Rate over time
- Average rewards progression
- Sensing rate statistics

## Research Paper

For detailed algorithm description and experimental results, see:
- [Paper PDF](docs/paper/target_defense_paper.pdf) (if available)
- [LaTeX Source](docs/paper/target_defense_paper.tex)

### Generating Paper Results

```bash
python generate_paper_results.py
```

This will run all experimental configurations and generate:
- Comparison tables (LaTeX format)
- Learning curves
- Performance bar charts

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black .
```

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