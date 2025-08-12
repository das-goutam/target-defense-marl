# Target Defense Paper Generation Guide

This guide helps you create a complete research paper documenting the Target Defense multi-agent RL system.

## Files Created

1. **target_defense_paper.tex** - Main LaTeX document for Overleaf
2. **references.bib** - Bibliography file with relevant citations
3. **generate_paper_results.py** - Script to run experiments and generate results
4. **paper_generation_README.md** - This guide

## Steps to Generate Paper

### 1. Run Experiments and Generate Results

```bash
# Run all experiments (this will take ~1-2 hours)
python generate_paper_results.py
```

This will:
- Run 7 different configurations (3v1, 3v2, 3v3, etc.)
- Save results to `paper_results.json`
- Generate plots in `paper_figures/` directory
- Create LaTeX tables in `paper_tables.tex`

### 2. Quick Test (Optional)

To test with fewer episodes:
```bash
# Edit generate_paper_results.py and change episodes=500 to episodes=100
python generate_paper_results.py
```

### 3. Upload to Overleaf

1. Create a new Overleaf project
2. Upload these files:
   - `target_defense_paper.tex`
   - `references.bib`
   - All files from `paper_figures/` directory
   - Any trajectory GIFs/PNGs from `visualizations/` you want to include

### 4. Compile in Overleaf

The document should compile without errors. Make sure to:
- Use PDFLaTeX compiler
- Enable automatic bibliography compilation

## Customizing the Paper

### Add Your Visualizations

1. Copy your best trajectory visualizations:
```bash
cp visualizations/run_*/trajectories/trajectory_final.png paper_figures/trajectory_example.png
```

2. Add to LaTeX:
```latex
\begin{figure}[h]
\centering
\includegraphics[width=0.8\textwidth]{trajectory_example.png}
\caption{Example trajectory showing successful interception}
\end{figure}
```

### Modify Experiments

Edit `generate_paper_results.py` to add new configurations:
```python
experiments = [
    ("5v5 Large", 5, 5, 5),  # Add this line
    # ... existing configs
]
```

### Update Results in Paper

After running experiments, the script generates:
- `paper_tables.tex` - Copy the LaTeX table into your document
- `paper_figures/*.png` - Reference these in your figures

## Paper Structure

The LaTeX document includes:

1. **Abstract** - Problem overview and key results
2. **Introduction** - Motivation and contributions
3. **Problem Formulation** - Mathematical description
4. **Reward Design** - Apollonius-based sparse rewards
5. **Learning Algorithm** - PPO implementation details
6. **Performance Metrics** - CIR and ASR definitions
7. **Experimental Results** - Comparison tables and learning curves
8. **Behavioral Analysis** - Emergent strategies
9. **Implementation Details** - VMAS integration
10. **Conclusion** - Summary and future work

## Key Results to Highlight

- **Complete Interception Rate (CIR)**: Percentage of episodes where ALL attackers are intercepted
- **Average Sensing Rate (ASR)**: Overall interception performance
- **Learning Efficiency**: Episodes needed to reach 90%+ CIR
- **Scalability**: Performance with different numbers of agents

## Figures to Generate

The script automatically creates:
1. Learning curves (CIR over episodes)
2. Reward progression
3. Final performance comparison (bar charts)
4. Trajectory visualizations (from training)

## Tips for Writing

1. **Emphasize novelty**: 
   - Sparse rewards with Apollonius theory
   - Complete interception metric
   - Generalized N-M-K configuration

2. **Include failure analysis**:
   - Show cases where defenders fail
   - Explain why certain configurations are harder

3. **Compare configurations**:
   - Balanced (3v3) vs. imbalanced (2v3, 4v3)
   - Effect of spawn positions

4. **Future work suggestions**:
   - Communication between defenders
   - Adaptive attacker strategies
   - Transfer learning

## Troubleshooting

### If experiments fail:
- Reduce number of episodes
- Run configurations one at a time
- Check GPU memory if using CUDA

### If LaTeX won't compile:
- Make sure all image files are uploaded
- Check that references.bib is included
- Verify table formatting in paper_tables.tex

### Missing plots:
- Ensure matplotlib is installed: `pip install matplotlib pandas`
- Check that paper_figures/ directory exists
- Verify data in paper_results.json

## Contact

For questions about the implementation, refer to the main README or the code comments in:
- `vmas_target_defense.py` - Environment implementation
- `rl_train_target_defense.py` - Training logic
- `train_with_visualization.py` - Visualization system