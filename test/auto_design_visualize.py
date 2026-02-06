"""Visualization tools for automated design results.

Copyright 2025 CktGen Authors.

Licensed under the MIT License.

This module provides visualization tools for comparing automated circuit design
results across different models. It generates bar charts and scatter plots to
compare accuracy and Figure of Merit (FoM) metrics.
"""


import matplotlib.pyplot as plt
import pickle
import os
import argparse
from pathlib import Path
from matplotlib import font_manager
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Configure matplotlib font and styling
font_manager.fontManager.addfont('./arial.ttf')

plt.rcParams['font.family'] = 'arial'
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.direction'] = 'out'
plt.rcParams['ytick.direction'] = 'out'
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18

#################################################################################

# Define color palette for different models
colors = [
    (0.35, 0.70, 0.90),  # Light blue
    (0.98, 0.60, 0.60),  # Light red
    (0.90, 0.80, 0.45),  # Yellow
    (0.70, 0.87, 0.54),  # Light green
    (0.80, 0.60, 0.79),  # Purple
]

def darken_color(rgb, factor=0.6):
    """Darkens an RGB color by multiplying each channel by a factor.
    
    Args:
        rgb: Tuple of (R, G, B) values in range [0, 1].
        factor: Multiplier for darkening (0-1). Lower values = darker. Default 0.6.
        
    Returns:
        tuple: Darkened (R, G, B) color tuple, clamped to [0, 1].
        
    Examples:
        >>> darken_color((0.9, 0.8, 0.7), 0.6)
        (0.54, 0.48, 0.42)
    """
    return tuple([max(0, min(1, c * factor)) for c in rgb])

deep_colors = [darken_color(c, 0.6) for c in colors]

#################################################################################
#                        Dataset Configuration
#################################################################################
# Define all dataset configurations
datasets_config = [
    {
        'name': '101',
        'title': 'Ckt-Bench-101',
        'bar_plt_name': '101.svg',
        'scatter_plt_name': '101_scatter.svg',
        'paths': {
            'cktgen': './checkpoints/cktgen/cktgen_auto_design_101.pkl',
            'cktgnn': './checkpoints/baselines/cktgnn/cktgnn_auto_design_101.pkl',
            'pace': './checkpoints/baselines/pace/pace_auto_design_101.pkl',
            'ldt': './checkpoints/baselines/ldt/ldt_auto_design_101.pkl',
            'cvaegan': './checkpoints/baselines/cvaegan/cvaegan_auto_design_101.pkl',
        }
    },
    {
        'name': '301',
        'title': 'Ckt-Bench-301',
        'bar_plt_name': '301.svg',
        'scatter_plt_name': '301_scatter.svg',
        'paths': {
            'cktgen': './checkpoints/cktgen/cktgen_auto_design_301.pkl',
            'cktgnn': './checkpoints/baselines/cktgnn/cktgnn_auto_design_301.pkl',
            'pace': './checkpoints/baselines/pace/pace_auto_design_301.pkl',
            'ldt': './checkpoints/baselines/ldt/ldt_auto_design_301.pkl',
            'cvaegan': './checkpoints/baselines/cvaegan/cvaegan_auto_design_301.pkl',
        }
    }
]

#################################################################################
#                        Generate All Charts
#################################################################################
# Parse command line arguments
parser = argparse.ArgumentParser(description='Auto design visualization')
parser.add_argument('--out_dir', type=str, default='./output', help='Output directory for charts')
parser.add_argument('--exp_name', type=str, default='auto_design_visualize', help='Experiment name')
args, _ = parser.parse_known_args()

# Create output directory if not exists
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {out_dir.absolute()}")

output_files = []
for dataset_config in datasets_config:
    # Load data
    with open(dataset_config['paths']['cktgen'], 'rb') as f:
        cktgen_data = pickle.load(f)
    with open(dataset_config['paths']['cktgnn'], 'rb') as f:
        cktgnn_data = pickle.load(f)
    with open(dataset_config['paths']['pace'], 'rb') as f:
        pace_data = pickle.load(f)
    with open(dataset_config['paths']['ldt'], 'rb') as f:
        ldt_data = pickle.load(f)
    with open(dataset_config['paths']['cvaegan'], 'rb') as f:
        cvaegan_data = pickle.load(f)
    
    model_data = {
        'PACE': pace_data,
        'CktGNN': cktgnn_data,
        'CVAEGAN': cvaegan_data,
        'LDT': ldt_data,
        'CktGen': cktgen_data,
    }
    
    title_name = dataset_config['title']
    
    #################################################################################
    #                                   Bar Chart
    #################################################################################
    # 1. Calculate average accuracy and FoM for each model
    model_names = list(model_data.keys())
    avg_acc = []
    avg_fom = []
    
    for model in model_names:
        d = model_data[model]
        # Convert tensors to numpy arrays if needed, then compute mean
        accs = [float(acc.cpu().numpy()) if hasattr(acc, 'cpu') else float(acc) for acc in d['best_acc']]
        foms = [float(fom.cpu().numpy()) if hasattr(fom, 'cpu') else float(fom) for fom in d['best_fom']]
        avg_acc.append(np.mean(accs))
        avg_fom.append(np.mean(foms))
    
    # Create dual-axis plot
    fig, ax1 = plt.subplots(figsize=(9,6))
    
    # Convert accuracy to percentage
    avg_acc_percent = [acc * 100 for acc in avg_acc]
    
    # Left y-axis: Accuracy as bar chart (in percentage)
    bar = ax1.bar(model_names, avg_acc_percent, color=colors, label='Specification accuracy', edgecolor='black')
    ax1.set_ylabel('Specification accuracy (%)', fontsize=18)
    ax1.set_ylim(0, 110)
    
    # Add accuracy values as text labels on top of bars (in percentage)
    for i, v in enumerate(avg_acc_percent):
        ax1.text(i, v+3, f'{v:.2f}%', ha='center', fontsize=18)
    
    # Right y-axis: Average FoM as line plot with colored scatter points
    ax2 = ax1.twinx()
    ax2.plot(model_names, avg_fom, color='gray', zorder=1, linewidth=2)  # Gray connecting line
    
    # Add scatter points for each model's FoM (colored by model)
    for i, (x, y) in enumerate(zip(model_names, avg_fom)):
        ax2.scatter(x, y, s=160, color=deep_colors[i], zorder=2, label=f'{model_names[i]} FoM', edgecolor='black')
    ax2.set_ylabel('Average FoM', fontsize=18)
    
    indicator_legend = [
        Patch(facecolor='gray', edgecolor='black', label='Specification accuracy (bar)'),
        Line2D([0], [0], marker='o', color='gray', markerfacecolor='white', markeredgecolor='black',
               linewidth=0, markersize=13, label='Average FoM (dot)')
    ]
    
    legend2 = ax1.legend(handles=indicator_legend, loc='upper left', fontsize=18)
    
    # Finalize and save the plot
    # plt.title(title_name, fontsize=18)
    plt.tight_layout()
    bar_output_path = out_dir / dataset_config['bar_plt_name']
    plt.savefig(bar_output_path, bbox_inches='tight')
    output_files.append(str(bar_output_path))
    plt.close()  # Close figure to release memory
    
    #################################################################################
    #                                   Scatter Plot
    #################################################################################
    data = []
    for i, (model, d) in enumerate(model_data.items()):
        acc = [float(x.cpu().numpy()) if hasattr(x, 'cpu') else float(x) for x in d['acc']]
        fom = [float(x.cpu().numpy()) if hasattr(x, 'cpu') else float(x) for x in d['fom']]
        # Convert accuracy to percentage
        for a, f in zip(acc, fom):
            data.append({'Model': model, 'Accuracy': a * 100, 'FoM': f})
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(8.2, 6))
    for i, model in enumerate(df['Model'].unique()):
        d = df[df['Model'] == model]
        ax.scatter(d['Accuracy'], d['FoM'], label=model, alpha=0.6, s=60, color=deep_colors[i])
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # plt.title(title_name, fontsize=18)
    plt.xlabel('Specification accuracy (%)', fontsize=18)
    plt.ylabel('FoM', fontsize=18)
    plt.legend(loc='lower right', fontsize=13)
    plt.tight_layout()
    scatter_output_path = out_dir / dataset_config['scatter_plt_name']
    plt.savefig(scatter_output_path, bbox_inches='tight')
    output_files.append(str(scatter_output_path))
    plt.close()  # Close figure to release memory

print("All charts generated successfully!")
print("Output files:")
for f in output_files:
    print(f"  - {f}")