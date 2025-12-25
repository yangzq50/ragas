#!/usr/bin/env python3
"""
Generate comparison figures for RAGAS evaluation metrics across retrieval policies.

This script loads evaluation results from 4 CSV files and generates box plots
comparing min, 25%, 50%, 75%, max, and mean for each policy.

Usage:
    .venv/bin/python dify/generate_metrics_figures.py
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Configuration
CSV_FILES = {
    'top3_rerank': 'evaluation_results_top3_rerank.csv',
    'top3_no_rerank': 'evaluation_results_top3_no_rerank.csv',
    'top10_rerank': 'evaluation_results_top10_rerank.csv',
    'top10_no_rerank': 'evaluation_results_top10_no_rerank.csv',
}

METRICS = ['faithfulness', 'answer_relevancy', 'context_precision', 'context_recall']

METRIC_LABELS = {
    'faithfulness': 'Faithfulness',
    'answer_relevancy': 'Answer Relevancy',
    'context_precision': 'Context Precision',
    'context_recall': 'Context Recall',
}

POLICY_LABELS = {
    'top3_rerank': 'Top-3\nRerank',
    'top3_no_rerank': 'Top-3\nNo Rerank',
    'top10_rerank': 'Top-10\nRerank',
    'top10_no_rerank': 'Top-10\nNo Rerank',
}

# Colors for each policy
COLORS = {
    'top3_rerank': '#3498db',      # Blue
    'top3_no_rerank': '#95a5a6',   # Gray
    'top10_rerank': '#2ecc71',     # Green
    'top10_no_rerank': '#e74c3c',  # Red
}


def load_data(base_path: str) -> dict:
    """Load all CSV files and return as a dictionary of DataFrames."""
    data = {}
    for policy, filename in CSV_FILES.items():
        filepath = os.path.join(base_path, filename)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            data[policy] = df
            print(f"Loaded {filename}: {len(df)} rows")
            
            # Report missing values
            for metric in METRICS:
                if metric in df.columns:
                    missing = df[metric].isna().sum()
                    if missing > 0:
                        print(f"  Warning: {missing} missing values in {metric}")
        else:
            print(f"Warning: {filepath} not found")
    return data


def generate_metric_figure(data: dict, metric: str, output_dir: str):
    """Generate a box plot for a single metric comparing all policies."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    policies = list(POLICY_LABELS.keys())
    box_data = []
    stats = []
    
    for policy in policies:
        if policy in data and metric in data[policy].columns:
            # Drop missing values for statistics
            values = data[policy][metric].dropna()
            box_data.append(values)
            
            # Calculate statistics
            stats.append({
                'policy': policy,
                'count': len(values),
                'missing': data[policy][metric].isna().sum(),
                'mean': values.mean(),
                'min': values.min(),
                'q25': values.quantile(0.25),
                'median': values.median(),
                'q75': values.quantile(0.75),
                'max': values.max(),
            })
        else:
            box_data.append([])
            stats.append(None)
    
    # Create box plots
    positions = np.arange(1, len(policies) + 1)
    bp = ax.boxplot(box_data, positions=positions, patch_artist=True, widths=0.6)
    
    # Color the boxes
    for i, (patch, policy) in enumerate(zip(bp['boxes'], policies)):
        patch.set_facecolor(COLORS[policy])
        patch.set_alpha(0.7)
    
    # Add mean markers
    for i, (policy, s) in enumerate(zip(policies, stats)):
        if s is not None:
            ax.scatter(i + 1, s['mean'], color='red', marker='D', s=50, zorder=5, 
                      label='Mean' if i == 0 else '')
    
    # Customize plot
    ax.set_xticklabels([POLICY_LABELS[p] for p in policies])
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title(f'{METRIC_LABELS[metric]} Comparison Across Retrieval Policies', fontsize=14, fontweight='bold')
    ax.set_ylim(-0.05, 1.1)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Add legend for mean
    ax.legend(loc='lower right')
    
    # Create table data with columns: policy, missing, mean, min, Q1, median, Q3, max
    table_data = []
    for s in stats:
        if s is not None:
            policy_name = POLICY_LABELS[s['policy']].replace('\n', ' ')
            table_data.append([
                policy_name,
                str(s['missing']),
                f"{s['mean']:.3f}",
                f"{s['min']:.3f}",
                f"{s['q25']:.3f}",
                f"{s['median']:.3f}",
                f"{s['q75']:.3f}",
                f"{s['max']:.3f}",
            ])
    
    col_labels = ['Policy', 'Missing', 'Mean', 'Min', 'Q1', 'Median', 'Q3', 'Max']
    
    # Add table below the plot
    plt.tight_layout(rect=[0, 0.18, 1, 1])
    
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        loc='bottom',
        cellLoc='center',
        bbox=[0.0, -0.45, 1.0, 0.30]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    
    # Style the header row
    for i, label in enumerate(col_labels):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    # Alternate row colors for readability
    for i in range(len(table_data)):
        for j in range(len(col_labels)):
            if i % 2 == 0:
                table[(i + 1, j)].set_facecolor('#D9E2F3')
            else:
                table[(i + 1, j)].set_facecolor('#FFFFFF')
    
    output_path = os.path.join(output_dir, f'{metric}_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return stats


def generate_summary_figure(all_stats: dict, output_dir: str):
    """Generate a summary figure comparing mean scores across all metrics."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    policies = list(POLICY_LABELS.keys())
    x = np.arange(len(METRICS))
    width = 0.2
    
    for i, policy in enumerate(policies):
        means = []
        for metric in METRICS:
            if metric in all_stats and policy in [s['policy'] for s in all_stats[metric] if s]:
                stat = next(s for s in all_stats[metric] if s and s['policy'] == policy)
                means.append(stat['mean'])
            else:
                means.append(0)
        
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, means, width, label=POLICY_LABELS[policy].replace('\n', ' '),
                     color=COLORS[policy], alpha=0.8)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            ax.annotate(f'{mean:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
    
    ax.set_ylabel('Mean Score', fontsize=12)
    ax.set_title('Mean Scores Comparison Across All Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'summary_comparison.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    # Determine paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(script_dir)  # ragas project root
    output_dir = os.path.join(script_dir, 'figures')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("\n=== Loading Data ===")
    data = load_data(base_path)
    
    if not data:
        print("Error: No data files found!")
        return
    
    # Generate figures for each metric
    print("\n=== Generating Figures ===")
    all_stats = {}
    for metric in METRICS:
        stats = generate_metric_figure(data, metric, output_dir)
        all_stats[metric] = stats
    
    # Generate summary figure
    generate_summary_figure(all_stats, output_dir)
    
    print("\n=== Complete ===")
    print(f"Generated {len(METRICS) + 1} figures in {output_dir}")


if __name__ == "__main__":
    main()
