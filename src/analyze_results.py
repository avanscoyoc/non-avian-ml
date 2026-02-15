#!/usr/bin/env python3
"""
Analyze batch inference results and generate confusion matrices.

Usage:
    python analyze_results.py <inference_results.csv>

Example:
    python analyze_results.py inference_results.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from sklearn.metrics import confusion_matrix, roc_auc_score, classification_report


def load_results(csv_path):
    """Load inference results from CSV."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} predictions")
    print(f"Classifiers: {df['classifier_species'].unique()}")
    print(f"Test species: {df['species'].unique()}")
    return df


def compute_confusion_matrices(df, threshold=0.5):
    """Compute confusion matrix for each classifier and test species combination."""
    results = []
    
    classifiers = df['classifier_species'].unique()
    test_species = df['species'].unique()
    
    for classifier in classifiers:
        for test_sp in test_species:
            # Filter data
            subset = df[(df['classifier_species'] == classifier) & (df['species'] == test_sp)]
            
            if len(subset) == 0:
                continue
            
            y_true = subset['present'].values
            y_pred = (subset['probability_present'] >= threshold).astype(int)
            y_prob = subset['probability_present'].values
            
            # Compute metrics
            cm = confusion_matrix(y_true, y_pred)
            
            # Handle edge cases where all predictions are same class
            if len(np.unique(y_true)) == 1:
                auc = np.nan
            else:
                try:
                    auc = roc_auc_score(y_true, y_prob)
                except:
                    auc = np.nan
            
            # Calculate metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results.append({
                'classifier_species': classifier,
                'test_species': test_sp,
                'n_samples': len(subset),
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'auc': auc,
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
                'true_positive': tp,
            })
    
    return pd.DataFrame(results)


def plot_confusion_matrices(df, output_dir='confusion_matrices', threshold=0.5):
    """Generate confusion matrix plots for each classifier-species pair."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    classifiers = df['classifier_species'].unique()
    test_species = df['species'].unique()
    
    for classifier in classifiers:
        for test_sp in test_species:
            subset = df[(df['classifier_species'] == classifier) & (df['species'] == test_sp)]
            
            if len(subset) == 0:
                continue
            
            y_true = subset['present'].values
            y_pred = (subset['probability_present'] >= threshold).astype(int)
            
            cm = confusion_matrix(y_true, y_pred)
            
            # Plot
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['Absent', 'Present'],
                       yticklabels=['Absent', 'Present'])
            plt.title(f'Classifier: {classifier}\nTest Species: {test_sp}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            filename = f"cm_{classifier}_{test_sp}.png"
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {filename}")


def plot_probability_distributions(df, output_dir='confusion_matrices'):
    """Plot probability distributions for each classifier-species pair."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    classifiers = df['classifier_species'].unique()
    test_species = df['species'].unique()
    
    for classifier in classifiers:
        for test_sp in test_species:
            subset = df[(df['classifier_species'] == classifier) & (df['species'] == test_sp)]
            
            if len(subset) == 0:
                continue
            
            # Separate by true label
            present = subset[subset['present'] == 1]['probability_present']
            absent = subset[subset['present'] == 0]['probability_present']
            
            # Plot
            plt.figure(figsize=(10, 6))
            
            if len(absent) > 0:
                plt.hist(absent, bins=30, alpha=0.5, label='True Absent', color='blue')
            if len(present) > 0:
                plt.hist(present, bins=30, alpha=0.5, label='True Present', color='red')
            
            plt.axvline(x=0.5, color='black', linestyle='--', label='Threshold')
            plt.xlabel('Probability Present')
            plt.ylabel('Count')
            plt.title(f'Classifier: {classifier} | Test Species: {test_sp}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            filename = f"dist_{classifier}_{test_sp}.png"
            plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {filename}")

def plot_summary_heatmap(metrics_df, metric='f1_score', output_dir='confusion_matrices'):
    """Create a single heatmap showing chosen metric for all classifier-species pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Pivot the data to create a matrix
    pivot_df = metrics_df.pivot(index='classifier_species', 
                                  columns='test_species', 
                                  values=metric)
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot heatmap
    sns.heatmap(pivot_df, annot=True, fmt='.3f', cmap='RdYlGn', 
                vmin=0, vmax=1, center=0.5,
                linewidths=0.5, cbar_kws={'label': metric.replace('_', ' ').title()})
    
    plt.title(f'{metric.replace("_", " ").title()} - All Classifiers vs Test Species', 
              fontsize=14, pad=20)
    plt.xlabel('Test Species', fontsize=12)
    plt.ylabel('Classifier Species', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    filename = f"summary_heatmap_{metric}.png"
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {filename}")
    return pivot_df


def main():
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    
    csv_path = sys.argv[1]
    
    if not Path(csv_path).exists():
        print(f"Error: File not found: {csv_path}")
        sys.exit(1)
    
    print("=" * 80)
    print("Analyzing Inference Results")
    print("=" * 80)
    print()
    
    # Load results
    df = load_results(csv_path)
    print()
    
    # Compute metrics
    print("Computing confusion matrices and metrics...")
    metrics_df = compute_confusion_matrices(df)
    
    # Save metrics to CSV
    metrics_csv = Path(csv_path).stem + "_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics to: {metrics_csv}")
    print()
    
    # Display summary
    print("=" * 80)
    print("METRICS SUMMARY")
    print("=" * 80)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', 20)
    print(metrics_df.to_string(index=False))
    print()
    
    # Generate plots
    #print("=" * 80)
    #print("Generating confusion matrix plots...")
    #plot_confusion_matrices(df)
    #print()
    #
    #print("Generating probability distribution plots...")
    #plot_probability_distributions(df)
    #print()

    # Generate summary heatmaps
    print("=" * 80)
    print("Generating summary heatmaps...")
    plot_summary_heatmap(metrics_df, metric='accuracy')
    plot_summary_heatmap(metrics_df, metric='f1_score')
    plot_summary_heatmap(metrics_df, metric='auc')
    print()
    
    print("=" * 80)
    print(f"Analysis complete! Check 'confusion_matrices/' directory for plots.")
    print("=" * 80)


if __name__ == "__main__":
    main()
