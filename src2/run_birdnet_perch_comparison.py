#!/usr/bin/env python3
"""
Run BirdNET and Perch comparison experiment on coyote data.
"""

import random
import sys
import os
sys.path.append('/workspaces/non-avian-ml/src2')

from config import load_config
from data_loader import load_audio_files, create_kfold_splits
from model_loader import load_model
from trainer import train_model, evaluate_model
from results import save_results
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


def run_birdnet_perch_experiment():
    """Run the complete BirdNET vs Perch experiment."""
    print("🚀 Starting BirdNET vs Perch Comparison Experiment")
    print("=" * 60)
    
    try:
        # Load configuration
        config = load_config('birdnet_perch_config.yaml')
        print(f"✓ Configuration loaded: {config.models} on {config.species}")
        print(f"  Training sizes: {config.training_sizes}")
        print(f"  K-folds: {config.n_folds}")
        print(f"  Random seeds: {config.random_seeds}")
        print(f"  K-fold seed: {config.kfold_seed}")
        
        all_results = []
        
        for species in config.species:
            print(f"\n🎯 Processing species: {species}")
            
            for model_name in config.models:
                print(f"\n📊 Testing model: {model_name}")
                
                for training_size in config.training_sizes:
                    print(f"\n  📈 Training size: {training_size}")
                    
                    # Collect results across all random seeds
                    all_seed_scores = []
                    
                    for seed_idx, random_seed in enumerate(config.random_seeds):
                        print(f"\n    🎲 Random seed {seed_idx + 1}/{len(config.random_seeds)} (seed={random_seed})")
                        
                        # Determine data type (perch uses data_5s, others use data)
                        datatype = "data_5s" if model_name == "perch" else "data"
                        
                        try:
                            # Load data with this random seed (affects which subset is selected)
                            files, labels = load_audio_files(
                                config.data_path, species, training_size, datatype, random_seed=random_seed
                            )
                            print(f"      ✓ Loaded {len(files)} files from {datatype}")
                            
                            # Create K-fold splits (use consistent kfold_seed)
                            splits = create_kfold_splits(
                                files, labels, config.n_folds, seed=config.kfold_seed
                            )
                            print(f"      ✓ Created {len(splits)} folds")
                            
                            fold_scores = []
                            
                            for fold, (train_idx, val_idx) in enumerate(splits):
                                print(f"      📋 Fold {fold + 1}/{config.n_folds}")
                                
                                # Split data
                                train_files = [files[i] for i in train_idx]
                                train_labels = [labels[i] for i in train_idx]
                                val_files = [files[i] for i in val_idx]
                                val_labels = [labels[i] for i in val_idx]
                                
                                print(f"        Train: {len(train_files)}, Val: {len(val_files)}")
                                
                                # Load and train model
                                original_model, device = load_model(model_name)
                                is_embedding = model_name in ["birdnet", "perch"]
                                
                                trained_model = train_model(
                                    original_model, train_files, train_labels, device, is_embedding
                                )
                                
                                # Evaluate
                                if is_embedding:
                                    score = evaluate_model(
                                        trained_model, val_files, val_labels, device, original_model
                                    )
                                else:
                                    score = evaluate_model(
                                        trained_model, val_files, val_labels, device
                                    )
                                
                                fold_scores.append(score)
                                print(f"        ✓ Fold {fold + 1} AUC: {score:.4f}")
                            
                            # Add this seed's fold scores to collection
                            all_seed_scores.extend(fold_scores)
                            
                            # Calculate statistics for this seed
                            mean_score = np.mean(fold_scores)
                            std_score = np.std(fold_scores)
                            print(f"      🎯 Seed {random_seed} Mean AUC: {mean_score:.4f} ± {std_score:.4f}")
                            
                        except Exception as e:
                            print(f"      ❌ Error with seed {random_seed}: {e}")
                            continue
                    
                    # Calculate overall statistics across all seeds
                    if all_seed_scores:
                        overall_mean = np.mean(all_seed_scores)
                        overall_std = np.std(all_seed_scores)
                        
                        # Save result
                        result = {
                            "species": species,
                            "model": model_name,
                            "training_size": training_size,
                            "mean_auc": overall_mean,
                            "std_auc": overall_std,
                            "fold_scores": all_seed_scores,
                            "n_seeds": len(config.random_seeds),
                            "n_total_folds": len(all_seed_scores),
                        }
                        all_results.append(result)
                        
                        print(f"    🎯 Overall Mean AUC: {overall_mean:.4f} ± {overall_std:.4f} "
                              f"(across {len(all_seed_scores)} folds from {len(config.random_seeds)} seeds)")
                    else:
                        print(f"    ❌ No valid results for {model_name} at size {training_size}")# Save all results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"{config.results_path}/birdnet_perch_comparison_{timestamp}.csv"
        save_results(all_results, output_file)
        print(f"\n💾 Results saved to: {output_file}")
        
        return all_results, output_file
        
    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None


def create_comparison_plot(results, output_file):
    """Create a comparison plot of BirdNET vs Perch performance."""
    print("\n📊 Creating comparison plot...")
    
    try:
        # Convert to DataFrame for easier plotting
        df = pd.DataFrame(results)
        
        # Set up the plot
        plt.figure(figsize=(12, 8))
        
        # Plot for each model
        models = df['model'].unique()
        colors = {'birdnet': '#1f77b4', 'perch': '#ff7f0e'}
        markers = {'birdnet': 'o', 'perch': 's'}
        
        for model in models:
            model_data = df[df['model'] == model]
            
            # Plot mean AUC with error bars
            plt.errorbar(
                model_data['training_size'], 
                model_data['mean_auc'],
                yerr=model_data['std_auc'],
                label=f'{model.upper()}',
                color=colors.get(model, 'gray'),
                marker=markers.get(model, 'o'),
                markersize=8,
                linewidth=2,
                capsize=5,
                capthick=2
            )
        
        # Customize plot
        plt.xlabel('Training Size (samples per class)', fontsize=12)
        plt.ylabel('ROC-AUC Score', fontsize=12)
        plt.title('BirdNET vs Perch Performance Comparison\nCoyote Classification', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.0)  # Set reasonable y-axis limits
        
        # Add some styling
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        # Save plot
        plot_file = output_file.replace('.csv', '_plot.png')
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Plot saved to: {plot_file}")
        
        # Show summary statistics
        print("\n📋 Performance Summary:")
        print("=" * 40)
        
        for model in models:
            model_data = df[df['model'] == model]
            avg_auc = model_data['mean_auc'].mean()
            best_auc = model_data['mean_auc'].max()
            best_size = model_data.loc[model_data['mean_auc'].idxmax(), 'training_size']
            
            print(f"{model.upper():>8}: Avg AUC = {avg_auc:.4f}, Best = {best_auc:.4f} (size {best_size})")
        
        # Show the plot
        plt.show()
        
        return plot_file
        
    except Exception as e:
        print(f"❌ Plot creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function."""
    print("BirdNET vs Perch Comparison Experiment")
    print("======================================")
    
    # Run the experiment
    results, output_file = run_birdnet_perch_experiment()
    
    if results and output_file:
        # Create comparison plot
        plot_file = create_comparison_plot(results, output_file)
        
        print("\n🎉 Experiment completed successfully!")
        print(f"📊 Results: {output_file}")
        if plot_file:
            print(f"📈 Plot: {plot_file}")
    else:
        print("\n❌ Experiment failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)