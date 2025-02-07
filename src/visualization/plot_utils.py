import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

class VisualizationUtils:
    def __init__(self, config):
        self.config = config
        self.output_dir = Path("./results/plots")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use(self.config['plots']['format']['style'])
        self.figsize = self.config['plots']['format']['figsize']
        self.dpi = self.config['plots']['format']['dpi']
        
    def plot_eigenvalue_distribution(self, metrics_history, save=True):
        """Plot eigenvalue distribution over time"""
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Extract singular values over time
        timestamps = []
        singular_values = []
        
        for snapshot in metrics_history:
            for layer_metric in snapshot['layer_metrics']:
                timestamps.append(snapshot['timestamp'])
                singular_values.append(layer_metric['singular_values'])
        
        # Create violin plot
        sns.violinplot(data=singular_values, orient='h')
        plt.title('Distribution of Singular Values Across Layers')
        plt.xlabel('Singular Value')
        plt.ylabel('Layer')
        
        if save:
            plt.savefig(self.output_dir / 'eigenvalue_distribution.png')
        plt.close()
        
    def plot_ratio_vs_generalization(self, metrics_history, save=True):
        """Plot relationship between eigenvalue ratio and generalization gap"""
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        ratios = [m['avg_eigenvalue_ratio'] for m in metrics_history]
        gaps = [m['generalization_gap'] for m in metrics_history]
        
        # Create DataFrame for statistical analysis
        df = pd.DataFrame({
            'Eigenvalue Ratio': ratios,
            'Generalization Gap': gaps
        })
        
        # Calculate correlation and p-value
        correlation = df.corr().iloc[0,1]
        
        # Scatter plot with trend line
        sns.regplot(data=df, x='Eigenvalue Ratio', y='Generalization Gap')
        plt.title(f'Eigenvalue Ratio vs Generalization Gap\nCorrelation: {correlation:.3f}')
        
        if save:
            plt.savefig(self.output_dir / 'ratio_vs_generalization.png')
        plt.close()
        
    def plot_layer_heatmap(self, metrics_history, save=True):
        """Plot heatmap of eigenvalue ratios across layers"""
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Extract ratios per layer
        layer_names = []
        layer_ratios = []
        
        for snapshot in metrics_history:
            current_ratios = []
            for metric in snapshot['layer_metrics']:
                if metric['layer_name'] not in layer_names:
                    layer_names.append(metric['layer_name'])
                current_ratios.append(metric['eigenvalue_ratio'])
            layer_ratios.append(current_ratios)
        
        # Create heatmap
        sns.heatmap(
            np.array(layer_ratios).T,
            yticklabels=layer_names,
            xticklabels=list(range(len(metrics_history))),
            cmap='viridis'
        )
        plt.title('Eigenvalue Ratios Across Layers')
        plt.xlabel('Training Step')
        plt.ylabel('Layer')
        
        if save:
            plt.savefig(self.output_dir / 'layer_heatmap.png')
        plt.close()
        
    def plot_training_dynamics(self, metrics_history, save=True):
        """Plot training dynamics including loss, accuracy, and stable rank"""
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), dpi=self.dpi)
        
        steps = [m['timestamp'] for m in metrics_history]
        train_acc = [m['train_acc'] for m in metrics_history]
        val_acc = [m['val_acc'] for m in metrics_history]
        stable_ranks = [m['avg_stable_rank'] for m in metrics_history]
        
        # Plot accuracies
        ax1.plot(steps, train_acc, label='Train')
        ax1.plot(steps, val_acc, label='Validation')
        ax1.set_title('Accuracy over Training')
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot generalization gap
        gaps = [t - v for t, v in zip(train_acc, val_acc)]
        ax2.plot(steps, gaps)
        ax2.set_title('Generalization Gap over Training')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Gap')
        
        # Plot stable rank
        ax3.plot(steps, stable_ranks)
        ax3.set_title('Average Stable Rank over Training')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Stable Rank')
        
        plt.tight_layout()
        if save:
            plt.savefig(self.output_dir / 'training_dynamics.png')
        plt.close()
        
    def plot_layer_comparison(self, fft_metrics, lft_metrics, save=True):
        """Compare layer-wise metrics between FFT and LFT"""
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        
        # Extract final metrics for each method
        fft_ratios = [m['eigenvalue_ratio'] for m in fft_metrics[-1]['layer_metrics']]
        lft_ratios = [m['eigenvalue_ratio'] for m in lft_metrics[-1]['layer_metrics']]
        
        # Create bar plot
        x = np.arange(len(fft_ratios))
        width = 0.35
        
        plt.bar(x - width/2, fft_ratios, width, label='FFT')
        plt.bar(x + width/2, lft_ratios, width, label='LFT')
        
        plt.title('Layer-wise Eigenvalue Ratios: FFT vs LFT')
        plt.xlabel('Layer')
        plt.ylabel('Eigenvalue Ratio')
        plt.legend()
        
        if save:
            plt.savefig(self.output_dir / 'layer_comparison.png')
        plt.close()

    def generate_analysis_report(self, metrics_history, controlled_results):
        """Generate comprehensive analysis report with all plots"""
        # Create all standard plots
        self.plot_eigenvalue_distribution(metrics_history)
        self.plot_ratio_vs_generalization(metrics_history)
        self.plot_layer_heatmap(metrics_history)
        self.plot_training_dynamics(metrics_history)
        
        # Analyze controlled experiment results
        target_ratios = [r['target_ratio'] for r in controlled_results]
        val_accs = [r['val_metrics']['accuracy'] for r in controlled_results]
        
        plt.figure(figsize=self.figsize, dpi=self.dpi)
        plt.plot(target_ratios, val_accs, marker='o')
        plt.title('Validation Accuracy vs Target Ratio')
        plt.xlabel('Target Eigenvalue Ratio')
        plt.ylabel('Validation Accuracy')
        plt.savefig(self.output_dir / 'controlled_experiment.png')
        plt.close()
        
        # Save numerical results
        results_df = pd.DataFrame({
            'Target Ratio': target_ratios,
            'Validation Accuracy': val_accs
        })
        results_df.to_csv(self.output_dir / 'controlled_results.csv', index=False)