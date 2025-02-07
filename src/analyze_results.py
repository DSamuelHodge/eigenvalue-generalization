import torch
import yaml
import logging
from pathlib import Path
from src.visualization.plot_utils import VisualizationUtils
from scipy import stats
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ResultAnalyzer:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.viz = VisualizationUtils(self.config)
        
    def load_results(self):
        """Load experimental results"""
        results_dir = Path("./results")
        
        # Load metrics history
        self.metrics_history = torch.load(results_dir / "metrics_history.pt")
        
        # Load controlled experiment results if they exist
        controlled_results_path = results_dir / "controlled_results.csv"
        if controlled_results_path.exists():
            self.controlled_results = pd.read_csv(controlled_results_path)
        else:
            self.controlled_results = None
            
    def analyze_eigenvalue_evolution(self):
        """Analyze how eigenvalues evolve during training"""
        logger.info("Analyzing eigenvalue evolution...")
        
        # Plot distributions
        self.viz.plot_eigenvalue_distribution(self.metrics_history)
        
        # Analyze trends
        ratios = [m['avg_eigenvalue_ratio'] for m in self.metrics_history]
        stable_ranks = [m['avg_stable_rank'] for m in self.metrics_history]
        
        # Compute trends
        ratio_trend = np.polyfit(range(len(ratios)), ratios, 1)[0]
        rank_trend = np.polyfit(range(len(stable_ranks)), stable_ranks, 1)[0]
        
        logger.info(f"Ratio trend: {ratio_trend:.4f}")
        logger.info(f"Stable rank trend: {rank_trend:.4f}")
        
        return {
            'ratio_trend': ratio_trend,
            'rank_trend': rank_trend
        }
        
    def analyze_generalization(self):
        """Analyze relationship with generalization"""
        logger.info("Analyzing generalization relationship...")
        
        # Extract metrics
        ratios = [m['avg_eigenvalue_ratio'] for m in self.metrics_history]
        gaps = [m['generalization_gap'] for m in self.metrics_history]
        
        # Compute correlation
        correlation, p_value = stats.pearsonr(ratios, gaps)
        
        # Plot relationship
        self.viz.plot_ratio_vs_generalization(self.metrics_history)
        
        logger.info(f"Correlation: {correlation:.4f} (p={p_value:.4f})")
        
        return {
            'correlation': correlation,
            'p_value': p_value
        }
        
    def analyze_layer_differences(self):
        """Analyze differences between layers"""
        logger.info("Analyzing layer differences...")
        
        # Plot layer heatmap
        self.viz.plot_layer_heatmap(self.metrics_history)
        
        # Analyze layer-wise patterns
        final_metrics = self.metrics_history[-1]['layer_metrics']
        layer_ratios = [m['eigenvalue_ratio'] for m in final_metrics]
        
        # Test for layer-wise differences
        f_stat, anova_p = stats.f_oneway(*layer_ratios)
        
        logger.info(f"Layer difference ANOVA: F={f_stat:.4f}, p={anova_p:.4f}")
        
        return {
            'f_statistic': f_stat,
            'p_value': anova_p
        }
        
    def analyze_controlled_experiment(self):
        """Analyze results from controlled experiment"""
        if self.controlled_results is None:
            logger.info("No controlled experiment results found")
            return None
            
        logger.info("Analyzing controlled experiment results...")
        
        # Find optimal ratio
        optimal_idx = self.controlled_results['Validation Accuracy'].idxmax()
        optimal_ratio = self.controlled_results.loc[optimal_idx, 'Target Ratio']
        
        # Fit regression
        X = self.controlled_results['Target Ratio'].values.reshape(-1, 1)
        y = self.controlled_results['Validation Accuracy'].values
        
        reg = stats.linregress(X.flatten(), y)
        
        logger.info(f"Optimal ratio: {optimal_ratio:.4f}")
        logger.info(f"Regression slope: {reg.slope:.4f} (p={reg.pvalue:.4f})")
        
        return {
            'optimal_ratio': optimal_ratio,
            'regression': reg._asdict()
        }
        
    def generate_report(self):
        """Generate comprehensive analysis report"""
        logger.info("Generating analysis report...")
        
        results = {
            'eigenvalue_evolution': self.analyze_eigenvalue_evolution(),
            'generalization': self.analyze_generalization(),
            'layer_differences': self.analyze_layer_differences(),
            'controlled_experiment': self.analyze_controlled_experiment()
        }
        
        # Save numerical results
        output_dir = Path("./results/analysis")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "analysis_results.yaml", "w") as f:
            yaml.dump(results, f)
            
        # Generate plots
        self.viz.generate_analysis_report(self.metrics_history, self.controlled_results)
        
        logger.info("Analysis complete. Results saved to ./results/analysis/")
        
        return results

def main():
    analyzer = ResultAnalyzer("configs/experiment_config.yaml")
    analyzer.load_results()
    results = analyzer.generate_report()
    
if __name__ == "__main__":
    main()