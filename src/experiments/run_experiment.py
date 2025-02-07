import torch
import yaml
import logging
from pathlib import Path
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer
from src.utils.data_handler import DataHandler
from src.analysis.eigenvalue_analyzer import EigenvalueAnalyzer
from src.analysis.eigenvalue_controller import EigenvalueController

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config_path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.analyzer = EigenvalueAnalyzer()
        self.controller = EigenvalueController()
        
    def setup(self):
        """Setup model and data"""
        logger.info("Setting up experiment...")
        
        # Initialize data handler
        self.data_handler = DataHandler(
            model_name=self.config['model']['name'],
            max_length=self.config['data']['max_length']
        )
        
        # Load dataset (using SST-2 as default)
        dataset = self.data_handler.load_glue_dataset("sst2")
        self.train_loader, self.eval_loader = self.data_handler.create_dataloaders(
            dataset,
            batch_size=self.config['data']['train_batch_size']
        )
        
        # Initialize model
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.config['model']['name'],
            num_labels=self.config['model']['num_labels']
        ).to(self.device)
        
    def run_fft_baseline(self):
        """Run full fine-tuning baseline"""
        logger.info("Running FFT baseline...")
        
        training_args = TrainingArguments(
            output_dir="./results/fft",
            learning_rate=self.config['training']['learning_rate'],
            num_train_epochs=self.config['training']['num_epochs'],
            per_device_train_batch_size=self.config['data']['train_batch_size'],
            per_device_eval_batch_size=self.config['data']['eval_batch_size'],
            warmup_steps=self.config['training']['warmup_steps'],
            weight_decay=self.config['training']['weight_decay'],
            logging_dir='./logs',
            logging_steps=100,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.train_loader.dataset,
            eval_dataset=self.eval_loader.dataset,
        )
        
        # Train and evaluate
        trainer.train()
        eval_results = trainer.evaluate()
        
        # Analyze eigenvalues
        self.analyzer.analyze_model(
            self.model,
            train_acc=trainer.state.log_history[-1]['train_loss'],
            val_acc=eval_results['eval_loss']
        )
        
        return eval_results
        
    def run_controlled_experiment(self):
        """Run experiment with controlled eigenvalue ratios"""
        logger.info("Running controlled experiment...")
        
        target_ratios = self.config['eigenvalue_analysis']['target_ratios']
        results = self.controller.run_controlled_experiment(
            model=self.model,
            target_ratios=target_ratios,
            train_loader=self.train_loader,
            val_loader=self.eval_loader
        )
        
        return results
    
    def analyze_results(self):
        """Analyze experimental results"""
        logger.info("Analyzing results...")
        
        correlations = self.analyzer.compute_correlations()
        logger.info(f"Correlations: {correlations}")
        
        # Save results
        Path("./results").mkdir(exist_ok=True)
        torch.save(self.analyzer.metrics_history, "./results/metrics_history.pt")
        
        return correlations

def main():
    experiment = ExperimentRunner("configs/experiment_config.yaml")
    experiment.setup()
    
    # Run baseline
    fft_results = experiment.run_fft_baseline()
    logger.info(f"FFT Results: {fft_results}")
    
    # Run controlled experiment
    controlled_results = experiment.run_controlled_experiment()
    logger.info(f"Controlled Results: {controlled_results}")
    
    # Analyze results
    analysis = experiment.analyze_results()
    logger.info(f"Analysis Results: {analysis}")

if __name__ == "__main__":
    main()