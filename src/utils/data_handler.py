from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

class DataHandler:
    def __init__(self, model_name="bert-base-uncased", max_length=128):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length

    def load_glue_dataset(self, task_name="sst2"):
        """Load and prepare GLUE dataset
        Using SST-2 by default as it's a binary classification task
        with reasonable size for quick experiments"""
        dataset = load_dataset("glue", task_name)
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["sentence"],
                padding="max_length",
                truncation=True,
                max_length=self.max_length
            )

        tokenized_datasets = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )

        return tokenized_datasets

    def create_dataloaders(self, dataset, batch_size=32):
        """Create PyTorch DataLoaders"""
        train_dataloader = DataLoader(
            dataset["train"],
            shuffle=True,
            batch_size=batch_size
        )
        
        eval_dataloader = DataLoader(
            dataset["validation"],
            batch_size=batch_size
        )
        
        return train_dataloader, eval_dataloader