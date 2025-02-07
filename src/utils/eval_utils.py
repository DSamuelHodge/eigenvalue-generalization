import torch
import numpy as np
from tqdm import tqdm
import time
from functools import wraps
from datasets import load_dataset

def track_memory_usage(func):
    """Decorator to track memory usage of experiments"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch.cuda.empty_cache()
        start_mem = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        
        result = func(*args, **kwargs)
        
        end_mem = torch.cuda.memory_allocated()
        peak_mem = torch.cuda.max_memory_allocated()
        print(f"Memory usage:")
        print(f"  Current: {(end_mem - start_mem)/1024/1024:.2f} MB")
        print(f"  Peak: {peak_mem/1024/1024:.2f} MB")
        return result
    return wrapper

@torch.no_grad()
def eval_with_eigenvalues(model, data_loader, analyzer, device="cuda"):
    """
    Evaluate model while tracking both performance and eigenvalue metrics
    """
    model.eval()
    model.to(device)
    
    metrics = {
        'loss': [],
        'eigenvalues': [],
        'memory_usage': []
    }
    
    for batch in tqdm(data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Track memory before forward pass
        start_mem = torch.cuda.memory_allocated()
        
        # Get model outputs
        outputs = model(**batch)
        
        # Calculate loss
        if torch.isfinite(outputs.logits).all():
            metrics['loss'].append(outputs.loss.item())
            
            # Get eigenvalue metrics
            layer_metrics = analyzer.analyze_model(
                model, 
                outputs.loss.item(),
                outputs.logits.mean().item()
            )
            metrics['eigenvalues'].append(layer_metrics)
            
            # Track memory usage
            end_mem = torch.cuda.memory_allocated()
            metrics['memory_usage'].append(end_mem - start_mem)
        
        torch.cuda.empty_cache()
    
    # Aggregate metrics
    return {
        'avg_loss': np.mean(metrics['loss']),
        'eigenvalue_metrics': metrics['eigenvalues'],
        'avg_memory': np.mean(metrics['memory_usage']) / 1024 / 1024  # Convert to MB
    }

@torch.no_grad()
def analyze_large_model_eigenvalues(model, analyzer, device="cuda"):
    """
    Analyze eigenvalues for large models layer by layer to manage memory
    """
    metrics = []
    
    def process_layer(layer_name, layer):
        layer = layer.to(device)
        metrics = analyzer.compute_layer_metrics(layer)
        layer = layer.cpu()
        torch.cuda.empty_cache()
        return {layer_name: metrics}
    
    # Process each model component separately
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            metrics.append(process_layer(name, module))
    
    return metrics

def get_test_data(dataset_name, tokenizer, seq_len=128, batch_size=32):
    """Load and prepare evaluation datasets"""
    if dataset_name == "wikitext2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    elif dataset_name == "c4":
        dataset = load_dataset("c4", "en", split="validation", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
        
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return torch.utils.data.DataLoader(
        tokenized_dataset,
        batch_size=batch_size,
        shuffle=False
    )