import pytest
import torch
import numpy as np
from src.analysis.eigenvalue_analyzer import EigenvalueAnalyzer
from src.analysis.eigenvalue_controller import EigenvalueController

@pytest.fixture
def sample_matrix():
    """Create a sample matrix with known eigenvalues"""
    # Create a matrix with singular values [2, 1, 0.5]
    U = torch.randn(3, 3)
    U, _ = torch.linalg.qr(U)  # Make U orthogonal
    V = torch.randn(3, 3)
    V, _ = torch.linalg.qr(V)  # Make V orthogonal
    s = torch.tensor([2.0, 1.0, 0.5])
    return U @ torch.diag(s) @ V.T

def test_eigenvalue_metrics(sample_matrix):
    analyzer = EigenvalueAnalyzer()
    metrics = analyzer.compute_layer_metrics(sample_matrix)
    
    # Known values for the sample matrix
    expected_stable_rank = (4 + 1 + 0.25) / 4  # (sum of squares) / (largest squared)
    expected_ratio = 2 / np.sqrt(1.25)  # largest / sqrt(sum of rest squared)
    
    assert np.abs(metrics['stable_rank'] - expected_stable_rank) < 1e-5
    assert np.abs(metrics['eigenvalue_ratio'] - expected_ratio) < 1e-5

def test_eigenvalue_modification():
    controller = EigenvalueController()
    
    # Create test matrix
    matrix = torch.tensor([[2.0, 0.0], [0.0, 1.0]])
    target_ratio = 3.0  # Want largest/rest = 3
    
    modified = controller.modify_eigenvalues(matrix, target_ratio)
    
    # Check if the modification achieved the target ratio
    s = torch.linalg.svd(modified, compute_uv=False)
    actual_ratio = s[0] / torch.sqrt(torch.sum(s[1:]**2))
    
    # Should be closer to target than original
    original_ratio = 2.0  # From original matrix
    assert abs(actual_ratio - target_ratio) < abs(original_ratio - target_ratio)

def test_controlled_experiment_shape():
    controller = EigenvalueController()
    
    # Create a minimal model for testing
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(10, 2)
            
        def forward(self, x):
            return self.linear(x)
    
    model = SimpleModel()
    
    # Create minimal dataloaders
    x = torch.randn(4, 10)
    y = torch.tensor([0, 1, 0, 1])
    dataloader = [(x, y)]
    
    # Run experiment
    results = controller.run_controlled_experiment(
        model=model,
        target_ratios=[1.0, 2.0],
        train_loader=dataloader,
        val_loader=dataloader
    )
    
    assert len(results) == 2  # One for each target ratio
    assert 'target_ratio' in results[0]
    assert 'train_metrics' in results[0]
    assert 'val_metrics' in results[0]

def test_analyzer_history():
    analyzer = EigenvalueAnalyzer()
    
    # Create a minimal model
    model = torch.nn.Linear(10, 2)
    
    # Add some metrics
    analyzer.analyze_model(model, train_acc=0.8, val_acc=0.7)
    analyzer.analyze_model(model, train_acc=0.85, val_acc=0.75)
    
    # Test correlation computation
    correlations = analyzer.compute_correlations()
    assert correlations is not None
    assert 'ratio_gap_correlation' in correlations
    assert 'ratio_accuracy_correlation' in correlations