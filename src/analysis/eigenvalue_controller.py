import numpy as np
import torch
import copy

class EigenvalueController:
    """Class to artificially control eigenvalue ratios"""
    
    def modify_eigenvalues(self, weight_matrix, target_ratio, strength=0.1):
        """Modify weight matrix to achieve target eigenvalue ratio"""
        # Convert to numpy if tensor
        if torch.is_tensor(weight_matrix):
            weight_matrix = weight_matrix.detach().cpu().numpy()
            
        U, s, Vh = np.linalg.svd(weight_matrix, full_matrices=False)
        
        # Current ratio
        current_ratio = s[0] / np.sqrt(np.sum(s[1:]**2))
        
        # Calculate needed adjustment
        adjustment = (target_ratio - current_ratio) * strength
        s[0] *= (1 + adjustment)
        
        # Reconstruct matrix
        modified_weights = U @ np.diag(s) @ Vh
        return torch.tensor(modified_weights)

    def run_controlled_experiment(self, model, target_ratios, train_loader, val_loader):
        """Run experiment with controlled eigenvalue ratios"""
        results = []
        
        for target_ratio in target_ratios:
            # Create modified copy of model
            modified_model = copy.deepcopy(model)
            
            # Modify eigenvalues of each linear layer
            for name, module in modified_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    modified_weights = self.modify_eigenvalues(
                        module.weight, 
                        target_ratio
                    )
                    module.weight.data = modified_weights.to(module.weight.device)
            
            # Evaluate and record results
            result = {
                'target_ratio': target_ratio,
                'train_metrics': self.evaluate_model(modified_model, train_loader),
                'val_metrics': self.evaluate_model(modified_model, val_loader)
            }
            results.append(result)
        
        return results

    @staticmethod
    def evaluate_model(model, loader):
        """Evaluate model and return comprehensive metrics"""
        model.eval()
        correct = 0
        total = 0
        all_losses = []
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(next(model.parameters()).device)
                targets = targets.to(next(model.parameters()).device)
                
                outputs = model(inputs)
                loss = torch.nn.functional.cross_entropy(outputs, targets)
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                all_losses.append(loss.item())
        
        return {
            'accuracy': correct / total,
            'avg_loss': np.mean(all_losses),
            'loss_std': np.std(all_losses)
        }