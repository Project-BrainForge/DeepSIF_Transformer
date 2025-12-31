"""
Optimized configuration for DeepSIF Transformer to prevent gradient issues and overfitting
"""

import torch
from torch import optim
import torch.nn as nn
import numpy as np


class OptimizedConfig:
    """Optimized hyperparameters and training configuration"""
    
    def __init__(self):
        # Model Architecture (Optimized for stability)
        self.model_config = {
            'num_sensor': 75,
            'num_source': 994,
            'transformer_layers': 4,  # Optimal depth - not too deep to avoid vanishing gradients
            'd_model': 256,  # Balanced size - not too large to avoid overfitting
            'nhead': 8,  # Good for 256 dimensions (32 per head)
            'dropout': 0.15,  # Higher dropout to prevent overfitting
            'spatial_activation': 'GELU',  # Better than ReLU for transformers
            'temporal_activation': 'GELU',
            'temporal_input_size': 500
        }
        
        # Training Hyperparameters (Optimized)
        self.training_config = {
            'learning_rate': 1e-4,  # More conservative LR for stability
            'batch_size': 8,  # Even smaller batch size for stability
            'epochs': 100,
            'warmup_epochs': 5,  # Shorter warmup
            'weight_decay': 1e-4,  # L2 regularization
            'gradient_clip': 0.5,  # Tighter gradient clipping
            'patience': 15,  # Early stopping patience
            'min_lr': 1e-6  # Minimum learning rate for scheduling
        }
        
        # Data Augmentation/Regularization
        self.data_config = {
            'train_split': 0.8,
            'val_split': 0.1,
            'test_split': 0.1,
            'noise_augmentation': True,  # Add noise during training
            'noise_std': 0.01,  # Standard deviation for noise
            'temporal_shift': True,  # Random temporal shifts
            'max_shift': 5  # Maximum samples to shift
        }
        
        # Logging configuration
        self.logging_config = {
            'level': 'INFO',  # DEBUG, INFO, WARNING, ERROR
            'log_frequency': {
                'batch_progress': 1,      # Log every N batches during training
                'batch_summary': 50,       # Detailed batch summary every N batches
                'memory_check': 100,       # GPU memory check every N batches
                'validation_progress': 20,  # Validation progress every N batches
                'checkpoint_save': 1,     # Save checkpoint every N epochs
                'curve_update': 20         # Update training curves every N epochs
            }
        }
        
        # Loss function configuration (More conservative weights)
        self.loss_config = {
            'primary_loss': 'mse',
            'loss_weights': {
                'reconstruction': 1.0,
                'sparsity': 0.001,  # Much lower sparsity weight for stability
                'temporal_smoothness': 0.0001  # Lower smoothness weight
            }
        }


def create_optimized_model():
    """Create model with optimized architecture"""
    from network import TransformerTemporalInverseNet, MLPSpatialFilter, TransformerTemporalFilter
    
    config = OptimizedConfig()
    
    model = TransformerTemporalInverseNet(
        **config.model_config,
        spatial_model=MLPSpatialFilter,
        temporal_model=TransformerTemporalFilter,
        spatial_output='value_activation',
        temporal_output='transformer'
    )
    
    return model, config


def create_optimized_optimizer(model, config):
    """Create optimizer with optimal settings"""
    
    # Separate parameters for different learning rates
    transformer_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'transformer' in name:
            transformer_params.append(param)
        else:
            other_params.append(param)
    
    # Use AdamW (better than Adam for transformers)
    optimizer = optim.AdamW([
        {'params': transformer_params, 'lr': config.training_config['learning_rate'], 'weight_decay': config.training_config['weight_decay']},
        {'params': other_params, 'lr': config.training_config['learning_rate'] * 2, 'weight_decay': config.training_config['weight_decay'] * 0.5}
    ], betas=(0.9, 0.95), eps=1e-8)  # Optimized betas for transformers
    
    return optimizer


def create_lr_scheduler(optimizer, config, steps_per_epoch):
    """Create learning rate scheduler with warmup and cosine annealing"""
    
    total_steps = config.training_config['epochs'] * steps_per_epoch
    warmup_steps = config.training_config['warmup_epochs'] * steps_per_epoch
    
    def lr_lambda(step):
        if step < warmup_steps:
            # Linear warmup starting from 0.1 (not 0!)
            return 0.1 + 0.9 * (step / warmup_steps)
        else:
            # Cosine annealing
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def create_optimized_loss_function(config):
    """Create multi-component loss function"""
    
    class OptimizedLoss(nn.Module):
        def __init__(self, loss_config):
            super().__init__()
            self.loss_config = loss_config
            self.mse_loss = nn.MSELoss()
            self.l1_loss = nn.L1Loss()
            
        def forward(self, predictions, targets):
            # Primary reconstruction loss
            recon_loss = self.mse_loss(predictions, targets)
            
            # Sparsity regularization (L1 on source activities)
            sparsity_loss = self.l1_loss(predictions, torch.zeros_like(predictions))
            
            # Temporal smoothness (encourage smooth temporal evolution)
            temporal_diff = predictions[:, 1:] - predictions[:, :-1]
            smoothness_loss = torch.mean(temporal_diff ** 2)
            
            # Combined loss
            total_loss = (
                self.loss_config['loss_weights']['reconstruction'] * recon_loss +
                self.loss_config['loss_weights']['sparsity'] * sparsity_loss +
                self.loss_config['loss_weights']['temporal_smoothness'] * smoothness_loss
            )
            
            return total_loss, {
                'reconstruction': recon_loss.item(),
                'sparsity': sparsity_loss.item(),
                'smoothness': smoothness_loss.item(),
                'total': total_loss.item()
            }
    
    return OptimizedLoss(config.loss_config)


def add_noise_augmentation(data, config):
    """Add noise augmentation during training"""
    if config.data_config['noise_augmentation'] and torch.rand(1) > 0.5:
        noise = torch.randn_like(data) * config.data_config['noise_std']
        data = data + noise
    return data


def add_temporal_shift(data, config):
    """Add random temporal shifts for data augmentation"""
    if config.data_config['temporal_shift'] and torch.rand(1) > 0.5:
        batch_size, seq_len, num_features = data.shape
        max_shift = min(config.data_config['max_shift'], seq_len // 10)
        
        # Clone the data to avoid in-place operation issues
        shifted_data = data.clone()
        
        for i in range(batch_size):
            shift = torch.randint(-max_shift, max_shift + 1, (1,)).item()
            if shift > 0:
                # Right shift (data moves right, zeros on left)
                shifted_data[i, shift:] = data[i, :-shift]
                shifted_data[i, :shift] = 0
            elif shift < 0:
                # Left shift (data moves left, zeros on right)  
                shift = abs(shift)
                shifted_data[i, :-shift] = data[i, shift:]
                shifted_data[i, -shift:] = 0
        
        return shifted_data
    return data


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    
    def __init__(self, patience=15, min_delta=1e-6, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False


def apply_gradient_clipping(model, max_norm=1.0):
    """Apply gradient clipping to prevent gradient explosion"""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
    return grad_norm.item() if hasattr(grad_norm, 'item') else float(grad_norm)


# Training best practices summary
TRAINING_BEST_PRACTICES = {
    'model_architecture': {
        'use_pre_norm': True,
        'use_gelu_activation': True,
        'moderate_depth': '4-6 layers optimal',
        'moderate_width': '256-512 d_model',
        'proper_dropout': '0.1-0.2 range',
        'layer_normalization': 'Essential for stability'
    },
    
    'training_strategy': {
        'learning_rate': '3e-4 to 1e-3 range',
        'optimizer': 'AdamW with weight decay',
        'lr_scheduling': 'Warmup + Cosine annealing',
        'gradient_clipping': '1.0 max norm',
        'batch_size': '16-32 for better generalization',
        'early_stopping': 'Monitor validation loss'
    },
    
    'regularization': {
        'dropout': '0.15 in transformer layers',
        'weight_decay': '1e-4 to 1e-3',
        'data_augmentation': 'Noise + temporal shifts',
        'label_smoothing': 'Optional for classification',
        'multi_component_loss': 'MSE + sparsity + smoothness'
    },
    
    'monitoring': {
        'track_gradients': 'Monitor gradient norms',
        'track_weights': 'Monitor weight magnitudes',
        'validation_metrics': 'Loss, correlation, precision/recall',
        'learning_curves': 'Plot train/val loss curves'
    }
}
