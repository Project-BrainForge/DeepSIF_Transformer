#!/usr/bin/env python3
"""
Optimized training script for DeepSIF Transformer with gradient stability and overfitting prevention
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import logging
import datetime
import matplotlib.pyplot as plt
import sys

import network
import loaders
from config_optimized import (
    OptimizedConfig, create_optimized_model, create_optimized_optimizer,
    create_lr_scheduler, create_optimized_loss_function, EarlyStopping,
    apply_gradient_clipping, add_noise_augmentation, add_temporal_shift
)


def safe_log_message(logger, level, message):
    """Safely log a message, handling potential Unicode issues"""
    try:
        # Try to log the message as-is
        getattr(logger, level)(message)
    except UnicodeEncodeError:
        # If that fails, strip emojis and try again
        import re
        # Remove all emoji characters
        emoji_pattern = re.compile("["
                                 "\U0001F600-\U0001F64F"  # emoticons
                                 "\U0001F300-\U0001F5FF"  # symbols & pictographs
                                 "\U0001F680-\U0001F6FF"  # transport & map symbols
                                 "\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                 "]+", flags=re.UNICODE)
        safe_message = emoji_pattern.sub('', message).strip()
        getattr(logger, level)(safe_message)


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length valid_labels"""
    valid_labels = [item['valid_labels'] for item in batch]
    
    batch_no_valid_labels = []
    for item in batch:
        new_item = {k: v for k, v in item.items() if k != 'valid_labels'}
        batch_no_valid_labels.append(new_item)
    
    collated = torch.utils.data.default_collate(batch_no_valid_labels)
    collated['valid_labels'] = valid_labels
    
    return collated


class SafeFormatter(logging.Formatter):
    """Formatter that handles Unicode safely for Windows console"""
    
    def __init__(self, fmt, use_emojis=True):
        super().__init__(fmt)
        self.use_emojis = use_emojis
        
        # Emoji mapping for console-safe alternatives
        self.emoji_replacements = {
            '🚀': '[START]',
            '🔄': '[EPOCH]', 
            '📊': '[INFO]',
            '✅': '[OK]',
            '📈': '[UP]',
            '📉': '[DOWN]',
            '💾': '[SAVE]',
            '⚠️': '[WARN]',
            '🎉': '[DONE]',
            '🛑': '[STOP]',
            '🔍': '[NEXT]',
            '📁': '[FILE]'
        }
    
    def format(self, record):
        # Get the formatted message
        message = super().format(record)
        
        # Replace emojis with safe alternatives if needed
        if not self.use_emojis:
            for emoji, replacement in self.emoji_replacements.items():
                message = message.replace(emoji, replacement)
        
        return message


def setup_logging(result_root, model_name, log_level='INFO', quiet=False):
    """Setup comprehensive logging with encoding safety"""
    os.makedirs(result_root, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('DeepSIF_Optimized')
    logger.setLevel(logging.DEBUG)  # Always DEBUG for file logging
    
    # Clear any existing handlers
    logger.handlers.clear()
    
    # File handler for detailed logs (always DEBUG level, with UTF-8 encoding)
    log_file = os.path.join(result_root, f'training_{model_name}.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler with configurable level and safe encoding
    console_handler = logging.StreamHandler(sys.stdout)
    console_level = getattr(logging, log_level.upper())
    
    if quiet:
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors
    else:
        console_handler.setLevel(console_level)
    
    # Check if console supports UTF-8 (emojis)
    try:
        # Try to encode a test emoji
        test_emoji = '🚀'
        test_emoji.encode(sys.stdout.encoding or 'utf-8')
        console_supports_emojis = True
    except (UnicodeEncodeError, AttributeError):
        console_supports_emojis = False
    
    # Detailed formatter for file (with emojis)
    detailed_formatter = SafeFormatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        use_emojis=True
    )
    file_handler.setFormatter(detailed_formatter)
    
    # Simple formatter for console (emojis based on support)
    simple_formatter = SafeFormatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        use_emojis=console_supports_emojis
    )
    console_handler.setFormatter(simple_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    # Log system info
    logger.info("="*80)
    logger.info("DEEPSIF TRANSFORMER TRAINING SESSION STARTED")
    logger.info("="*80)
    logger.info(f"Log file: {log_file}")
    logger.info(f"Result directory: {result_root}")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device count: {torch.cuda.device_count()}")
        logger.info(f"Current CUDA device: {torch.cuda.current_device()}")
        logger.info(f"CUDA device name: {torch.cuda.get_device_name()}")
    
    return logger


def monitor_gradients(model, logger, epoch):
    """Monitor gradient norms for debugging"""
    total_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            param_count += 1
            
    total_norm = total_norm ** (1. / 2)
    
    if epoch % 10 == 0:  # Log every 10 epochs
        logger.info(f"Epoch {epoch} - Gradient norm: {total_norm:.6f}")
        
    return total_norm


def train_epoch(model, train_loader, optimizer, loss_function, config, device, logger, epoch):
    """Optimized training epoch with comprehensive logging"""
    epoch_start_time = time.time()
    model.train()
    
    epoch_losses = []
    epoch_metrics = {'reconstruction': [], 'sparsity': [], 'smoothness': [], 'total': []}
    gradient_norms = []
    learning_rates = []
    
    logger.info(f"Starting training epoch {epoch}")
    logger.debug(f"Training on {len(train_loader)} batches")
    
    for batch_idx, batch in enumerate(train_loader):
        batch_start_time = time.time()
        
        # Move data to device and log data info
        data = batch['data'].to(device)
        target = batch['nmm'].to(device)
        
        if batch_idx == 0:  # Log data shapes for first batch
            logger.debug(f"Input data shape: {data.shape}")
            logger.debug(f"Target data shape: {target.shape}")
            logger.debug(f"Data device: {data.device}")
            logger.debug(f"Data dtype: {data.dtype}")
        
        # Data augmentation
        augmentation_applied = []
        if model.training:
            original_data = data.clone()
            data = add_noise_augmentation(data, config)
            if not torch.equal(original_data, data):
                augmentation_applied.append("noise")
                
            data = add_temporal_shift(data, config)
            if not torch.equal(original_data, data):
                augmentation_applied.append("temporal_shift")
        
        if batch_idx == 0 and augmentation_applied:
            logger.debug(f"Data augmentations applied: {', '.join(augmentation_applied)}")
        
        # Forward pass
        forward_start = time.time()
        optimizer.zero_grad()
        output = model(data)
        forward_time = time.time() - forward_start
        
        if batch_idx == 0:
            logger.debug(f"Model output shape: {output['last'].shape}")
            logger.debug(f"Forward pass time: {forward_time:.4f}s")
        
        # Calculate loss
        loss_start = time.time()
        loss, loss_components = loss_function(output['last'], target)
        loss_time = time.time() - loss_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        backward_time = time.time() - backward_start
        
        # Gradient clipping and monitoring
        grad_norm = apply_gradient_clipping(model, config.training_config['gradient_clip'])
        if grad_norm is not None:
            gradient_norms.append(grad_norm)
        else:
            gradient_norms.append(0.0)
        
        # Optimizer step
        optimizer.step()
        
        # Track learning rate
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates.append(current_lr)
        
        # Track metrics
        epoch_losses.append(loss.item())
        for key, value in loss_components.items():
            epoch_metrics[key].append(value)
        
        batch_time = time.time() - batch_start_time
        
        # Detailed logging every 25 batches, summary every 50
        if batch_idx % 25 == 0:
            grad_norm_str = f"{grad_norm:.6f}" if grad_norm is not None else "N/A"
            
            if batch_idx % 50 == 0:
                # Full summary log
                logger.info(
                    f"Epoch {epoch:3d}, Batch {batch_idx:4d}/{len(train_loader)} | "
                    f"Loss: {loss.item():.6f} | GradNorm: {grad_norm_str} | "
                    f"LR: {current_lr:.2e} | Time: {batch_time:.3f}s"
                )
                logger.debug(f"Loss breakdown - Recon: {loss_components['reconstruction']:.6f}, "
                            f"Sparsity: {loss_components['sparsity']:.6f}, "
                            f"Smooth: {loss_components['smoothness']:.6f}")
            else:
                # Brief log
                logger.debug(
                    f"Batch {batch_idx:4d}: Loss={loss.item():.6f}, "
                    f"GradNorm={grad_norm_str}, Time={batch_time:.3f}s"
                )
        
        # Memory logging every 100 batches
        if batch_idx % 100 == 0 and torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            memory_reserved = torch.cuda.memory_reserved(device) / 1024**3    # GB
            logger.debug(f"GPU Memory - Allocated: {memory_allocated:.2f}GB, "
                        f"Reserved: {memory_reserved:.2f}GB")
    
    # Calculate epoch averages
    avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
    avg_gradient_norm = np.mean(gradient_norms) if gradient_norms else 0.0
    avg_lr = np.mean(learning_rates) if learning_rates else 0.0
    
    epoch_time = time.time() - epoch_start_time
    
    # Comprehensive epoch summary
    logger.info("="*60)
    logger.info(f"EPOCH {epoch} TRAINING SUMMARY")
    logger.info("="*60)
    logger.info(f"Total time: {epoch_time:.2f}s")
    logger.info(f"Average batch time: {epoch_time/len(train_loader):.3f}s")
    logger.info(f"Samples processed: {len(train_loader) * config.training_config['batch_size']}")
    logger.info(f"Average learning rate: {avg_lr:.2e}")
    logger.info(f"Average gradient norm: {avg_gradient_norm:.6f}")
    logger.info(f"Loss components:")
    for key, value in avg_metrics.items():
        logger.info(f"  {key}: {value:.6f}")
    
    # Gradient health check
    if avg_gradient_norm > 5.0:
        logger.warning(f"High gradient norm detected: {avg_gradient_norm:.6f}")
    elif avg_gradient_norm < 1e-5:
        logger.warning(f"Very low gradient norm detected: {avg_gradient_norm:.6f}")
    else:
        logger.debug("Gradient norms are healthy")
    
    # Monitor gradients
    monitor_gradients(model, logger, epoch)
    
    return avg_metrics, avg_gradient_norm


def validate_epoch(model, val_loader, loss_function, device, logger, epoch):
    """Validation epoch with comprehensive logging"""
    val_start_time = time.time()
    model.eval()
    
    val_losses = []
    val_metrics = {'reconstruction': [], 'sparsity': [], 'smoothness': [], 'total': []}
    
    logger.info(f"Starting validation for epoch {epoch}")
    logger.debug(f"Validating on {len(val_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            batch_start_time = time.time()
            
            data = batch['data'].to(device)
            target = batch['nmm'].to(device)
            
            # Forward pass
            output = model(data)
            loss, loss_components = loss_function(output['last'], target)
            
            val_losses.append(loss.item())
            for key, value in loss_components.items():
                val_metrics[key].append(value)
            
            batch_time = time.time() - batch_start_time
            
            # Log validation progress
            if batch_idx % 20 == 0:
                logger.debug(f"Validation batch {batch_idx:3d}/{len(val_loader)}: "
                           f"Loss={loss.item():.6f}, Time={batch_time:.3f}s")
    
    # Calculate averages
    avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
    val_time = time.time() - val_start_time
    
    # Validation summary
    logger.info("-"*60)
    logger.info(f"EPOCH {epoch} VALIDATION SUMMARY")
    logger.info("-"*60)
    logger.info(f"Total validation time: {val_time:.2f}s")
    logger.info(f"Average batch time: {val_time/len(val_loader):.3f}s")
    logger.info(f"Validation loss components:")
    for key, value in avg_metrics.items():
        logger.info(f"  {key}: {value:.6f}")
    
    return avg_metrics


def plot_training_curves(train_history, val_history, result_root):
    """Plot training curves for monitoring"""
    epochs = range(1, len(train_history['total']) + 1)
    
    plt.figure(figsize=(15, 10))
    
    # Total loss
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_history['total'], 'b-', label='Train')
    plt.plot(epochs, val_history['total'], 'r-', label='Validation')
    plt.title('Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Reconstruction loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_history['reconstruction'], 'b-', label='Train')
    plt.plot(epochs, val_history['reconstruction'], 'r-', label='Validation')
    plt.title('Reconstruction Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Sparsity loss
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_history['sparsity'], 'b-', label='Train')
    plt.plot(epochs, val_history['sparsity'], 'r-', label='Validation')
    plt.title('Sparsity Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Smoothness loss
    plt.subplot(2, 2, 4)
    plt.plot(epochs, train_history['smoothness'], 'b-', label='Train')
    plt.plot(epochs, val_history['smoothness'], 'r-', label='Validation')
    plt.title('Temporal Smoothness Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(result_root, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Optimized DeepSIF Transformer Training')
    parser.add_argument('--data_path', default='labeled_dataset', type=str, help='Path to labeled dataset')
    parser.add_argument('--model_id', default='optimized', type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode with smaller dataset')
    parser.add_argument('--log_level', default='INFO', type=str, choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help='Logging level (DEBUG for detailed logs)')
    parser.add_argument('--quiet', action='store_true', help='Reduce console output (logs still saved to file)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config = OptimizedConfig()
    
    result_root = f'model_result/{args.model_id}_optimized_transformer'
    logger = setup_logging(result_root, args.model_id, args.log_level, args.quiet)
    
    logger.info(f"Starting optimized training on device: {device}")
    logger.info(f"Configuration: {config.model_config}")
    
    # Load data
    logger.info("Loading dataset...")
    import glob
    all_files = glob.glob(os.path.join(args.data_path, "sample_*.mat"))
    all_files.sort()
    
    logger.info(f"Found {len(all_files)} data files in {args.data_path}")
    
    if len(all_files) == 0:
        logger.error(f"No sample_*.mat files found in {args.data_path}")
        logger.error("Please ensure your labeled dataset is in the correct directory")
        return
    
    if args.debug:
        all_files = all_files[:100]  # Use only 100 samples for debugging
        logger.warning("DEBUG MODE: Using only 100 samples for faster testing")
    
    # Split data
    n_total = len(all_files)
    n_train = int(n_total * config.data_config['train_split'])
    n_val = int(n_total * config.data_config['val_split'])
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    test_files = all_files[n_train + n_val:]
    
    logger.info("Dataset split details:")
    logger.info(f"  Total files: {n_total}")
    logger.info(f"  Training: {len(train_files)} ({len(train_files)/n_total*100:.1f}%)")
    logger.info(f"  Validation: {len(val_files)} ({len(val_files)/n_total*100:.1f}%)")
    logger.info(f"  Test: {len(test_files)} ({len(test_files)/n_total*100:.1f}%)")
    
    # Log some example filenames
    logger.debug(f"First training file: {train_files[0] if train_files else 'None'}")
    logger.debug(f"First validation file: {val_files[0] if val_files else 'None'}")
    logger.debug(f"First test file: {test_files[0] if test_files else 'None'}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    logger.debug("Initializing training dataset...")
    train_dataset = loaders.LabeledDatasetLoader(
        args.data_path, fwd=None, args_params={'dataset_len': len(train_files)}
    )
    train_dataset.file_list = train_files
    
    logger.debug("Initializing validation dataset...")
    val_dataset = loaders.LabeledDatasetLoader(
        args.data_path, fwd=None, args_params={'dataset_len': len(val_files)}
    )
    val_dataset.file_list = val_files
    
    # Test loading one sample to verify data format
    logger.debug("Testing data loading...")
    try:
        sample = train_dataset[0]
        logger.info(f"Data loading test successful:")
        logger.info(f"  EEG data shape: {sample['data'].shape}")
        logger.info(f"  Source data shape: {sample['nmm'].shape}")
        logger.info(f"  Valid labels: {len(sample['valid_labels'])} active sources")
        logger.info(f"  SNR: {sample['snr']}")
    except Exception as e:
        logger.error(f"Data loading test failed: {e}")
        return
    
    logger.debug("Creating data loaders with batch processing...")
    train_loader = DataLoader(
        train_dataset, batch_size=config.training_config['batch_size'],
        shuffle=True, collate_fn=custom_collate_fn, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.training_config['batch_size'],
        shuffle=False, collate_fn=custom_collate_fn, num_workers=0
    )
    
    logger.info(f"Data loaders created:")
    logger.info(f"  Batch size: {config.training_config['batch_size']}")
    logger.info(f"  Training batches: {len(train_loader)}")
    logger.info(f"  Validation batches: {len(val_loader)}")
    logger.info(f"  Samples per epoch: {len(train_loader) * config.training_config['batch_size']}")
    
    # Create model
    logger.info("Creating optimized model...")
    model, _ = create_optimized_model()
    model = model.to(device)
    
    # Log model details
    total_params = model.count_parameters()
    logger.info("Model architecture details:")
    logger.info(f"  Architecture: {type(model).__name__}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Model size (MB): {total_params * 4 / 1024**2:.2f}")  # 4 bytes per float32 param
    logger.info(f"  Device: {device}")
    
    # Log model configuration
    logger.debug("Model configuration:")
    for key, value in config.model_config.items():
        logger.debug(f"  {key}: {value}")
    
    # Create optimizer and scheduler
    logger.info("Setting up training components...")
    
    logger.debug("Creating optimizer...")
    optimizer = create_optimized_optimizer(model, config)
    logger.info(f"Optimizer: {type(optimizer).__name__}")
    logger.info(f"Learning rate: {config.training_config['learning_rate']}")
    logger.info(f"Weight decay: {config.training_config['weight_decay']}")
    
    logger.debug("Creating learning rate scheduler...")
    scheduler = create_lr_scheduler(optimizer, config, len(train_loader))
    
    logger.debug("Creating loss function...")
    loss_function = create_optimized_loss_function(config)
    logger.info("Loss function: Multi-component (MSE + Sparsity + Smoothness)")
    logger.debug("Loss weights:")
    for key, weight in config.loss_config['loss_weights'].items():
        logger.debug(f"  {key}: {weight}")
    
    logger.debug("Setting up early stopping...")
    early_stopping = EarlyStopping(patience=config.training_config['patience'])
    logger.info(f"Early stopping patience: {config.training_config['patience']} epochs")
    
    # Training history
    train_history = {'total': [], 'reconstruction': [], 'sparsity': [], 'smoothness': []}
    val_history = {'total': [], 'reconstruction': [], 'sparsity': [], 'smoothness': []}
    gradient_norms = []
    
    # Resume training if specified
    start_epoch = 0
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming training from checkpoint: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            
            logger.info(f"Successfully resumed from epoch {start_epoch}")
            logger.info(f"Previous best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
            
            # Load training history if available
            if 'train_history' in checkpoint:
                train_history = checkpoint['train_history']
                val_history = checkpoint['val_history']
                logger.info(f"Loaded training history: {len(train_history['total'])} previous epochs")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            logger.info("Starting fresh training instead...")
            start_epoch = 0
    elif args.resume:
        logger.warning(f"Checkpoint file not found: {args.resume}")
        logger.info("Starting fresh training...")
        
    best_val_loss = float('inf')
    
    # Training loop
    logger.info("🚀 STARTING TRAINING LOOP")
    logger.info(f"Total epochs planned: {config.training_config['epochs']}")
    logger.info(f"Starting from epoch: {start_epoch}")
    logger.info(f"Estimated training time: {(config.training_config['epochs'] - start_epoch) * len(train_loader) * 0.1 / 60:.1f} minutes")
    
    start_time = time.time()
    
    try:
        for epoch in range(start_epoch, config.training_config['epochs']):
            epoch_start_time = time.time()
            
            logger.info("🔄 " + "="*80)
            logger.info(f"🔄 STARTING EPOCH {epoch}/{config.training_config['epochs'] - 1}")
            logger.info("🔄 " + "="*80)
            
            # Training
            train_metrics, avg_grad_norm = train_epoch(
                model, train_loader, optimizer, loss_function, config, device, logger, epoch
            )
            
            # Validation
            val_metrics = validate_epoch(model, val_loader, loss_function, device, logger, epoch)
            
            # Update learning rate
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            if abs(old_lr - new_lr) > 1e-8:
                logger.info(f"📉 Learning rate updated: {old_lr:.2e} → {new_lr:.2e}")
            
            # Record history
            for key in train_history.keys():
                train_history[key].append(train_metrics[key])
                val_history[key].append(val_metrics[key])
            gradient_norms.append(avg_grad_norm)
            
            # Calculate improvement metrics
            train_improvement = ""
            val_improvement = ""
            if len(train_history['total']) > 1:
                train_diff = train_history['total'][-1] - train_history['total'][-2]
                val_diff = val_history['total'][-1] - val_history['total'][-2]
                train_improvement = f"({train_diff:+.6f})" if train_diff != 0 else ""
                val_improvement = f"({val_diff:+.6f})" if val_diff != 0 else ""
            
            # Epoch summary logging
            epoch_time = time.time() - epoch_start_time
            total_elapsed = time.time() - start_time
            remaining_epochs = config.training_config['epochs'] - epoch - 1
            estimated_remaining = remaining_epochs * (total_elapsed / (epoch - start_epoch + 1)) if epoch > start_epoch else 0
            
            grad_norm_display = f"{avg_grad_norm:.6f}" if avg_grad_norm is not None else "N/A"
            
            logger.info("📊 " + "="*80)
            logger.info("📊 EPOCH SUMMARY")
            logger.info("📊 " + "="*80)
            logger.info(f"📊 Epoch: {epoch:3d}/{config.training_config['epochs']-1}")
            logger.info(f"📊 Time: {epoch_time:6.2f}s (Total: {total_elapsed/60:.1f}min, Est. remaining: {estimated_remaining/60:.1f}min)")
            logger.info(f"📊 Learning Rate: {new_lr:.2e}")
            logger.info(f"📊 Gradient Norm: {grad_norm_display}")
            logger.info(f"📊 Train Loss: {train_metrics['total']:.6f} {train_improvement}")
            logger.info(f"📊 Val Loss: {val_metrics['total']:.6f} {val_improvement}")
            
            # Performance indicators
            if len(val_history['total']) > 1:
                if val_history['total'][-1] < val_history['total'][-2]:
                    logger.info("📈 Validation improving!")
                elif val_history['total'][-1] > val_history['total'][-2]:
                    logger.info("📉 Validation worsening")
            
            # Save best model
            is_best = val_metrics['total'] < best_val_loss
            if is_best:
                improvement = best_val_loss - val_metrics['total']
                best_val_loss = val_metrics['total']
                
                logger.info("💾 " + "="*60)
                logger.info("💾 NEW BEST MODEL!")
                logger.info("💾 " + "="*60)
                logger.info(f"💾 New best validation loss: {best_val_loss:.6f}")
                logger.info(f"💾 Improvement: {improvement:.6f}")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'train_history': train_history,
                    'val_history': val_history
                }, os.path.join(result_root, 'model_best.pth'))
                
                logger.info(f"💾 Best model saved to: {result_root}/model_best.pth")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0 or epoch == config.training_config['epochs'] - 1:
                checkpoint_path = os.path.join(result_root, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_metrics['total'],
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'train_history': train_history,
                    'val_history': val_history
                }, checkpoint_path)
                logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
            
            # Plot training curves every 20 epochs
            if epoch % 20 == 0 and epoch > 0:
                logger.info("📈 Updating training curves...")
                plot_training_curves(train_history, val_history, result_root)
            
            # Early stopping check
            if early_stopping(val_metrics['total'], model):
                logger.info("🛑 " + "="*60)
                logger.info("🛑 EARLY STOPPING TRIGGERED")
                logger.info("🛑 " + "="*60)
                logger.info(f"🛑 Stopped at epoch {epoch}")
                logger.info(f"🛑 No improvement for {early_stopping.patience} epochs")
                logger.info(f"🛑 Best validation loss: {best_val_loss:.6f}")
                break
                
    except KeyboardInterrupt:
        logger.info("⚠️ " + "="*60)
        logger.info("⚠️ TRAINING INTERRUPTED BY USER")
        logger.info("⚠️ " + "="*60)
        logger.info("⚠️ Saving current progress...")
        
        # Save interrupted state
        interrupted_checkpoint = os.path.join(result_root, 'interrupted_checkpoint.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_loss': val_metrics['total'] if 'val_metrics' in locals() else float('inf'),
            'best_val_loss': best_val_loss,
            'config': config,
            'train_history': train_history,
            'val_history': val_history,
            'interrupted': True
        }, interrupted_checkpoint)
        logger.info(f"⚠️ Progress saved to: {interrupted_checkpoint}")
    
    # Final results
    total_time = time.time() - start_time
    epochs_completed = len(train_history['total'])
    
    logger.info("🎉 " + "="*80)
    logger.info("🎉 TRAINING SESSION COMPLETED")
    logger.info("🎉 " + "="*80)
    logger.info(f"🎉 Total training time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    logger.info(f"🎉 Epochs completed: {epochs_completed}")
    logger.info(f"🎉 Average time per epoch: {total_time/epochs_completed:.2f}s")
    logger.info(f"🎉 Best validation loss: {best_val_loss:.6f}")
    
    if epochs_completed > 1:
        total_improvement = train_history['total'][0] - train_history['total'][-1]
        logger.info(f"🎉 Total training loss improvement: {total_improvement:.6f}")
        
        val_improvement = val_history['total'][0] - best_val_loss
        logger.info(f"🎉 Total validation loss improvement: {val_improvement:.6f}")
    
    # Final plots and save
    logger.info("📊 Generating final training curves...")
    plot_training_curves(train_history, val_history, result_root)
    
    # Save comprehensive training history
    logger.info("💾 Saving training history...")
    history_data = {
        'train_history': train_history,
        'val_history': val_history,
        'gradient_norms': gradient_norms,
        'total_time': total_time,
        'epochs_completed': epochs_completed,
        'best_val_loss': best_val_loss,
        'config': {k: str(v) for k, v in config.model_config.items()},
        'training_config': {k: str(v) for k, v in config.training_config.items()}
    }
    
    savemat(os.path.join(result_root, 'training_history.mat'), history_data)
    
    # Log file locations
    logger.info("📁 " + "="*60)
    logger.info("📁 OUTPUT FILES")
    logger.info("📁 " + "="*60)
    logger.info(f"📁 Best model: {result_root}/model_best.pth")
    logger.info(f"📁 Training history: {result_root}/training_history.mat")
    logger.info(f"📁 Training curves: {result_root}/training_curves.png")
    logger.info(f"📁 Log file: {result_root}/training_{args.model_id}.log")
    
    # Final recommendations
    logger.info("🔍 " + "="*60)
    logger.info("🔍 NEXT STEPS")
    logger.info("🔍 " + "="*60)
    logger.info(f"🔍 Evaluate your model:")
    logger.info(f"🔍   python eval_transformer_sim.py --model_id {args.model_id}")
    logger.info(f"🔍 Test on real data:")
    logger.info(f"🔍   python eval_transformer_real.py --model_id {args.model_id}")
    logger.info(f"🔍 Resume training:")
    logger.info(f"🔍   python train_optimized.py --resume {result_root}/model_best.pth")
    
    logger.info("✅ Training session completed successfully!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
