#!/usr/bin/env python3
"""
Training script for DeepSIF with 2D CNN Spatial Filter using electrode topology
"""

import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import logging
import sys

import network
import loaders
from config_optimized import (
    OptimizedConfig, create_optimized_optimizer,
    create_lr_scheduler, create_optimized_loss_function, EarlyStopping,
    apply_gradient_clipping, add_noise_augmentation, add_temporal_shift
)


def custom_collate_fn(batch):
    """Custom collate function to handle variable-length valid_labels and grid conversion"""
    valid_labels = [item['valid_labels'] for item in batch]
    
    batch_no_valid_labels = []
    for item in batch:
        new_item = {k: v for k, v in item.items() if k != 'valid_labels'}
        batch_no_valid_labels.append(new_item)
    
    collated = torch.utils.data.default_collate(batch_no_valid_labels)
    collated['valid_labels'] = valid_labels
    
    return collated


def create_cnn_spatial_model(config):
    """Create model with CNN2DSpatialFilter spatial component"""
    model = network.TransformerTemporalInverseNet(
        num_sensor=config.model_config['num_sensor'],
        num_source=config.model_config['num_source'],
        transformer_layers=config.model_config['transformer_layers'],
        spatial_model=network.CNN2DSpatialFilter,  # Use CNN instead of MLP
        spatial_activation=config.model_config['spatial_activation'],
        temporal_activation=config.model_config['temporal_activation'],
        temporal_input_size=config.model_config['temporal_input_size'],
        d_model=config.model_config['d_model'],
        nhead=config.model_config['nhead'],
        dropout=config.model_config['dropout']
    )
    return model, config


class SafeFormatter(logging.Formatter):
    """Formatter that handles Unicode safely"""
    
    def format(self, record):
        message = super().format(record)
        # Replace emojis with ASCII alternatives
        emoji_replacements = {
            '🚀': '[START]', '🔄': '[EPOCH]', '📊': '[INFO]',
            '✅': '[OK]', '📈': '[UP]', '📉': '[DOWN]',
            '💾': '[SAVE]', '⚠️': '[WARN]', '🎉': '[DONE]',
            '🛑': '[STOP]', '🔍': '[NEXT]', '📁': '[FILE]'
        }
        for emoji, replacement in emoji_replacements.items():
            message = message.replace(emoji, replacement)
        return message


def setup_logging(result_root, model_name):
    """Setup comprehensive logging"""
    os.makedirs(result_root, exist_ok=True)
    
    logger = logging.getLogger('DeepSIF_CNN')
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # File handler
    log_file = os.path.join(result_root, f'training_cnn_{model_name}.log')
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    
    formatter = SafeFormatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info("="*80)
    logger.info("CNN SPATIAL FILTER TRAINING SESSION")
    logger.info("="*80)
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    
    return logger


def train_epoch(model, train_loader, optimizer, loss_function, config, device, logger, epoch):
    """Training epoch"""
    epoch_start_time = time.time()
    model.train()
    
    epoch_losses = []
    epoch_metrics = {'reconstruction': [], 'sparsity': [], 'smoothness': [], 'total': []}
    gradient_norms = []
    
    logger.info(f"[EPOCH {epoch}] Starting training on {len(train_loader)} batches")
    
    for batch_idx, batch in enumerate(train_loader):
        data = batch['data'].to(device)
        target = batch['nmm'].to(device)
        
        # Data augmentation
        if model.training:
            data = add_noise_augmentation(data, config)
            data = add_temporal_shift(data, config)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(data)
        loss, loss_components = loss_function(output['last'], target)
        
        # Backward pass
        loss.backward()
        grad_norm = apply_gradient_clipping(model, config.training_config['gradient_clip'])
        optimizer.step()
        
        # Track metrics
        epoch_losses.append(loss.item())
        for key, value in loss_components.items():
            epoch_metrics[key].append(value)
        if grad_norm is not None:
            gradient_norms.append(grad_norm)
        
        # Logging
        if batch_idx % 25 == 0:
            grad_norm_val = grad_norm if grad_norm else 0.0
            logger.info(
                f"[Batch {batch_idx:4d}/{len(train_loader)}] Loss: {loss.item():.6f} | "
                f"GradNorm: {grad_norm_val:.6f}"
            )
    
    # Epoch summary
    avg_metrics = {key: np.mean(values) for key, values in epoch_metrics.items()}
    epoch_time = time.time() - epoch_start_time
    
    logger.info("-"*60)
    logger.info(f"[EPOCH {epoch} SUMMARY] Time: {epoch_time:.2f}s")
    logger.info(f"  Total Loss: {avg_metrics['total']:.6f}")
    logger.info(f"  Reconstruction: {avg_metrics['reconstruction']:.6f}")
    logger.info(f"  Sparsity: {avg_metrics['sparsity']:.6f}")
    logger.info(f"  Smoothness: {avg_metrics['smoothness']:.6f}")
    
    return avg_metrics, np.mean(gradient_norms) if gradient_norms else 0.0


def validate_epoch(model, val_loader, loss_function, device, logger, epoch):
    """Validation epoch"""
    val_start_time = time.time()
    model.eval()
    
    val_losses = []
    val_metrics = {'reconstruction': [], 'sparsity': [], 'smoothness': [], 'total': []}
    
    logger.info(f"[EPOCH {epoch}] Starting validation on {len(val_loader)} batches")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            data = batch['data'].to(device)
            target = batch['nmm'].to(device)
            
            output = model(data)
            loss, loss_components = loss_function(output['last'], target)
            
            val_losses.append(loss.item())
            for key, value in loss_components.items():
                val_metrics[key].append(value)
            
            if batch_idx % 20 == 0:
                logger.info(f"[Val Batch {batch_idx:3d}/{len(val_loader)}] Loss: {loss.item():.6f}")
    
    # Validation summary
    avg_metrics = {key: np.mean(values) for key, values in val_metrics.items()}
    val_time = time.time() - val_start_time
    
    logger.info("-"*60)
    logger.info(f"[VALIDATION SUMMARY] Time: {val_time:.2f}s")
    logger.info(f"  Total Loss: {avg_metrics['total']:.6f}")
    logger.info(f"  Reconstruction: {avg_metrics['reconstruction']:.6f}")
    logger.info(f"  Sparsity: {avg_metrics['sparsity']:.6f}")
    logger.info(f"  Smoothness: {avg_metrics['smoothness']:.6f}")
    
    return avg_metrics


def main():
    parser = argparse.ArgumentParser(description='CNN Spatial Filter Training')
    parser.add_argument('--data_path', default='labeled_dataset', type=str, help='Path to labeled dataset')
    parser.add_argument('--model_id', default='cnn_spatial', type=str, help='Model identifier')
    parser.add_argument('--device', default='cuda:0', type=str, help='Device to use')
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true', help='Debug mode with smaller dataset')
    parser.add_argument('--epochs', default=100, type=int, help='Number of training epochs (default: 100)')
    parser.add_argument('--batch_size', default=8, type=int, help='Batch size (default: 8)')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate (default: 1e-4)')
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    config = OptimizedConfig()
    
    # Override config with command-line arguments if provided
    if args.epochs != 100:
        config.training_config['epochs'] = args.epochs
    if args.batch_size != 8:
        config.training_config['batch_size'] = args.batch_size
    if args.lr != 1e-4:
        config.training_config['learning_rate'] = args.lr
    
    result_root = f'model_result/{args.model_id}_cnn_spatial'
    logger = setup_logging(result_root, args.model_id)
    
    logger.info(f"Device: {device}")
    logger.info(f"Configuration: {config.model_config}")
    
    # Load data
    logger.info("Loading dataset...")
    import glob
    all_files = glob.glob(os.path.join(args.data_path, "sample_*.mat"))
    all_files.sort()
    
    logger.info(f"Found {len(all_files)} files in {args.data_path}")
    
    if len(all_files) == 0:
        logger.error(f"No sample_*.mat files found in {args.data_path}")
        return
    
    if args.debug:
        all_files = all_files[:100]
        logger.warning("DEBUG MODE: Using 100 samples")
    
    # Split data
    n_total = len(all_files)
    n_train = int(n_total * config.data_config['train_split'])
    n_val = int(n_total * config.data_config['val_split'])
    
    train_files = all_files[:n_train]
    val_files = all_files[n_train:n_train + n_val]
    
    logger.info(f"Dataset split: Train={len(train_files)}, Val={len(val_files)}, Test={n_total-n_train-n_val}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    
    train_dataset = loaders.LabeledDatasetLoader(
        args.data_path, fwd=None, args_params={'dataset_len': len(train_files)}
    )
    train_dataset.file_list = train_files
    
    val_dataset = loaders.LabeledDatasetLoader(
        args.data_path, fwd=None, args_params={'dataset_len': len(val_files)}
    )
    val_dataset.file_list = val_files
    
    # Test data loading
    logger.info("Testing data loading...")
    try:
        sample = train_dataset[0]
        logger.info(f"EEG shape: {sample['data'].shape}, NMM shape: {sample['nmm'].shape}")
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        return
    
    train_loader = DataLoader(
        train_dataset, batch_size=config.training_config['batch_size'],
        shuffle=True, collate_fn=custom_collate_fn, num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=config.training_config['batch_size'],
        shuffle=False, collate_fn=custom_collate_fn, num_workers=0
    )
    
    logger.info(f"Batch size: {config.training_config['batch_size']}")
    logger.info(f"Training batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    # Create model with CNN spatial filter
    logger.info("Creating CNN spatial filter model...")
    model, _ = create_cnn_spatial_model(config)
    model = model.to(device)
    
    logger.info(f"Model parameters: {model.count_parameters():,}")
    
    # Setup training components
    logger.info("Setting up optimizer and scheduler...")
    optimizer = create_optimized_optimizer(model, config)
    scheduler = create_lr_scheduler(optimizer, config, len(train_loader))
    loss_function = create_optimized_loss_function(config)
    early_stopping = EarlyStopping(patience=config.training_config['patience'])
    
    # Training history
    train_history = {'total': [], 'reconstruction': [], 'sparsity': [], 'smoothness': []}
    val_history = {'total': [], 'reconstruction': [], 'sparsity': [], 'smoothness': []}
    
    start_epoch = 0
    best_val_loss = float('inf')
    
    # Resume if specified
    if args.resume and os.path.exists(args.resume):
        logger.info(f"Resuming from: {args.resume}")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint.get('best_val_loss', float('inf'))
            logger.info(f"Resumed at epoch {start_epoch}")
        except Exception as e:
            logger.error(f"Failed to resume: {e}")
            logger.info("Starting fresh...")
    
    # Training loop
    logger.info("="*80)
    logger.info(f"STARTING TRAINING: {config.training_config['epochs']} epochs")
    logger.info("="*80)
    
    start_time = time.time()
    
    try:
        for epoch in range(start_epoch, config.training_config['epochs']):
            logger.info("\n" + "="*80)
            logger.info(f"EPOCH {epoch}/{config.training_config['epochs']-1}")
            logger.info("="*80)
            
            # Training
            train_metrics, grad_norm = train_epoch(
                model, train_loader, optimizer, loss_function, config, device, logger, epoch
            )
            
            # Validation
            val_metrics = validate_epoch(model, val_loader, loss_function, device, logger, epoch)
            
            # Update learning rate
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step()
            new_lr = optimizer.param_groups[0]['lr']
            
            if abs(old_lr - new_lr) > 1e-8:
                logger.info(f"Learning rate: {old_lr:.2e} -> {new_lr:.2e}")
            
            # Record history
            for key in train_history.keys():
                train_history[key].append(train_metrics[key])
                val_history[key].append(val_metrics[key])
            
            # Save best model
            is_best = val_metrics['total'] < best_val_loss
            if is_best:
                best_val_loss = val_metrics['total']
                logger.info(f"[NEW BEST] Validation loss: {best_val_loss:.6f}")
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'train_history': train_history,
                    'val_history': val_history,
                    'model_type': 'cnn_spatial'
                }, os.path.join(result_root, 'model_best.pth'))
            
            # Save checkpoint every epoch
            if epoch % 1 == 0:
                checkpoint_path = os.path.join(result_root, f'checkpoint_epoch_{epoch}.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['total'],
                    'best_val_loss': best_val_loss,
                    'config': config,
                    'model_type': 'cnn_spatial'
                }, checkpoint_path)
            
            # Early stopping
            if early_stopping(val_metrics['total'], model):
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'best_val_loss': best_val_loss,
            'interrupted': True
        }, os.path.join(result_root, 'interrupted_checkpoint.pth'))
    
    # Final summary
    total_time = time.time() - start_time
    epochs_completed = len(train_history['total'])
    
    logger.info("="*80)
    logger.info("TRAINING COMPLETED")
    logger.info("="*80)
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    logger.info(f"Epochs completed: {epochs_completed}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Results saved to: {result_root}")
    logger.info("="*80)
    
    # Save training history
    savemat(os.path.join(result_root, 'training_history.mat'), {
        'train_history': train_history,
        'val_history': val_history,
        'total_time': total_time,
        'epochs_completed': epochs_completed,
        'best_val_loss': best_val_loss,
        'model_type': 'cnn_spatial'
    })


if __name__ == '__main__':
    main()
