#!/usr/bin/env python3
"""
Evaluation script for CNN Spatial Filter on simulated data
"""

import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import glob

import torch
import network


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


def main():
    start_time = time.time()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='CNN Spatial Filter Evaluation on Simulated Data')
    parser.add_argument('--device', default='cpu', type=str, help='Device running the code')
    parser.add_argument('--model_id', type=str, default='cnn_spatial', help='Model identifier')
    parser.add_argument('--resume', default='', type=str, help='Epoch to resume from')
    parser.add_argument('--data_dir', default='labeled_dataset', type=str, help='Test data directory')
    parser.add_argument('--info', default='', type=str, help='Additional information')
    
    args = parser.parse_args()
    
    # Prepare
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    result_root = f'model_result/{args.model_id}_cnn_spatial'
    
    if not os.path.exists(result_root):
        print(f"ERROR: No model {args.model_id}")
        return
    
    # Load model
    print("="*80)
    print("LOADING CNN SPATIAL FILTER MODEL")
    print("="*80)
    
    if args.resume:
        fn = os.path.join(result_root, 'epoch_' + args.resume)
    else:
        # Try to find best checkpoint
        checkpoint_names = ['model_best.pth', 'checkpoint_epoch_0.pth']
        fn = None
        for name in checkpoint_names:
            potential_path = os.path.join(result_root, name)
            if os.path.exists(potential_path):
                fn = potential_path
                break
        
        if fn is None:
            print(f"ERROR: No checkpoint found in {result_root}")
            print(f"Available files: {os.listdir(result_root) if os.path.exists(result_root) else 'Directory not found'}")
            return
    
    print(f"Loading checkpoint: {fn}")
    
    if os.path.isfile(fn):
        print(f"Found checkpoint: {fn}")
        checkpoint = torch.load(fn, map_location=device, weights_only=False)
        
        # Create model
        config = checkpoint.get('config')
        model_type = checkpoint.get('model_type', 'unknown')
        
        print(f"Model type: {model_type}")
        
        if config:
            print("Creating model from checkpoint config...")
            net = network.TransformerTemporalInverseNet(
                num_sensor=config.model_config['num_sensor'],
                num_source=config.model_config['num_source'],
                transformer_layers=config.model_config['transformer_layers'],
                spatial_model=network.CNN2DSpatialFilter,  # Use CNN spatial filter
                spatial_activation=config.model_config['spatial_activation'],
                temporal_activation=config.model_config['temporal_activation'],
                temporal_input_size=config.model_config['temporal_input_size'],
                d_model=config.model_config['d_model'],
                nhead=config.model_config['nhead'],
                dropout=config.model_config['dropout']
            ).to(device)
        else:
            print("Creating default CNN model...")
            net = network.TransformerTemporalInverseNet(
                spatial_model=network.CNN2DSpatialFilter
            ).to(device)
        
        # Load weights
        net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        best_result = checkpoint.get('best_val_loss', 'N/A')
        print(f"Best validation loss: {best_result}")
    else:
        print(f"ERROR: Checkpoint not found: {fn}")
        return
    
    print(f"Model parameters: {net.count_parameters():,}")
    print(f"Prepare time: {time.time() - start_time:.2f}s")
    
    # Evaluation
    print("\n" + "="*80)
    print("EVALUATING ON SIMULATED DATA")
    print("="*80)
    
    net.eval()
    
    # Load test data
    start_time = time.time()
    
    flist = glob.glob(os.path.join(args.data_dir, 'sample_*.mat'))
    
    if len(flist) == 0:
        print(f"ERROR: No sample_*.mat files in {args.data_dir}")
        return
    
    flist = sorted(flist)
    print(f"Found {len(flist)} test samples")
    
    # Use subset for evaluation
    test_samples = flist[-100:] if len(flist) > 100 else flist
    print(f"Evaluating on {len(test_samples)} samples")
    
    test_data = []
    test_targets = []
    
    for filepath in test_samples:
        try:
            data_mat = loadmat(filepath)
            
            # Load EEG data
            if 'eeg_data' in data_mat:
                data = data_mat['eeg_data']
            elif 'data' in data_mat:
                data = data_mat['data']
            else:
                print(f"WARNING: Unknown format in {filepath}")
                continue
            
            # Load target source
            if 'source_activity' in data_mat:
                target = data_mat['source_activity']
            elif 'nmm' in data_mat:
                target = data_mat['nmm']
            else:
                target = None
            
            # Normalize
            data = data / (np.max(np.abs(data[:])) + 1e-10)
            if target is not None:
                target = target / (np.max(np.abs(target[:])) + 1e-10)
            
            test_data.append(data)
            if target is not None:
                test_targets.append(target)
            
        except Exception as e:
            print(f"Error loading {filepath}: {e}")
            continue
    
    if len(test_data) == 0:
        print(f"ERROR: No valid test data loaded")
        return
    
    print(f"Loaded {len(test_data)} test samples")
    
    # Convert to tensors and run inference
    data = torch.from_numpy(np.array(test_data)).to(device, torch.float)
    print(f"Input shape: {data.shape}")
    
    with torch.no_grad():
        out = net(data)['last']
    
    all_out = out.detach().cpu().numpy()
    print(f"Output shape: {all_out.shape}")
    print(f"Inference time: {time.time() - start_time:.2f}s")
    
    # Calculate metrics if we have targets
    if len(test_targets) > 0:
        test_targets = np.array(test_targets)
        print(f"\nTarget shape: {test_targets.shape}")
        
        # Calculate MSE
        mse = np.mean((all_out - test_targets) ** 2)
        print(f"MSE: {mse:.6f}")
        
        # Calculate correlation
        all_out_flat = all_out.reshape(-1)
        test_targets_flat = test_targets.reshape(-1)
        correlation = np.corrcoef(all_out_flat, test_targets_flat)[0, 1]
        print(f"Correlation: {correlation:.6f}")
    
    # Save results
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    output_filename = os.path.join(result_root, f'cnn_sim_results.mat')
    
    save_data = {
        'predictions': all_out,
        'model_info': {
            'model_id': args.model_id,
            'checkpoint': os.path.basename(fn),
            'num_parameters': net.count_parameters(),
            'device': str(device),
            'model_type': 'cnn_spatial'
        },
        'num_samples': len(test_data)
    }
    
    if len(test_targets) > 0:
        save_data['targets'] = test_targets
        save_data['metrics'] = {
            'mse': mse,
            'correlation': correlation
        }
    
    savemat(output_filename, save_data)
    print(f"Results saved: {output_filename}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED")
    print("="*80)
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    main()
