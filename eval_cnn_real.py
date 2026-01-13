#!/usr/bin/env python3
"""
Evaluation script for CNN Spatial Filter on real data
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
    parser = argparse.ArgumentParser(description='CNN Spatial Filter Real Data Evaluation')
    parser.add_argument('--device', default='cpu', type=str, help='Device running the code')
    parser.add_argument('--model_id', type=str, default='cnn_spatial', help='Model identifier')
    parser.add_argument('--resume', default='', type=str, help='Epoch to resume from')
    parser.add_argument('--data_dir', default='real_data', type=str, help='Real data directory')
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
    print("LOADING CNN SPATIAL FILTER MODEL FOR REAL DATA")
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
    print("EVALUATING ON REAL DATA")
    print("="*80)
    
    net.eval()
    
    # Check for different subject directories
    subject_list = ['real_data', 'VEP', 'test_data']
    
    for subject in subject_list:
        folder_name = args.data_dir if subject == 'real_data' else f'source/{subject}'
        
        subject_start_time = time.time()
        
        # Look for .mat files
        flist = glob.glob(os.path.join(folder_name, 'sample_*.mat'))
        if len(flist) == 0:
            flist = glob.glob(os.path.join(folder_name, 'data*.mat'))
        
        if len(flist) == 0:
            print(f'WARNING: No files in {folder_name}')
            continue
        
        flist = sorted(flist)
        print(f"\nFound {len(flist)} files in {folder_name}")
        
        test_data = []
        
        for filepath in flist:
            try:
                data_mat = loadmat(filepath)
                
                # Handle different data formats
                if 'eeg_data' in data_mat:
                    data = data_mat['eeg_data']
                elif 'data' in data_mat:
                    data = data_mat['data']
                else:
                    print(f"WARNING: Unknown format in {filepath}")
                    continue
                
                # Normalize
                data = data / (np.max(np.abs(data[:])) + 1e-10)
                test_data.append(data)
                
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
        
        if len(test_data) == 0:
            print(f'WARNING: No valid data in {folder_name}')
            continue
        
        print(f"Loaded {len(test_data)} samples from {folder_name}")
        
        # Convert to tensor and run inference
        data = torch.from_numpy(np.array(test_data)).to(device, torch.float)
        print(f"Input shape: {data.shape}")
        
        with torch.no_grad():
            out = net(data)['last']
        
        all_out = out.detach().cpu().numpy()
        print(f"Output shape: {all_out.shape}")
        
        # Save results
        output_filename = os.path.join(folder_name, f'cnn_real_{args.model_id}.mat')
        savemat(output_filename, {
            'predictions': all_out,
            'model_info': {
                'model_id': args.model_id,
                'checkpoint': os.path.basename(fn),
                'num_parameters': net.count_parameters(),
                'device': str(device),
                'model_type': 'cnn_spatial',
                'subject': subject
            },
            'num_samples': len(test_data)
        })
        
        print(f'Results saved: {output_filename}')
        print(f'Processing time: {time.time() - subject_start_time:.2f}s')
    
    # Summary
    print("\n" + "="*80)
    print("REAL DATA EVALUATION COMPLETED")
    print("="*80)
    print(f"Total time: {time.time() - start_time:.2f}s")


if __name__ == '__main__':
    main()
