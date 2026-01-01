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
    # Separate the valid_labels which have variable lengths
    valid_labels = [item['valid_labels'] for item in batch]
    
    # Remove valid_labels from items for default collation
    batch_no_valid_labels = []
    for item in batch:
        new_item = {k: v for k, v in item.items() if k != 'valid_labels'}
        batch_no_valid_labels.append(new_item)
    
    # Use default collation for the rest
    collated = torch.utils.data.default_collate(batch_no_valid_labels)
    
    # Add back valid_labels as a list
    collated['valid_labels'] = valid_labels
    
    return collated


def main():
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Transformer Real Data Evaluation')
    parser.add_argument('--device', default='cpu', type=str, help='device running the code')
    parser.add_argument('--model_id', type=str, default='1', help='model id')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--data_dir', default='real_data', type=str, help='real data directory')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    result_root = 'model_result/{}_optimized_transformer'.format(args.model_id)
    if not os.path.exists(result_root):
        print("ERROR: No model {}".format(args.model_id))
        return

    # =============================== LOAD MODEL =====================================================================================================
    if args.resume:
        fn = os.path.join(result_root, 'epoch_' + args.resume)
    else:
        # Try different checkpoint names in order of preference
        checkpoint_names = [ 'model_best.pth']
        fn = None
        for name in checkpoint_names:
            potential_path = os.path.join(result_root, name)
            if os.path.exists(potential_path):
                fn = potential_path
                break
        
        if fn is None:
            print("ERROR: no checkpoint found in {}".format(result_root))
            print("Available files:", os.listdir(result_root) if os.path.exists(result_root) else "Directory not found")
            return
    
    print("=> Load checkpoint", fn)
    if os.path.isfile(fn):
        print("=> Found checkpoint '{}'".format(fn))
        checkpoint = torch.load(fn, map_location=torch.device('cpu'), weights_only=False)
        
        # Handle different checkpoint formats
        if 'best_result' in checkpoint:
            # Old format
            best_result = checkpoint['best_result']
            arch = checkpoint.get('arch', 'TransformerTemporalInverseNet')
            net = network.__dict__[arch](*checkpoint['attribute_list']).to(device)
            net.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # New format from train_optimized.py
            print("=> Failed to load best result")

            best_result = checkpoint.get('best_val_loss', 'N/A')
            config = checkpoint.get('config')
            
            # Create model from config
            if config:
                print("=> Creating model from config")
                net = network.TransformerTemporalInverseNet(
                    num_sensor=config.model_config['num_sensor'],
                    num_source=config.model_config['num_source'],
                    transformer_layers=config.model_config['transformer_layers'],
                    d_model=config.model_config['d_model'],
                    nhead=config.model_config['nhead'],
                    dropout=config.model_config['dropout'],
                    spatial_activation=config.model_config['spatial_activation'],
                    temporal_activation=config.model_config['temporal_activation'],
                    temporal_input_size=config.model_config['temporal_input_size']
                ).to(device)
            else:
                # Default architecture if config not available
                print("=> Creating default model")
                net = network.TransformerTemporalInverseNet().to(device)
            
            # Load model weights
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        print("=> Loaded checkpoint {}, best validation loss: {}".format(fn, best_result))
    else:
        print("ERROR: no checkpoint found")
        return

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # =============================== EVALUATION =====================================================================================================
    net.eval()
    
    # Check for different subject directories
    subject_list = ['VEP', 'real_data', 'test_data']  # Add more as needed
    
    for pii in subject_list:
        folder_name = args.data_dir if pii == 'real_data' else f'source/{pii}'
        start_time = time.time()
        
        # Look for .mat files (both sample_*.mat and data*.mat formats)
        flist = glob.glob(os.path.join(folder_name, 'sample_*.mat'))
        if len(flist) == 0:
            flist = glob.glob(os.path.join(folder_name, 'data*.mat'))
        
        if len(flist) == 0:
            print('WARNING: NO FILE IN FOLDER {}.'.format(folder_name))
            continue
            
        flist = sorted(flist)
        print(f"Found {len(flist)} files in {folder_name}")
        
        test_data = []
        for i in flist:
            try:
                data_mat = loadmat(i)
                
                # Handle different data formats
                if 'eeg_data' in data_mat:
                    # Labeled dataset format
                    data = data_mat['eeg_data']
                elif 'data' in data_mat:
                    # Original format
                    data = data_mat['data']
                else:
                    print(f"WARNING: Unknown data format in {i}")
                    continue
                    
                # Normalize data (same as original eval_real.py)
                data = data / np.max(np.abs(data[:]))
                test_data.append(data)
                
            except Exception as e:
                print(f"Error loading {i}: {e}")
                continue

        if len(test_data) == 0:
            print(f"WARNING: No valid data files in {folder_name}")
            continue
            
        # Convert to tensor and run inference
        data = torch.from_numpy(np.array(test_data)).to(device, torch.float)
        print(f"Input data shape: {data.shape}")
        
        with torch.no_grad():
            out = net(data)['last']
            
        # Calculate the output
        all_out = out.detach().cpu().numpy()
        print(f"Output shape: {all_out.shape}")
        
        # Save results
        output_filename = os.path.join(folder_name, f'transformer_test_{args.model_id}_{os.path.basename(fn)[-8:]}.mat')
        savemat(output_filename, {
            'all_out': all_out,
            'model_info': {
                'arch': checkpoint.get('arch', 'TransformerTemporalInverseNet'),
                'model_id': args.model_id,
                'checkpoint': fn,
                'num_parameters': net.count_parameters()
            }
        })
        print('Save output as:', output_filename)
        print(f'Processing {pii} took: {time.time() - start_time:.2f} seconds')
        
    print('Total run time:', time.time() - start_time)


if __name__ == '__main__':
    main()
