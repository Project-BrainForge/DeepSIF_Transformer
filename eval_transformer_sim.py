import argparse
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime
import collections

import torch
from torch.utils.data import DataLoader

import network
import loaders
from utils import get_otsu_regions, ispadding


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
    parser = argparse.ArgumentParser(description='DeepSIF Transformer Simulation Evaluation')
    parser.add_argument('--workers', default=0, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:0', type=str, help='device running the code')
    parser.add_argument('--dat', default='LabeledDatasetLoader', type=str, help='data loader')
    parser.add_argument('--test', default='labeled_dataset', type=str, help='test dataset directory')
    parser.add_argument('--model_id', type=str, default='1', help='model id')
    parser.add_argument('--resume', default='', type=str, help='epoch id to resume')
    parser.add_argument('--fwd', default='', type=str, help='forward matrix to use (optional for pre-processed data)')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    parser.add_argument('--num_samples', default=100, type=int, help='number of samples to evaluate')
    
    args = parser.parse_args()

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")

    # Load distance matrix if available
    dis_matrix = None
    if os.path.exists('anatomy/dis_matrix_fs_20k.mat'):
        dis_matrix = loadmat('anatomy/dis_matrix_fs_20k.mat')['raw_dis_matrix']
        print("Loaded distance matrix for localization error calculation")
    else:
        print("Warning: Distance matrix not found, localization error will not be calculated")

    result_root = 'model_result/{}_transformer_model'.format(args.model_id)
    if not os.path.exists(result_root):
        print("ERROR: No model {}".format(args.model_id))
        return
    
    # Forward matrix (optional for pre-processed data)
    fwd = None
    if args.fwd and os.path.exists(f'anatomy/{args.fwd}'):
        fwd = loadmat(f'anatomy/{args.fwd}')['fwd']
        print(f"Loaded forward matrix: {fwd.shape}")

    # ================================== LOAD DATA ===================================================================================================
    test_data = loaders.__dict__[args.dat](
        args.test, 
        fwd=fwd,
        args_params={'dataset_len': args.num_samples}
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        num_workers=args.workers, 
        pin_memory=True, 
        shuffle=False,
        collate_fn=custom_collate_fn
    )

    # =============================== LOAD MODEL =====================================================================================================
    if args.resume:
        fn = os.path.join(result_root, 'epoch_' + args.resume)
    else:
        # Try different checkpoint names in order of preference
        checkpoint_names = ['model_best.pth', 'model_best.pth.tar', 'checkpoint_epoch_0.pth']
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
        checkpoint = torch.load(fn, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if 'best_result' in checkpoint:
            # Old format
            best_result = checkpoint['best_result']
            arch = checkpoint.get('arch', 'TransformerTemporalInverseNet')
            net = network.__dict__[arch](*checkpoint['attribute_list']).to(device)
            net.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            # New format from train_optimized.py
            best_result = checkpoint.get('best_val_loss', 'N/A')
            config = checkpoint.get('config')
            
            # Create model from config
            if config:
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
                net = network.TransformerTemporalInverseNet().to(device)
            
            # Load model weights
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        
        print("=> Loaded checkpoint {}, best validation loss: {}".format(fn, best_result))

        # Define logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        log_file = os.path.join(result_root, f'eval_outputs_{arch}.log')
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        logger.info("=================== Evaluation mode: {} ====================================".format(datetime.datetime.now()))
        logger.info("Testing data is {}".format(args.test))
        # Save every parameters in args
        for v in args.__dict__:
            if v not in ['workers']:
                logger.info('{} is {}'.format(v, args.__dict__[v]))
    else:
        print("ERROR: no checkpoint found")
        return

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # =============================== EVALUATION =====================================================================================================
    net.eval()

    eval_dict = collections.defaultdict(list)
    eval_dict['all_out'] = []                    # DeepSIF output
    eval_dict['all_nmm'] = []                    # Ground truth source activity
    eval_dict['all_regions'] = []                # DeepSIF identified source regions
    eval_dict['all_loss'] = 0                    # MSE Loss
    eval_dict['precision'] = []                  # Precision scores
    eval_dict['recall'] = []                     # Recall scores  
    eval_dict['le'] = []                         # Localization errors
    eval_dict['correlations'] = []               # Temporal correlations
    
    criterion = torch.nn.MSELoss(reduction='sum')

    total_samples = 0
    with torch.no_grad():
        for batch_idx, sample_batch in enumerate(test_loader):
            
            data = sample_batch['data'].to(device, torch.float)
            nmm = sample_batch['nmm'].numpy()
            labels = sample_batch['label'].numpy()  # Padded labels
            valid_labels = sample_batch['valid_labels']  # Unpadded labels (list)
            
            model_output = net(data)
            out = model_output['last']
            
            # Calculate loss function
            nmm_torch = sample_batch['nmm'].to(device, torch.float)
            eval_dict['all_loss'] = eval_dict['all_loss'] + criterion(out, nmm_torch).item()
            
            # Convert to numpy for analysis
            out_numpy = out.cpu().numpy()
            
            # ----- OTSU THRESHOLDING FOR REGION IDENTIFICATION --------------------------------------------------
            # Convert valid_labels to the format expected by get_otsu_regions
            batch_labels = []
            for i, vlabels in enumerate(valid_labels):
                # Create padded label array for this sample
                padded_label = np.full(70, 15213)  # Use same padding as original
                if len(vlabels) > 0:
                    padded_label[:len(vlabels)] = vlabels
                batch_labels.append(padded_label)
            
            batch_labels = np.array(batch_labels)
            
            # Calculate metrics
            if dis_matrix is not None:
                eval_results = get_otsu_regions(out_numpy, batch_labels, 
                                              args_params={'dis_matrix': dis_matrix})
                eval_dict['precision'].extend(eval_results['precision'])
                eval_dict['recall'].extend(eval_results['recall'])
                eval_dict['le'].extend(eval_results['le'])
            else:
                eval_results = get_otsu_regions(out_numpy, batch_labels)
                if 'precision' in eval_results:
                    eval_dict['precision'].extend(eval_results['precision'])
                if 'recall' in eval_results:
                    eval_dict['recall'].extend(eval_results['recall'])

            eval_dict['all_regions'].extend(eval_results['all_regions'])
            eval_dict['all_out'].extend(eval_results['all_out'])
            
            # ------------------------------------------------------------------------------------
            # Save ground truth for center regions
            for kk in range(out.size(0)):
                vlabels = valid_labels[kk]
                if len(vlabels) > 0:
                    # Save activity in the center region (first active region)
                    center_region = vlabels[0]
                    if center_region < nmm.shape[2]:
                        eval_dict['all_nmm'].append(nmm[kk, :, center_region])
                    else:
                        # If center region is out of bounds, save zeros
                        eval_dict['all_nmm'].append(np.zeros(nmm.shape[1]))
                else:
                    # No active regions
                    eval_dict['all_nmm'].append(np.zeros(nmm.shape[1]))
                    
            total_samples += out.size(0)
            
            # Calculate temporal correlations for this batch
            for kk in range(out.size(0)):
                pred_sig = out_numpy[kk]  # (time, sources)
                target_sig = nmm[kk]      # (time, sources)
                
                # Calculate correlation for active sources only
                vlabels = valid_labels[kk]
                if len(vlabels) > 0:
                    correlations = []
                    for region in vlabels:
                        if region < pred_sig.shape[1] and region < target_sig.shape[1]:
                            pred_region = pred_sig[:, region]
                            target_region = target_sig[:, region]
                            
                            if np.std(pred_region) > 1e-8 and np.std(target_region) > 1e-8:
                                corr = np.corrcoef(pred_region, target_region)[0, 1]
                                if not np.isnan(corr):
                                    correlations.append(corr)
                    
                    if correlations:
                        eval_dict['correlations'].append(np.mean(correlations))
                    else:
                        eval_dict['correlations'].append(0.0)
                else:
                    eval_dict['correlations'].append(0.0)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)}")
                logger.info(f"Processed batch {batch_idx}/{len(test_loader)}")

    # =============================== SAVE RESULTS ===============================================================================================
    
    # Calculate summary statistics
    eval_dict['summary'] = {
        'total_samples': total_samples,
        'avg_loss': eval_dict['all_loss'] / total_samples if total_samples > 0 else 0,
        'mean_precision': np.mean(eval_dict['precision']) if eval_dict['precision'] else 0,
        'std_precision': np.std(eval_dict['precision']) if eval_dict['precision'] else 0,
        'mean_recall': np.mean(eval_dict['recall']) if eval_dict['recall'] else 0,
        'std_recall': np.std(eval_dict['recall']) if eval_dict['recall'] else 0,
        'mean_correlation': np.mean(eval_dict['correlations']) if eval_dict['correlations'] else 0,
        'std_correlation': np.std(eval_dict['correlations']) if eval_dict['correlations'] else 0
    }
    
    if eval_dict['le']:
        eval_dict['summary']['mean_le'] = np.mean(eval_dict['le'])
        eval_dict['summary']['std_le'] = np.std(eval_dict['le'])
    
    # Print summary
    print("\n=== EVALUATION SUMMARY ===")
    print(f"Total samples: {eval_dict['summary']['total_samples']}")
    print(f"Average Loss: {eval_dict['summary']['avg_loss']:.6f}")
    print(f"Mean Precision: {eval_dict['summary']['mean_precision']:.4f} ± {eval_dict['summary']['std_precision']:.4f}")
    print(f"Mean Recall: {eval_dict['summary']['mean_recall']:.4f} ± {eval_dict['summary']['std_recall']:.4f}")
    print(f"Mean Correlation: {eval_dict['summary']['mean_correlation']:.4f} ± {eval_dict['summary']['std_correlation']:.4f}")
    if 'mean_le' in eval_dict['summary']:
        print(f"Mean Localization Error: {eval_dict['summary']['mean_le']:.4f} ± {eval_dict['summary']['std_le']:.4f}")
    
    # Log summary
    logger.info("=== EVALUATION SUMMARY ===")
    for key, value in eval_dict['summary'].items():
        logger.info(f"{key}: {value}")
    
    # Save results
    output_file = fn + '_preds_{}_{}.mat'.format(os.path.basename(args.test), args.info)
    savemat(output_file, eval_dict)
    print(f"\nResults saved to: {output_file}")
    logger.info(f"Results saved to: {output_file}")


if __name__ == '__main__':
    main()
