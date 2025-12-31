import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

import network
import loaders
from utils import get_otsu_regions, compute_correlation


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
    parser = argparse.ArgumentParser(description='DeepSIF Transformer Evaluation')
    parser.add_argument('--model_path', type=str, required=True, help='path to the trained model')
    parser.add_argument('--data_path', default='labeled_dataset', type=str, help='path to evaluation data')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size for evaluation')
    parser.add_argument('--device', default='cuda:0', type=str, help='device to run evaluation')
    parser.add_argument('--output_dir', default='results', type=str, help='output directory for results')
    parser.add_argument('--fwd', default='', type=str, help='forward matrix file')
    parser.add_argument('--num_samples', default=100, type=int, help='number of samples to evaluate')
    
    args = parser.parse_args()
    
    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    # Create model architecture
    arch = checkpoint.get('arch', 'TransformerTemporalInverseNet')
    attribute_list = checkpoint['attribute_list']
    
    if arch == 'TransformerTemporalInverseNet':
        net = network.__dict__[arch](*attribute_list).to(device)
    else:
        net = network.__dict__[arch](*attribute_list).to(device)
    
    net.load_state_dict(checkpoint['state_dict'])
    net.eval()
    
    print(f"Loaded {arch} with {net.count_parameters()} parameters")
    
    # Load pre-processed data
    eval_data = loaders.LabeledDatasetLoader(
        args.data_path,
        fwd=None,  # Not needed for pre-processed data
        args_params={'dataset_len': args.num_samples}
    )
    eval_loader = DataLoader(
        eval_data, 
        batch_size=args.batch_size, 
        num_workers=0, 
        shuffle=False, 
        collate_fn=custom_collate_fn
    )
    
    print(f"Evaluating on {len(eval_data)} samples")
    
    # Evaluation metrics storage
    all_predictions = []
    all_targets = []
    all_labels = []
    all_correlations = []
    all_mse_losses = []
    
    criterion = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch_idx, sample_batch in enumerate(eval_loader):
            data = sample_batch['data'].to(device)
            target = sample_batch['nmm'].to(device)
            labels = sample_batch['label']  # Padded labels for consistency
            valid_labels = sample_batch['valid_labels']  # Unpadded labels for evaluation
            
            # Forward pass
            model_output = net(data)
            prediction = model_output['last']
            
            # Compute losses
            mse_loss = criterion(prediction, target).mean(dim=(1, 2))
            
            # Move to CPU for analysis
            pred_cpu = prediction.cpu().numpy()
            target_cpu = target.cpu().numpy()
            
            # Store results
            all_predictions.extend(pred_cpu)
            all_targets.extend(target_cpu)
            all_labels.extend(valid_labels)  # Use valid_labels for evaluation
            all_mse_losses.extend(mse_loss.cpu().numpy())
            
            # Compute correlations for this batch
            batch_correlations = []
            for i in range(pred_cpu.shape[0]):
                corr = compute_correlation(pred_cpu[i], target_cpu[i])
                batch_correlations.append(np.mean(corr[corr > 0]))  # Average non-zero correlations
            all_correlations.extend(batch_correlations)
            
            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(eval_loader)}")
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_mse_losses = np.array(all_mse_losses)
    all_correlations = np.array(all_correlations)
    
    # Compute evaluation metrics
    print("\n=== EVALUATION RESULTS ===")
    print(f"Mean MSE Loss: {np.mean(all_mse_losses):.6f} ± {np.std(all_mse_losses):.6f}")
    print(f"Mean Correlation: {np.nanmean(all_correlations):.4f} ± {np.nanstd(all_correlations):.4f}")
    
    # Source localization analysis using Otsu thresholding
    print("\nPerforming source localization analysis...")
    localization_results = get_otsu_regions(all_predictions, np.array(all_labels))
    
    if 'precision' in localization_results:
        precision = localization_results['precision']
        recall = localization_results['recall']
        print(f"Mean Precision: {np.mean(precision):.4f} ± {np.std(precision):.4f}")
        print(f"Mean Recall: {np.mean(recall):.4f} ± {np.std(recall):.4f}")
    
    # Save detailed results
    results = {
        'predictions': all_predictions,
        'targets': all_targets,
        'labels': all_labels,
        'mse_losses': all_mse_losses,
        'correlations': all_correlations,
        'localization_results': localization_results,
        'model_info': {
            'arch': arch,
            'model_path': args.model_path,
            'num_parameters': net.count_parameters()
        }
    }
    
    # Save results
    results_file = os.path.join(args.output_dir, 'evaluation_results.mat')
    savemat(results_file, results)
    print(f"Results saved to {results_file}")
    
    # Create visualizations
    create_evaluation_plots(results, args.output_dir)
    
    print("Evaluation completed!")


def create_evaluation_plots(results, output_dir):
    """Create evaluation plots and save them"""
    
    # Plot 1: MSE Loss Distribution
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.hist(results['mse_losses'], bins=50, alpha=0.7)
    plt.xlabel('MSE Loss')
    plt.ylabel('Frequency')
    plt.title('MSE Loss Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Correlation Distribution  
    plt.subplot(2, 2, 2)
    valid_corr = results['correlations'][~np.isnan(results['correlations'])]
    plt.hist(valid_corr, bins=50, alpha=0.7, color='orange')
    plt.xlabel('Correlation')
    plt.ylabel('Frequency')
    plt.title('Correlation Distribution')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Sample prediction vs target
    plt.subplot(2, 2, 3)
    sample_idx = 0
    pred_sample = results['predictions'][sample_idx]
    target_sample = results['targets'][sample_idx]
    
    # Show first few time points and sources
    time_points = min(50, pred_sample.shape[0])
    sources = min(10, pred_sample.shape[1])
    
    plt.plot(pred_sample[:time_points, :sources], 'b-', alpha=0.7, label='Predicted')
    plt.plot(target_sample[:time_points, :sources], 'r--', alpha=0.7, label='Target')
    plt.xlabel('Time Points')
    plt.ylabel('Amplitude')
    plt.title(f'Sample {sample_idx} - Prediction vs Target')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Localization performance
    plt.subplot(2, 2, 4)
    if 'precision' in results['localization_results']:
        precision = results['localization_results']['precision']
        recall = results['localization_results']['recall']
        plt.scatter(precision, recall, alpha=0.6)
        plt.xlabel('Precision')
        plt.ylabel('Recall')
        plt.title('Localization Performance')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot source activation maps for a few samples
    plt.figure(figsize=(15, 10))
    
    n_samples = min(4, len(results['predictions']))
    for i in range(n_samples):
        pred = results['predictions'][i]
        target = results['targets'][i]
        
        # Sum over time to get total activation per source
        pred_sum = np.sum(np.abs(pred), axis=0)
        target_sum = np.sum(np.abs(target), axis=0)
        
        plt.subplot(2, n_samples, i + 1)
        plt.plot(pred_sum, 'b-', label='Predicted')
        plt.plot(target_sum, 'r--', label='Target')
        plt.title(f'Sample {i} - Predicted')
        plt.xlabel('Source Index')
        plt.ylabel('Total Activation')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, n_samples, i + 1 + n_samples)
        plt.plot(target_sum, 'r-', label='Target')
        plt.title(f'Sample {i} - Target')
        plt.xlabel('Source Index')
        plt.ylabel('Total Activation')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'source_activations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Evaluation plots saved to {output_dir}")


if __name__ == '__main__':
    main()
