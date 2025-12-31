#!/usr/bin/env python3
"""
Demo script for DeepSIF Transformer
This script demonstrates the basic functionality without requiring the full dataset.
"""

import torch
import numpy as np
from scipy.io import savemat
import os
import network
import loaders
from utils import create_synthetic_forward_matrix


def create_demo_data(num_samples=10, data_dir='demo_data'):
    """Create synthetic demo data in the format from extract_labeled_data.py"""
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    print(f"Creating {num_samples} demo samples in {data_dir}/")
    print("Note: This mimics the format from extract_labeled_data.py (pre-processed data)")
    
    for i in range(num_samples):
        # Create pre-processed EEG data (already normalized and processed)
        eeg_data = np.random.randn(500, 75) * 0.1
        
        # Create pre-processed source data (already normalized)
        source_data = np.zeros((500, 994))
        active_sources = np.random.choice(994, size=np.random.randint(2, 8), replace=False)
        
        for src in active_sources:
            # Create realistic source signals
            t = np.linspace(0, 1, 500)
            freq = np.random.uniform(1, 10)  # Hz
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.5, 2.0)
            
            signal = amplitude * np.sin(2 * np.pi * freq * t + phase) * np.exp(-t*2)
            source_data[:, src] = signal
        
        # Normalize the data (as done in extract_labeled_data.py)
        if np.max(np.abs(eeg_data)) > 0:
            eeg_data = eeg_data / np.max(np.abs(eeg_data))
        if np.max(np.abs(source_data)) > 0:
            source_data = source_data / np.max(np.abs(source_data))
        
        # Create labels (active sources only - no padding as in processed data)
        labels = active_sources.astype(float)
        
        # Create other required fields (as single values, not arrays)
        index = float(i)
        snr = 10.0  # 10 dB SNR
        
        # Save as .mat file in the format from extract_labeled_data.py
        savemat(f'{data_dir}/sample_{i:05d}.mat', {
            'eeg_data': eeg_data,
            'source_data': source_data, 
            'labels': labels,  # Active sources only
            'index': index,
            'snr': snr
        })
    
    print(f"Demo data created successfully!")
    print(f"Format matches extract_labeled_data.py output")
    return data_dir


def test_data_loader(data_dir):
    """Test the data loader with demo data"""
    print("\n=== Testing Data Loader ===")
    print("Loading pre-processed data (no additional processing needed)")
    
    dataset = loaders.LabeledDatasetLoader(
        data_dir,
        fwd=None,  # Not needed for pre-processed data
        args_params={'dataset_len': 5}
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"EEG data shape: {sample['data'].shape}")
    print(f"Source data shape: {sample['nmm'].shape}")
    print(f"Padded labels shape: {sample['label'].shape}")
    print(f"Valid labels: {sample['valid_labels']}")
    print(f"SNR: {sample['snr']}")
    print(f"Data types: EEG={sample['data'].dtype}, Source={sample['nmm'].dtype}")
    
    return dataset


def test_model():
    """Test the Transformer model"""
    print("\n=== Testing Transformer Model ===")
    
    # Create model
    model = network.TransformerTemporalInverseNet(
        num_sensor=75,
        num_source=994,
        transformer_layers=2,
        d_model=256,
        nhead=4,
        dropout=0.1
    )
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 4
    seq_len = 500
    num_sensors = 75
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, seq_len, num_sensors)
    
    model.eval()
    with torch.no_grad():
        output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output['last'].shape}")
    print(f"Expected output shape: ({batch_size}, {seq_len}, 994)")
    
    return model


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


def test_training_loop(data_dir):
    """Test a minimal training loop"""
    print("\n=== Testing Training Loop ===")
    
    # Create data loader for pre-processed data
    dataset = loaders.LabeledDatasetLoader(data_dir, fwd=None)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=2, 
        shuffle=True, 
        collate_fn=custom_collate_fn
    )
    
    # Create model
    model = network.TransformerTemporalInverseNet(
        num_sensor=75,
        num_source=994,
        transformer_layers=2,
        d_model=128,
        nhead=4,
        dropout=0.1
    )
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    # Training loop for a few iterations
    model.train()
    losses = []
    
    for epoch in range(2):
        epoch_losses = []
        for batch_idx, batch in enumerate(dataloader):
            data = batch['data']
            target = batch['nmm']
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output['last'], target)
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            
        losses.extend(epoch_losses)
        avg_loss = np.mean(epoch_losses)
        print(f"Epoch {epoch+1}, Average Loss: {avg_loss:.6f}")
    
    print(f"Training test completed. Final loss: {losses[-1]:.6f}")
    return model, losses


def main():
    """Run all demo tests"""
    print("DeepSIF Transformer Demo")
    print("=" * 40)
    
    # Create demo data
    data_dir = create_demo_data(num_samples=10)
    
    # Test data loader
    dataset = test_data_loader(data_dir)
    
    # Test model architecture
    model = test_model()
    
    # Test training loop
    trained_model, losses = test_training_loop(data_dir)
    
    print("\n=== Demo Summary ===")
    print(f"✓ Demo data created: {data_dir} (pre-processed format)")
    print(f"✓ Data loader working: {len(dataset)} samples")
    print(f"✓ Model architecture: {model.count_parameters():,} parameters")
    print(f"✓ Training loop: Final loss {losses[-1]:.6f}")
    
    print("\nDemo completed successfully!")
    print("\nTo use your actual labeled dataset:")
    print("1. Place your sample_*.mat files from extract_labeled_data.py in labeled_dataset/")
    print("2. Run: python main.py --train labeled_dataset --arch TransformerTemporalInverseNet")
    print("\nThe data loader will automatically load the pre-processed .mat files!")


if __name__ == '__main__':
    main()
