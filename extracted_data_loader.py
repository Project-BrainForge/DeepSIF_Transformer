#!/usr/bin/env python3
"""
Data loader for extracted EEG and Source data files

This loader can work with the output from extract_eeg_source_data.py
Supports loading EEG-only, Source-only, or paired EEG-Source data.
"""

import os
import glob
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.io import loadmat
import numpy as np


class ExtractedEEGDataset(Dataset):
    """Dataset for loading only EEG data from extracted files"""
    
    def __init__(self, eeg_data_dir):
        """
        Args:
            eeg_data_dir (str): Directory containing sample_*_eeg.mat files
        """
        self.eeg_data_dir = eeg_data_dir
        self.eeg_files = glob.glob(os.path.join(eeg_data_dir, "sample_*_eeg.mat"))
        self.eeg_files.sort()
        
        if len(self.eeg_files) == 0:
            raise ValueError(f"No EEG files found in {eeg_data_dir}")
    
    def __len__(self):
        return len(self.eeg_files)
    
    def __getitem__(self, idx):
        eeg_file = self.eeg_files[idx]
        data = loadmat(eeg_file)
        
        eeg_data = torch.FloatTensor(data['eeg_data'])  # (500, 75)
        sample_id = data['sample_id'][0] if isinstance(data['sample_id'], np.ndarray) else data['sample_id']
        
        # Include metadata if available
        metadata = {}
        for key in ['labels', 'snr', 'index']:
            if key in data and data[key] is not None:
                metadata[key] = data[key]
        
        return {
            'eeg_data': eeg_data,
            'sample_id': sample_id,
            'metadata': metadata
        }


class ExtractedSourceDataset(Dataset):
    """Dataset for loading only Source data from extracted files"""
    
    def __init__(self, source_data_dir):
        """
        Args:
            source_data_dir (str): Directory containing sample_*_source.mat files
        """
        self.source_data_dir = source_data_dir
        self.source_files = glob.glob(os.path.join(source_data_dir, "sample_*_source.mat"))
        self.source_files.sort()
        
        if len(self.source_files) == 0:
            raise ValueError(f"No source files found in {source_data_dir}")
    
    def __len__(self):
        return len(self.source_files)
    
    def __getitem__(self, idx):
        source_file = self.source_files[idx]
        data = loadmat(source_file)
        
        source_data = torch.FloatTensor(data['source_data'])  # (500, 994)
        sample_id = data['sample_id'][0] if isinstance(data['sample_id'], np.ndarray) else data['sample_id']
        
        # Include metadata if available
        metadata = {}
        for key in ['labels', 'snr', 'index']:
            if key in data and data[key] is not None:
                metadata[key] = data[key]
        
        return {
            'source_data': source_data,
            'sample_id': sample_id,
            'metadata': metadata
        }


class ExtractedPairedDataset(Dataset):
    """Dataset for loading paired EEG and Source data from extracted files"""
    
    def __init__(self, extracted_data_dir):
        """
        Args:
            extracted_data_dir (str): Base directory containing eeg_data/ and source_data/ subdirectories
        """
        self.eeg_data_dir = os.path.join(extracted_data_dir, 'eeg_data')
        self.source_data_dir = os.path.join(extracted_data_dir, 'source_data')
        
        # Find all sample IDs
        eeg_files = glob.glob(os.path.join(self.eeg_data_dir, "sample_*_eeg.mat"))
        source_files = glob.glob(os.path.join(self.source_data_dir, "sample_*_source.mat"))
        
        # Extract sample IDs
        eeg_ids = set(os.path.basename(f).replace('sample_', '').replace('_eeg.mat', '') for f in eeg_files)
        source_ids = set(os.path.basename(f).replace('sample_', '').replace('_source.mat', '') for f in source_files)
        
        # Find common sample IDs
        self.sample_ids = sorted(list(eeg_ids.intersection(source_ids)))
        
        if len(self.sample_ids) == 0:
            raise ValueError(f"No matching EEG-Source pairs found in {extracted_data_dir}")
    
    def __len__(self):
        return len(self.sample_ids)
    
    def __getitem__(self, idx):
        sample_id = self.sample_ids[idx]
        
        # Load EEG data
        eeg_file = os.path.join(self.eeg_data_dir, f"sample_{sample_id}_eeg.mat")
        eeg_data_dict = loadmat(eeg_file)
        eeg_data = torch.FloatTensor(eeg_data_dict['eeg_data'])  # (500, 75)
        
        # Load Source data
        source_file = os.path.join(self.source_data_dir, f"sample_{sample_id}_source.mat")
        source_data_dict = loadmat(source_file)
        source_data = torch.FloatTensor(source_data_dict['source_data'])  # (500, 994)
        
        # Combine metadata from both files
        metadata = {}
        for key in ['labels', 'snr', 'index']:
            # Prefer metadata from EEG file, fallback to source file
            if key in eeg_data_dict and eeg_data_dict[key] is not None:
                metadata[key] = eeg_data_dict[key]
            elif key in source_data_dict and source_data_dict[key] is not None:
                metadata[key] = source_data_dict[key]
        
        return {
            'eeg_data': eeg_data,
            'source_data': source_data,
            'sample_id': sample_id,
            'metadata': metadata
        }


def create_eeg_dataloader(eeg_data_dir, batch_size=16, shuffle=True, num_workers=0):
    """Create DataLoader for EEG-only data"""
    dataset = ExtractedEEGDataset(eeg_data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_source_dataloader(source_data_dir, batch_size=16, shuffle=True, num_workers=0):
    """Create DataLoader for Source-only data"""
    dataset = ExtractedSourceDataset(source_data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_paired_dataloader(extracted_data_dir, batch_size=16, shuffle=True, num_workers=0):
    """Create DataLoader for paired EEG-Source data"""
    dataset = ExtractedPairedDataset(extracted_data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def test_extracted_loaders():
    """Test function for the extracted data loaders"""
    
    print("🔧 Testing Extracted Data Loaders")
    print("=" * 40)
    
    # Test directories (adjust paths as needed)
    test_dirs = {
        'extracted_data': 'extracted_data',
        'demo_extracted_data': 'demo_extracted_data'
    }
    
    for name, base_dir in test_dirs.items():
        if not os.path.exists(base_dir):
            print(f"⚠️  {name} directory not found, skipping...")
            continue
        
        print(f"\n📁 Testing with {name}/")
        
        try:
            # Test EEG-only loader
            eeg_dir = os.path.join(base_dir, 'eeg_data')
            if os.path.exists(eeg_dir):
                eeg_loader = create_eeg_dataloader(eeg_dir, batch_size=2)
                eeg_batch = next(iter(eeg_loader))
                print(f"✅ EEG loader: batch_size={len(eeg_batch['eeg_data'])}, "
                      f"eeg_shape={eeg_batch['eeg_data'].shape}")
            
            # Test Source-only loader
            source_dir = os.path.join(base_dir, 'source_data')
            if os.path.exists(source_dir):
                source_loader = create_source_dataloader(source_dir, batch_size=2)
                source_batch = next(iter(source_loader))
                print(f"✅ Source loader: batch_size={len(source_batch['source_data'])}, "
                      f"source_shape={source_batch['source_data'].shape}")
            
            # Test paired loader
            if os.path.exists(eeg_dir) and os.path.exists(source_dir):
                paired_loader = create_paired_dataloader(base_dir, batch_size=2)
                paired_batch = next(iter(paired_loader))
                print(f"✅ Paired loader: batch_size={len(paired_batch['eeg_data'])}, "
                      f"eeg_shape={paired_batch['eeg_data'].shape}, "
                      f"source_shape={paired_batch['source_data'].shape}")
        
        except Exception as e:
            print(f"❌ Error testing {name}: {str(e)}")
    
    print("\n✅ Loader testing completed!")


if __name__ == "__main__":
    test_extracted_loaders()
