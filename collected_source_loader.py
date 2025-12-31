#!/usr/bin/env python3
"""
Data loader for collected source data files (output from extract_all_source_data.py)

Similar to how eval_transformer_real.py processes data, but specialized for 
working with consolidated source data collections.
"""

import os
import glob
from torch.utils.data import Dataset, DataLoader
import torch
from scipy.io import loadmat, savemat
import numpy as np


class CollectedSourceDataset(Dataset):
    """Dataset for loading consolidated source data collections"""
    
    def __init__(self, collection_file, return_metadata=False):
        """
        Args:
            collection_file (str): Path to consolidated source data .mat file
            return_metadata (bool): Whether to return metadata with each sample
        """
        self.collection_file = collection_file
        self.return_metadata = return_metadata
        
        # Load the consolidated data
        data = loadmat(collection_file)
        self.all_data = data['all_data']  # Shape: (n_samples, 500, 994)
        self.metadata = data.get('metadata', None)
        self.info = data.get('info', {})
        
        # Validate data
        if len(self.all_data.shape) != 3:
            raise ValueError(f"Expected 3D data (n_samples, 500, 994), got shape {self.all_data.shape}")
        
        self.n_samples = self.all_data.shape[0]
        
        print(f"Loaded {self.n_samples} source samples from {os.path.basename(collection_file)}")
        print(f"Data shape: {self.all_data.shape}")
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        source_data = torch.FloatTensor(self.all_data[idx])  # Shape: (500, 994)
        
        sample = {
            'source_data': source_data,
            'sample_idx': idx
        }
        
        # Add metadata if requested and available
        if self.return_metadata and self.metadata is not None and idx < len(self.metadata):
            sample['metadata'] = self.metadata[idx]
        
        return sample


class MultiCollectionDataset(Dataset):
    """Dataset for loading multiple source data collections"""
    
    def __init__(self, collection_dir, file_pattern="*.mat", return_metadata=False):
        """
        Args:
            collection_dir (str): Directory containing collection .mat files
            file_pattern (str): Pattern to match collection files
            return_metadata (bool): Whether to return metadata with each sample
        """
        self.collection_dir = collection_dir
        self.return_metadata = return_metadata
        
        # Find collection files
        collection_files = glob.glob(os.path.join(collection_dir, file_pattern))
        collection_files.sort()
        
        if not collection_files:
            raise ValueError(f"No collection files found in {collection_dir} with pattern {file_pattern}")
        
        # Load all collections
        self.collections = []
        self.cumulative_sizes = [0]
        total_samples = 0
        
        for collection_file in collection_files:
            try:
                data = loadmat(collection_file)
                all_data = data['all_data']
                metadata = data.get('metadata', None)
                info = data.get('info', {})
                
                collection_info = {
                    'file': collection_file,
                    'data': all_data,
                    'metadata': metadata,
                    'info': info,
                    'n_samples': all_data.shape[0]
                }
                
                self.collections.append(collection_info)
                total_samples += all_data.shape[0]
                self.cumulative_sizes.append(total_samples)
                
                print(f"Loaded collection: {os.path.basename(collection_file)} ({all_data.shape[0]} samples)")
                
            except Exception as e:
                print(f"Warning: Failed to load {collection_file}: {e}")
        
        if not self.collections:
            raise ValueError(f"No valid collection files found in {collection_dir}")
        
        self.total_samples = total_samples
        print(f"Total samples across {len(self.collections)} collections: {self.total_samples}")
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Find which collection this index belongs to
        collection_idx = 0
        for i, cum_size in enumerate(self.cumulative_sizes[1:], 1):
            if idx < cum_size:
                collection_idx = i - 1
                break
        
        # Calculate local index within the collection
        local_idx = idx - self.cumulative_sizes[collection_idx]
        collection = self.collections[collection_idx]
        
        source_data = torch.FloatTensor(collection['data'][local_idx])  # Shape: (500, 994)
        
        sample = {
            'source_data': source_data,
            'sample_idx': idx,
            'collection_idx': collection_idx,
            'local_idx': local_idx,
            'collection_file': os.path.basename(collection['file'])
        }
        
        # Add metadata if requested and available
        if (self.return_metadata and collection['metadata'] is not None 
            and local_idx < len(collection['metadata'])):
            sample['metadata'] = collection['metadata'][local_idx]
        
        return sample


def create_collected_source_dataloader(collection_file, batch_size=16, shuffle=False, num_workers=0, 
                                     return_metadata=False):
    """Create DataLoader for a single source data collection"""
    dataset = CollectedSourceDataset(collection_file, return_metadata=return_metadata)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def create_multi_collection_dataloader(collection_dir, batch_size=16, shuffle=False, num_workers=0,
                                     file_pattern="*.mat", return_metadata=False):
    """Create DataLoader for multiple source data collections"""
    dataset = MultiCollectionDataset(collection_dir, file_pattern=file_pattern, 
                                   return_metadata=return_metadata)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def process_collected_source_data(collection_dir, model, device, output_dir="processed_results"):
    """
    Process collected source data through a model (similar to eval_transformer_real.py)
    
    Args:
        collection_dir (str): Directory containing source data collections
        model: Trained model for processing
        device: Device to run on
        output_dir (str): Directory to save processed results
    """
    print(f"🔄 Processing collected source data from {collection_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all collection files
    collection_files = glob.glob(os.path.join(collection_dir, "*.mat"))
    collection_files.sort()
    
    if not collection_files:
        print(f"❌ No collection files found in {collection_dir}")
        return
    
    model.eval()
    
    for collection_file in collection_files:
        print(f"\n📄 Processing: {os.path.basename(collection_file)}")
        
        try:
            # Load collection
            data = loadmat(collection_file)
            all_source_data = data['all_data']  # Shape: (n_samples, 500, 994)
            info = data.get('info', {})
            
            print(f"   Data shape: {all_source_data.shape}")
            
            # Convert to tensor and process
            source_tensor = torch.from_numpy(all_source_data).to(device, torch.float)
            
            with torch.no_grad():
                # Assuming model can process source data directly
                # Adjust this based on your model's input requirements
                processed_output = model(source_tensor)
                if isinstance(processed_output, dict):
                    processed_output = processed_output.get('last', processed_output)
            
            # Convert back to numpy
            processed_data = processed_output.detach().cpu().numpy()
            print(f"   Processed shape: {processed_data.shape}")
            
            # Save processed results
            base_name = os.path.splitext(os.path.basename(collection_file))[0]
            output_filename = f"processed_{base_name}.mat"
            output_path = os.path.join(output_dir, output_filename)
            
            savemat(output_path, {
                'processed_data': processed_data,
                'original_info': info,
                'processing_info': {
                    'model_type': type(model).__name__,
                    'input_shape': all_source_data.shape,
                    'output_shape': processed_data.shape,
                    'device': str(device)
                }
            })
            
            print(f"   💾 Saved: {output_filename}")
            
        except Exception as e:
            print(f"❌ Error processing {os.path.basename(collection_file)}: {e}")
    
    print(f"\n✅ Processing completed. Results saved to: {output_dir}")


def analyze_collections(collection_dir):
    """Analyze all source data collections in a directory"""
    
    print(f"🔍 Analyzing Source Data Collections in {collection_dir}")
    print("=" * 60)
    
    if not os.path.exists(collection_dir):
        print(f"❌ Directory not found: {collection_dir}")
        return
    
    collection_files = glob.glob(os.path.join(collection_dir, "*.mat"))
    collection_files.sort()
    
    if not collection_files:
        print(f"❌ No collection files found in {collection_dir}")
        return
    
    total_samples = 0
    total_size_mb = 0
    
    for i, collection_file in enumerate(collection_files, 1):
        try:
            data = loadmat(collection_file)
            all_data = data['all_data']
            info = data.get('info', {})
            
            file_size_mb = os.path.getsize(collection_file) / (1024 * 1024)
            total_size_mb += file_size_mb
            total_samples += all_data.shape[0]
            
            print(f"\n{i}. {os.path.basename(collection_file)}")
            print(f"   📊 Shape: {all_data.shape}")
            print(f"   📈 Samples: {all_data.shape[0]}")
            print(f"   💾 Size: {file_size_mb:.1f} MB")
            print(f"   📐 Range: [{all_data.min():.6f}, {all_data.max():.6f}]")
            print(f"   📊 Mean: {all_data.mean():.6f}, Std: {all_data.std():.6f}")
            
            # Show info if available
            if info:
                print(f"   📝 Info: {list(info.keys())}")
                
        except Exception as e:
            print(f"❌ Error analyzing {os.path.basename(collection_file)}: {e}")
    
    print(f"\n📊 SUMMARY:")
    print(f"   Total collections: {len(collection_files)}")
    print(f"   Total samples: {total_samples}")
    print(f"   Total size: {total_size_mb:.1f} MB")


def test_collected_loaders():
    """Test the collected source data loaders"""
    
    print("🔧 Testing Collected Source Data Loaders")
    print("=" * 50)
    
    # Test directories
    test_dirs = ['source_collections', 'demo_source_collections']
    
    for test_dir in test_dirs:
        if not os.path.exists(test_dir):
            print(f"⚠️  {test_dir} not found, skipping...")
            continue
        
        print(f"\n📁 Testing with {test_dir}/")
        
        try:
            # Test multi-collection loader
            loader = create_multi_collection_dataloader(test_dir, batch_size=4)
            batch = next(iter(loader))
            
            print(f"✅ Multi-collection loader:")
            print(f"   Batch size: {len(batch['source_data'])}")
            print(f"   Source data shape: {batch['source_data'].shape}")
            print(f"   Collection files: {set(batch['collection_file'])}")
            
            # Test single file if available
            collection_files = glob.glob(os.path.join(test_dir, "*.mat"))
            if collection_files:
                single_loader = create_collected_source_dataloader(collection_files[0], batch_size=2)
                single_batch = next(iter(single_loader))
                
                print(f"✅ Single collection loader:")
                print(f"   Batch size: {len(single_batch['source_data'])}")
                print(f"   Source data shape: {single_batch['source_data'].shape}")
        
        except Exception as e:
            print(f"❌ Error testing {test_dir}: {e}")
    
    print("\n✅ Loader testing completed!")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == '--test':
            test_collected_loaders()
        elif sys.argv[1] == '--analyze':
            analyze_dir = sys.argv[2] if len(sys.argv) > 2 else 'source_collections'
            analyze_collections(analyze_dir)
    else:
        test_collected_loaders()
