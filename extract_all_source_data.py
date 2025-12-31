#!/usr/bin/env python3
"""
Extract all source data from labeled dataset and save as consolidated .mat files

Similar to eval_transformer_real.py but focuses on extracting and consolidating 
source data from multiple directories into single "all_data" files.

Usage:
    python extract_all_source_data.py --data_dir labeled_dataset --output_dir source_collections
    python extract_all_source_data.py --data_dir extracted_data/source_data --output_dir source_collections --extracted_format
"""

import os
import argparse
import glob
import time
from scipy.io import loadmat, savemat
import numpy as np
from tqdm import tqdm


def extract_source_from_file(file_path, extracted_format=False):
    """
    Extract source data from a single .mat file
    
    Args:
        file_path (str): Path to the .mat file
        extracted_format (bool): Whether file is from extracted format
        
    Returns:
        tuple: (source_data, metadata) or (None, None) if failed
    """
    try:
        data_mat = loadmat(file_path)
        
        if extracted_format:
            # From extracted_data format: sample_*_source.mat
            source_data = data_mat.get('source_data')
            metadata = {
                'sample_id': data_mat.get('sample_id'),
                'labels': data_mat.get('labels'),
                'snr': data_mat.get('snr'),
                'index': data_mat.get('index'),
                'original_file': data_mat.get('original_file')
            }
        else:
            # From labeled_dataset format: sample_*.mat
            source_data = data_mat.get('source_data')
            metadata = {
                'labels': data_mat.get('labels'),
                'snr': data_mat.get('snr'),
                'index': data_mat.get('index'),
                'eeg_data': data_mat.get('eeg_data')  # Keep EEG for reference
            }
        
        if source_data is None:
            return None, None
            
        # Verify expected shape: (500, 994)
        if source_data.shape != (500, 994):
            print(f"WARNING: Unexpected source_data shape {source_data.shape} in {file_path}")
        
        return source_data, metadata
        
    except Exception as e:
        print(f"ERROR: Failed to load {file_path}: {e}")
        return None, None


def process_directory(data_dir, extracted_format=False, normalize=True):
    """
    Process all source data files in a directory
    
    Args:
        data_dir (str): Directory containing source data files
        extracted_format (bool): Whether files are from extracted format
        normalize (bool): Whether to normalize the data
        
    Returns:
        dict: Processed data with 'all_data', 'metadata', and 'info'
    """
    print(f"\n📁 Processing directory: {data_dir}")
    
    # Find files based on format
    if extracted_format:
        pattern = os.path.join(data_dir, "sample_*_source.mat")
    else:
        pattern = os.path.join(data_dir, "sample_*.mat")
    
    file_list = glob.glob(pattern)
    file_list.sort()
    
    if len(file_list) == 0:
        print(f"WARNING: No source files found in {data_dir}")
        return None
    
    print(f"Found {len(file_list)} files")
    
    all_source_data = []
    all_metadata = []
    successful_files = []
    failed_files = []
    
    # Process files
    for file_path in tqdm(file_list, desc="Extracting source data"):
        source_data, metadata = extract_source_from_file(file_path, extracted_format)
        
        if source_data is not None:
            # Normalize data if requested (same as eval_transformer_real.py)
            if normalize:
                source_data = source_data / np.max(np.abs(source_data[:]))
            
            all_source_data.append(source_data)
            all_metadata.append(metadata)
            successful_files.append(os.path.basename(file_path))
        else:
            failed_files.append(os.path.basename(file_path))
    
    if len(all_source_data) == 0:
        print(f"ERROR: No valid source data found in {data_dir}")
        return None
    
    # Convert to numpy array
    all_data = np.array(all_source_data)  # Shape: (n_samples, 500, 994)
    
    print(f"✅ Successfully processed {len(all_source_data)} files")
    print(f"   All source data shape: {all_data.shape}")
    if failed_files:
        print(f"⚠️  Failed to process {len(failed_files)} files")
    
    return {
        'all_data': all_data,
        'metadata': all_metadata,
        'info': {
            'num_samples': len(all_source_data),
            'data_shape': all_data.shape,
            'successful_files': successful_files,
            'failed_files': failed_files,
            'source_directory': data_dir,
            'normalized': normalize,
            'extracted_format': extracted_format
        }
    }


def main():
    parser = argparse.ArgumentParser(description='Extract all source data from labeled dataset')
    parser.add_argument('--data_dir', default='labeled_dataset', type=str, 
                       help='Input directory containing source data files')
    parser.add_argument('--output_dir', default='source_collections', type=str,
                       help='Output directory for consolidated source data')
    parser.add_argument('--extracted_format', action='store_true',
                       help='Input files are from extracted format (sample_*_source.mat)')
    parser.add_argument('--no_normalize', action='store_true',
                       help='Skip data normalization')
    parser.add_argument('--subject_dirs', nargs='*', default=['real_data', 'test_data'],
                       help='Additional subject directories to process')
    parser.add_argument('--output_prefix', default='all_source_data', type=str,
                       help='Prefix for output filenames')
    
    args = parser.parse_args()
    start_time = time.time()
    
    print("=" * 60)
    print("SOURCE DATA COLLECTION STARTED")
    print("=" * 60)
    print(f"Input directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Extracted format: {args.extracted_format}")
    print(f"Normalize data: {not args.no_normalize}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process main data directory
    if os.path.exists(args.data_dir):
        print(f"\n🔍 Processing main directory: {args.data_dir}")
        
        result = process_directory(
            args.data_dir, 
            extracted_format=args.extracted_format,
            normalize=not args.no_normalize
        )
        
        if result:
            # Save consolidated source data
            dir_name = os.path.basename(args.data_dir.rstrip('/'))
            output_filename = f"{args.output_prefix}_{dir_name}.mat"
            output_path = os.path.join(args.output_dir, output_filename)
            
            savemat(output_path, {
                'all_data': result['all_data'],
                'metadata': result['metadata'],
                'info': result['info']
            })
            
            print(f"💾 Saved consolidated data to: {output_path}")
            print(f"   Shape: {result['all_data'].shape}")
            print(f"   Samples: {result['info']['num_samples']}")
    else:
        print(f"❌ Main directory not found: {args.data_dir}")
    
    # Process additional subject directories
    for subject_dir in args.subject_dirs:
        subject_path = os.path.join('source', subject_dir) if subject_dir != 'real_data' else subject_dir
        
        if os.path.exists(subject_path):
            print(f"\n🔍 Processing subject directory: {subject_path}")
            
            result = process_directory(
                subject_path,
                extracted_format=False,  # Assume standard format for subject dirs
                normalize=not args.no_normalize
            )
            
            if result:
                # Save consolidated source data for this subject
                output_filename = f"{args.output_prefix}_{subject_dir}.mat"
                output_path = os.path.join(args.output_dir, output_filename)
                
                savemat(output_path, {
                    'all_data': result['all_data'],
                    'metadata': result['metadata'],
                    'info': result['info']
                })
                
                print(f"💾 Saved {subject_dir} data to: {output_path}")
                print(f"   Shape: {result['all_data'].shape}")
                print(f"   Samples: {result['info']['num_samples']}")
        else:
            print(f"⚠️  Subject directory not found: {subject_path}")
    
    total_time = time.time() - start_time
    print("=" * 60)
    print("SOURCE DATA COLLECTION COMPLETED")
    print("=" * 60)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Output directory: {args.output_dir}")
    
    # List generated files
    output_files = glob.glob(os.path.join(args.output_dir, "*.mat"))
    if output_files:
        print(f"\n📁 Generated files ({len(output_files)}):")
        for output_file in output_files:
            file_size = os.path.getsize(output_file) / (1024 * 1024)  # MB
            print(f"   {os.path.basename(output_file)} ({file_size:.1f} MB)")


def verify_collected_data(data_dir="source_collections"):
    """Verify the collected source data files"""
    
    print(f"\n🔍 Verifying collected data in {data_dir}")
    
    if not os.path.exists(data_dir):
        print(f"❌ Directory not found: {data_dir}")
        return
    
    mat_files = glob.glob(os.path.join(data_dir, "*.mat"))
    
    for mat_file in mat_files:
        try:
            data = loadmat(mat_file)
            all_data = data['all_data']
            info = data['info']
            
            print(f"\n📄 {os.path.basename(mat_file)}:")
            print(f"   Data shape: {all_data.shape}")
            print(f"   Samples: {info['num_samples'][0][0] if hasattr(info['num_samples'][0], '__len__') else info['num_samples']}")
            print(f"   Source dir: {info['source_directory'][0] if hasattr(info['source_directory'], '__len__') else info['source_directory']}")
            
            # Basic stats
            print(f"   Data range: [{all_data.min():.6f}, {all_data.max():.6f}]")
            print(f"   Mean: {all_data.mean():.6f}, Std: {all_data.std():.6f}")
            
        except Exception as e:
            print(f"❌ Error verifying {os.path.basename(mat_file)}: {e}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--verify':
        verify_collected_data()
    else:
        main()
