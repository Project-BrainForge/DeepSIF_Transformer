#!/usr/bin/env python3
"""
Extract EEG and Source data from labeled dataset into separate .mat files

This script processes the labeled_dataset directory and extracts:
- eeg_data (500 x 75) -> saved to eeg_data/sample_XXXXX_eeg.mat  
- source_data (500 x 994) -> saved to source_data/sample_XXXXX_source.mat

Usage:
    python extract_eeg_source_data.py --input_dir labeled_dataset --output_dir extracted_data
"""

import os
import argparse
import glob
from scipy.io import loadmat, savemat
import numpy as np
from tqdm import tqdm
import logging


def setup_logging():
    """Setup logging for extraction process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('extract_eeg_source.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def extract_sample_data(mat_file_path):
    """
    Extract EEG and source data from a single .mat file
    
    Args:
        mat_file_path (str): Path to the .mat file
        
    Returns:
        tuple: (eeg_data, source_data, metadata) or (None, None, None) if failed
    """
    try:
        data = loadmat(mat_file_path)
        
        # Extract the main fields
        eeg_data = data.get('eeg_data')
        source_data = data.get('source_data')
        labels = data.get('labels')
        snr = data.get('snr')
        index = data.get('index')
        
        # Verify data shapes
        if eeg_data is None:
            return None, None, None, "Missing eeg_data"
        if source_data is None:
            return None, None, None, "Missing source_data"
            
        # Expected shapes: eeg_data(500, 75), source_data(500, 994)
        expected_eeg_shape = (500, 75)
        expected_source_shape = (500, 994)
        
        if eeg_data.shape != expected_eeg_shape:
            return None, None, None, f"Unexpected eeg_data shape: {eeg_data.shape}, expected: {expected_eeg_shape}"
        if source_data.shape != expected_source_shape:
            return None, None, None, f"Unexpected source_data shape: {source_data.shape}, expected: {expected_source_shape}"
        
        # Create metadata dictionary
        metadata = {
            'labels': labels,
            'snr': snr,
            'index': index,
            'original_file': os.path.basename(mat_file_path)
        }
        
        return eeg_data, source_data, metadata, None
        
    except Exception as e:
        return None, None, None, f"Error loading {mat_file_path}: {str(e)}"


def save_extracted_data(sample_id, eeg_data, source_data, metadata, eeg_dir, source_dir):
    """
    Save extracted data to separate .mat files
    
    Args:
        sample_id (str): Sample identifier (e.g., '00000')
        eeg_data (np.ndarray): EEG data array (500, 75)
        source_data (np.ndarray): Source data array (500, 994)
        metadata (dict): Additional metadata
        eeg_dir (str): Directory to save EEG files
        source_dir (str): Directory to save source files
    """
    # Save EEG data
    eeg_filename = f"sample_{sample_id}_eeg.mat"
    eeg_path = os.path.join(eeg_dir, eeg_filename)
    eeg_save_dict = {
        'eeg_data': eeg_data,
        'sample_id': sample_id,
        'data_shape': eeg_data.shape,
        'data_type': 'eeg',
        **metadata  # Include all metadata
    }
    savemat(eeg_path, eeg_save_dict)
    
    # Save Source data
    source_filename = f"sample_{sample_id}_source.mat"
    source_path = os.path.join(source_dir, source_filename)
    source_save_dict = {
        'source_data': source_data,
        'sample_id': sample_id,
        'data_shape': source_data.shape,
        'data_type': 'source',
        **metadata  # Include all metadata
    }
    savemat(source_path, source_save_dict)
    
    return eeg_path, source_path


def main():
    parser = argparse.ArgumentParser(description='Extract EEG and Source data from labeled dataset')
    parser.add_argument('--input_dir', default='labeled_dataset', type=str, 
                       help='Input directory containing sample_*.mat files')
    parser.add_argument('--output_dir', default='extracted_data', type=str,
                       help='Output directory for extracted data')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to process (for testing)')
    parser.add_argument('--verify_extraction', action='store_true',
                       help='Verify extracted files by reloading and checking')
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    logger.info("=" * 60)
    logger.info("EEG AND SOURCE DATA EXTRACTION STARTED")
    logger.info("=" * 60)
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory not found: {args.input_dir}")
        return
    
    # Find all sample files
    pattern = os.path.join(args.input_dir, "sample_*.mat")
    sample_files = glob.glob(pattern)
    sample_files.sort()
    
    if len(sample_files) == 0:
        logger.error(f"No sample_*.mat files found in {args.input_dir}")
        return
    
    logger.info(f"Found {len(sample_files)} sample files")
    
    # Limit samples if specified
    if args.max_samples:
        sample_files = sample_files[:args.max_samples]
        logger.info(f"Processing only first {len(sample_files)} samples")
    
    # Create output directories
    eeg_dir = os.path.join(args.output_dir, 'eeg_data')
    source_dir = os.path.join(args.output_dir, 'source_data')
    
    os.makedirs(eeg_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)
    
    logger.info(f"Output directories:")
    logger.info(f"  EEG data: {eeg_dir}")
    logger.info(f"  Source data: {source_dir}")
    
    # Process files
    successful_extractions = 0
    failed_extractions = 0
    extraction_errors = []
    
    logger.info("Starting extraction process...")
    
    for sample_file in tqdm(sample_files, desc="Extracting data"):
        # Extract sample ID from filename
        basename = os.path.basename(sample_file)
        # Expecting format: sample_XXXXX.mat
        sample_id = basename.replace('sample_', '').replace('.mat', '')
        
        # Extract data
        eeg_data, source_data, metadata, error = extract_sample_data(sample_file)
        
        if error:
            failed_extractions += 1
            extraction_errors.append(f"Sample {sample_id}: {error}")
            logger.warning(f"Failed to extract {basename}: {error}")
            continue
        
        try:
            # Save extracted data
            eeg_path, source_path = save_extracted_data(
                sample_id, eeg_data, source_data, metadata, eeg_dir, source_dir
            )
            successful_extractions += 1
            
            # Log first few extractions for verification
            if successful_extractions <= 5:
                logger.info(f"✅ Extracted {basename}:")
                logger.info(f"   EEG: {eeg_data.shape} -> {os.path.basename(eeg_path)}")
                logger.info(f"   Source: {source_data.shape} -> {os.path.basename(source_path)}")
                if metadata['labels'] is not None:
                    logger.info(f"   Labels: {len(metadata['labels'])} active sources")
                if metadata['snr'] is not None:
                    logger.info(f"   SNR: {metadata['snr']}")
        
        except Exception as e:
            failed_extractions += 1
            extraction_errors.append(f"Sample {sample_id}: Error saving - {str(e)}")
            logger.error(f"Failed to save {basename}: {str(e)}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("EXTRACTION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total samples processed: {len(sample_files)}")
    logger.info(f"Successful extractions: {successful_extractions}")
    logger.info(f"Failed extractions: {failed_extractions}")
    
    if successful_extractions > 0:
        logger.info(f"\n📁 Extracted data saved to:")
        logger.info(f"   EEG files: {eeg_dir}/ (sample_*_eeg.mat)")
        logger.info(f"   Source files: {source_dir}/ (sample_*_source.mat)")
    
    if extraction_errors:
        logger.warning(f"\n⚠️  Extraction errors ({len(extraction_errors)}):")
        for error in extraction_errors[:10]:  # Show first 10 errors
            logger.warning(f"   {error}")
        if len(extraction_errors) > 10:
            logger.warning(f"   ... and {len(extraction_errors) - 10} more errors")
    
    # Verification step
    if args.verify_extraction and successful_extractions > 0:
        logger.info("\n🔍 Verifying extracted files...")
        verify_extracted_data(eeg_dir, source_dir, logger)
    
    logger.info("=" * 60)
    logger.info("EXTRACTION COMPLETED")
    logger.info("=" * 60)


def verify_extracted_data(eeg_dir, source_dir, logger):
    """Verify that extracted files can be loaded correctly"""
    
    # Get a few sample files to verify
    eeg_files = glob.glob(os.path.join(eeg_dir, "*.mat"))[:5]
    source_files = glob.glob(os.path.join(source_dir, "*.mat"))[:5]
    
    logger.info(f"Verifying {len(eeg_files)} EEG files and {len(source_files)} source files...")
    
    verification_success = True
    
    for eeg_file in eeg_files:
        try:
            data = loadmat(eeg_file)
            eeg_data = data['eeg_data']
            expected_shape = (500, 75)
            
            if eeg_data.shape != expected_shape:
                logger.error(f"❌ EEG verification failed: {os.path.basename(eeg_file)} has shape {eeg_data.shape}, expected {expected_shape}")
                verification_success = False
            else:
                logger.debug(f"✅ EEG file verified: {os.path.basename(eeg_file)}")
                
        except Exception as e:
            logger.error(f"❌ Error verifying {os.path.basename(eeg_file)}: {str(e)}")
            verification_success = False
    
    for source_file in source_files:
        try:
            data = loadmat(source_file)
            source_data = data['source_data']
            expected_shape = (500, 994)
            
            if source_data.shape != expected_shape:
                logger.error(f"❌ Source verification failed: {os.path.basename(source_file)} has shape {source_data.shape}, expected {expected_shape}")
                verification_success = False
            else:
                logger.debug(f"✅ Source file verified: {os.path.basename(source_file)}")
                
        except Exception as e:
            logger.error(f"❌ Error verifying {os.path.basename(source_file)}: {str(e)}")
            verification_success = False
    
    if verification_success:
        logger.info("✅ All verification checks passed!")
    else:
        logger.warning("⚠️  Some verification checks failed!")


if __name__ == "__main__":
    main()
