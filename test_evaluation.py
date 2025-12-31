#!/usr/bin/env python3
"""
Test script for evaluation files
Verifies that the evaluation scripts can load models and run inference
"""

import os
import sys
import subprocess
import glob


def test_model_loading():
    """Test if we can load a trained model"""
    print("=== Testing Model Loading ===")
    
    # Check for existing models
    model_dirs = glob.glob("model_result/*_transformer_model")
    if not model_dirs:
        print("No trained models found. Please train a model first.")
        return False
    
    for model_dir in model_dirs:
        model_id = os.path.basename(model_dir).split('_')[0]
        
        # Try different checkpoint names in order of preference
        checkpoint_names = ['model_best.pth', 'model_best.pth.tar', 'checkpoint_epoch_0.pth']
        model_file = None
        for name in checkpoint_names:
            potential_path = os.path.join(model_dir, name)
            if os.path.exists(potential_path):
                model_file = potential_path
                break
        
        if model_file:
            print(f"✓ Found model {model_id}: {model_file}")
            return True, model_id
        else:
            epoch_files = glob.glob(os.path.join(model_dir, 'epoch_*'))
            if epoch_files:
                latest_epoch = sorted(epoch_files)[-1]
                epoch_num = os.path.basename(latest_epoch).split('_')[1]
                print(f"✓ Found model {model_id}, latest epoch: {epoch_num}")
                return True, model_id, epoch_num
    
    print("No valid model files found.")
    return False


def test_data_availability():
    """Test if evaluation data is available"""
    print("\n=== Testing Data Availability ===")
    
    # Check for labeled dataset
    labeled_files = glob.glob("labeled_dataset/sample_*.mat")
    if labeled_files:
        print(f"✓ Found {len(labeled_files)} labeled dataset files")
        return True
    
    # Check for demo data
    demo_files = glob.glob("demo_data/sample_*.mat")
    if demo_files:
        print(f"✓ Found {len(demo_files)} demo data files (can use for testing)")
        return True
        
    print("No evaluation data found. Please prepare data or run demo.py first.")
    return False


def run_evaluation_test(model_id, epoch=None, data_dir="demo_data"):
    """Run a quick evaluation test"""
    print(f"\n=== Testing Evaluation Scripts ===")
    
    # Test eval_transformer_sim.py
    print("Testing eval_transformer_sim.py...")
    cmd = [
        sys.executable, "eval_transformer_sim.py",
        "--model_id", str(model_id),
        "--test", data_dir,
        "--batch_size", "2",
        "--num_samples", "4",
        "--device", "cpu"
    ]
    
    if epoch:
        cmd.extend(["--resume", str(epoch)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ eval_transformer_sim.py completed successfully")
        else:
            print(f"✗ eval_transformer_sim.py failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ eval_transformer_sim.py timed out")
        return False
    except Exception as e:
        print(f"✗ Error running eval_transformer_sim.py: {e}")
        return False
    
    # Test eval_transformer_real.py
    print("Testing eval_transformer_real.py...")
    cmd = [
        sys.executable, "eval_transformer_real.py",
        "--model_id", str(model_id),
        "--data_dir", data_dir,
        "--device", "cpu"
    ]
    
    if epoch:
        cmd.extend(["--resume", str(epoch)])
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print("✓ eval_transformer_real.py completed successfully")
            return True
        else:
            print(f"✗ eval_transformer_real.py failed:")
            print(result.stderr)
            return False
    except subprocess.TimeoutExpired:
        print("✗ eval_transformer_real.py timed out")
        return False
    except Exception as e:
        print(f"✗ Error running eval_transformer_real.py: {e}")
        return False


def main():
    print("DeepSIF Transformer - Evaluation Test Script")
    print("=" * 50)
    
    # Test model loading
    model_result = test_model_loading()
    if not model_result or model_result is False:
        print("\nSuggestion: Train a model first using:")
        print("python main.py --train labeled_dataset --arch TransformerTemporalInverseNet --epoch 2")
        return
    
    if len(model_result) == 2:
        success, model_id = model_result
        epoch = None
    else:
        success, model_id, epoch = model_result
    
    # Test data availability  
    if not test_data_availability():
        print("\nSuggestion: Create demo data first using:")
        print("python demo.py")
        return
    
    # Determine data directory
    data_dir = "labeled_dataset" if glob.glob("labeled_dataset/sample_*.mat") else "demo_data"
    
    # Run evaluation tests
    if run_evaluation_test(model_id, epoch, data_dir):
        print("\n" + "=" * 50)
        print("✓ All evaluation tests passed!")
        print("\nYou can now run full evaluations using:")
        print(f"python eval_transformer_sim.py --model_id {model_id}")
        print(f"python eval_transformer_real.py --model_id {model_id}")
    else:
        print("\n" + "=" * 50)
        print("✗ Some evaluation tests failed. Check the output above.")


if __name__ == '__main__':
    main()
