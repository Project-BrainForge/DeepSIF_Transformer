# Quick Start Guide - CNN Spatial Filter

Get started with the CNN spatial filter in 5 minutes!

## Step 1: Install MNE (1 minute)

```bash
pip install mne
```

Verify:
```bash
python -c "import mne; print('MNE installed:', mne.__version__)"
```

## Step 2: Train CNN Model (5-10 minutes on GPU)

```bash
python train_cnn.py --model_id my_first_cnn --device cuda:0
```

**Expected output:**
```
================================================================================
CNN SPATIAL FILTER TRAINING SESSION
================================================================================
PyTorch version: 2.x.x
CUDA available: True
CUDA device: GeForce RTX 3090
...
[EPOCH 0] Starting training on 87 batches
[Batch    0/   87] Loss: 0.123456 | GradNorm: 0.234567
...
```

**Files created:**
```
model_result/my_first_cnn_cnn_spatial/
├── model_best.pth           ← Best model weights
├── checkpoint_epoch_0.pth   ← Checkpoint
└── training_cnn_my_first_cnn.log  ← Training logs
```

**Tip:** On CPU, training is slow. Use `--device cuda:0` for GPU acceleration.

## Step 3: Evaluate on Test Data (1 minute)

```bash
python eval_cnn_sim.py --model_id my_first_cnn --device cuda:0
```

**Expected output:**
```
================================================================================
LOADING CNN SPATIAL FILTER MODEL FOR SIMULATED DATA
================================================================================
Found checkpoint: model_result/my_first_cnn_cnn_spatial/model_best.pth
Model type: cnn_spatial
Model parameters: 2,034,567

Found 100 test samples
Input shape: (100, 500, 75)
Output shape: (100, 500, 994)

MSE: 0.012345
Correlation: 0.876543

Results saved: model_result/my_first_cnn_cnn_spatial/cnn_sim_results.mat
```

## Step 4: Evaluate on Real Data (1 minute)

```bash
python eval_cnn_real.py --model_id my_first_cnn --device cuda:0
```

**Expected output:**
```
================================================================================
EVALUATING ON REAL DATA
================================================================================

Found 42 files in real_data
Loaded 42 samples from real_data
Input shape: (42, 500, 75)
Output shape: (42, 500, 994)

Results saved: real_data/cnn_real_my_first_cnn.mat
Processing time: 2.34s
```

## Step 5: View Results (optional)

Load and examine results:

```python
import scipy.io as sio
import numpy as np

# Load simulated evaluation
results = sio.loadmat('model_result/my_first_cnn_cnn_spatial/cnn_sim_results.mat')
print(f"Predictions shape: {results['predictions'].shape}")
print(f"MSE: {results['metrics']['mse'][0,0]:.6f}")
print(f"Correlation: {results['metrics']['correlation'][0,0]:.6f}")

# Load real data evaluation
real_results = sio.loadmat('real_data/cnn_real_my_first_cnn.mat')
print(f"Real data predictions shape: {real_results['predictions'].shape}")
```

## Common Issues

### Issue: `ImportError: No module named 'mne'`

**Solution:**
```bash
pip install mne
```

### Issue: `CUDA out of memory`

**Solution:** Use CPU instead
```bash
python train_cnn.py --model_id my_cnn --device cpu
```

Or reduce batch size in `config_optimized.py`:
```python
'batch_size': 4,  # Reduce from 8 to 4
```

### Issue: Slow training on CPU

**Solution:** Use GPU
```bash
python train_cnn.py --model_id my_cnn --device cuda:0
```

Training time: ~1 minute per epoch on GPU vs 15+ minutes on CPU

## Next Steps

- **Detailed guide:** Read [README_CNN_SPATIAL.md](README_CNN_SPATIAL.md)
- **Compare models:** Train both Transformer and CNN:
  ```bash
  python train_optimized.py --model_id transformer
  python train_cnn.py --model_id cnn
  python eval_cnn_real.py --model_id cnn
  python eval_transformer_real.py --model_id transformer
  ```
- **Advanced training:** Customize hyperparameters in `config_optimized.py`
- **Troubleshooting:** Check [README_CNN_SPATIAL.md - Troubleshooting](README_CNN_SPATIAL.md#troubleshooting)

## File Summary

| File | Purpose | Command |
|------|---------|---------|
| `train_cnn.py` | Train CNN model | `python train_cnn.py --model_id {name}` |
| `eval_cnn_sim.py` | Test on simulated data | `python eval_cnn_sim.py --model_id {name}` |
| `eval_cnn_real.py` | Test on real data | `python eval_cnn_real.py --model_id {name}` |
| `network.py` | Contains CNN2DSpatialFilter | (imported internally) |
| `README_CNN_SPATIAL.md` | Full documentation | (read for details) |

## Key Concepts

**CNN2DSpatialFilter:**
- Converts 75 EEG electrodes → 64×64 2D grid using electrode topology
- Applies 2D convolutions to learn spatial electrode relationships
- Outputs 500 learned features per time step

**Electrode Topology:**
- Loads from `anatomy/electrode_75.mat` (EEGLAB format)
- Uses MNE for coordinate transformation
- Projects 3D positions to 2D grid via stereographic projection

**Data Format:**
- Input: (batch, 500 time steps, 75 electrodes)
- Output: (batch, 500 time steps, 994 brain sources)

## Timing

| Step | Time |
|------|------|
| Install MNE | <1 min |
| Train model (1 epoch) | 1 min (GPU) / 15 min (CPU) |
| Full training (30 epochs) | 30 min (GPU) / 8 hours (CPU) |
| Evaluation | <1 min |
| **Total (full pipeline)** | ~40 min (GPU) |

---

That's it! You now have a trained CNN spatial filter model. 🎉

For detailed information, see [README_CNN_SPATIAL.md](README_CNN_SPATIAL.md)
