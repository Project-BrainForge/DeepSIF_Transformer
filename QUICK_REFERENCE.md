# Quick Start - CNN Spatial Filter

## 30-Second Setup

```bash
cd /Users/deshitha.gallage/Documents/Personal/DeepSIF_Transformer
source venv/bin/activate
```

## One-Command Training

```bash
# Train for 2 epochs
python train_cnn.py --model_id my_model --epochs 2

# Train for 50 epochs with custom hyperparameters
python train_cnn.py --model_id my_model --epochs 50 --batch_size 16 --lr 0.0001
```

## Evaluation

```bash
# Evaluate on simulated data (10 test samples)
python eval_cnn_sim.py --model_id my_model

# Evaluate on real EEG data
python eval_cnn_real.py --model_id my_model
```

## Output Locations

```
model_result/my_model_cnn_spatial/
├── checkpoint_epoch_0.pth        # After first epoch
├── checkpoint_epoch_1.pth        # After second epoch
├── model_best.pth                # Best validation checkpoint
├── training_cnn_my_model.log     # Training log
├── training_history.mat          # Loss history
└── cnn_sim_results.mat           # Evaluation results
```

## Monitor Training

```bash
# Watch the training log in real-time
tail -f model_result/my_model_cnn_spatial/training_cnn_my_model.log
```

## Key Fixes Applied

| Issue | Fix | File |
|-------|-----|------|
| EEGLAB loading | Proper structured array indexing | network.py |
| Spherical conversion | Manual sin/cos implementation | network.py |
| Logging error | Pre-compute format values | train_cnn.py |

## Common Hyperparameters

```python
# Default training config
epochs = 40
batch_size = 8
learning_rate = 0.00001
optimizer = 'Adam'
scheduler = 'ReduceLROnPlateau'

# Good starting points
epochs = 50-100 (for convergence)
batch_size = 8-16 (depending on RAM)
learning_rate = 1e-4 to 1e-5 (conservative)
```

## Full Pipeline (End-to-End)

```bash
# 1. Train
python train_cnn.py --model_id pipeline_demo --epochs 5

# 2. Evaluate on simulated data
python eval_cnn_sim.py --model_id pipeline_demo

# 3. Evaluate on real data
python eval_cnn_real.py --model_id pipeline_demo

# Check results
ls model_result/pipeline_demo_cnn_spatial/
```

## Status Check

```bash
# Verify model checkpoint exists
ls -lh model_result/my_model_cnn_spatial/model_best.pth

# Check logs
cat model_result/my_model_cnn_spatial/training_cnn_my_model.log | tail -50

# View training history
python -c "import scipy.io as sio; h = sio.loadmat('model_result/my_model_cnn_spatial/training_history.mat'); print(h.keys())"
```

## Troubleshooting

**No checkpoints found?**
```bash
# Check model_id matches
python eval_cnn_sim.py --model_id my_model
# vs
# Check if directory exists
ls model_result/ | grep cnn
```

**GPU not detected?**
```bash
# Force CPU
python train_cnn.py --model_id my_model --device cpu

# Or let it auto-detect
python train_cnn.py --model_id my_model  # Uses CUDA if available
```

**Training too slow?**
```bash
# Increase batch size
python train_cnn.py --model_id my_model --batch_size 32

# Run on GPU (if available)
# Device will auto-detect CUDA
```

## Performance Metrics

| Metric | Value |
|--------|-------|
| Training time/epoch | 3-4 minutes (CPU) |
| Parameters | ~4M |
| Input shape | (batch, 500, 75) |
| Output shape | (batch, 500, 994) |
| Memory/batch=8 | ~2.5GB |

