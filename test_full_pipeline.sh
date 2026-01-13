#!/bin/bash
set -e

echo "=========================================="
echo "FULL CNN SPATIAL FILTER PIPELINE TEST"
echo "=========================================="
echo ""

source venv/bin/activate

# Test 1: Training
echo "1. Testing Training Pipeline..."
python3 train_cnn.py --model_id pipeline_test --epochs 1 > /tmp/train_test.log 2>&1
TRAIN_RESULT=$?
if [ $TRAIN_RESULT -eq 0 ]; then
    echo "   ✓ Training completed successfully"
    grep "TRAINING COMPLETED" /tmp/train_test.log
else
    echo "   ✗ Training failed"
    exit 1
fi

echo ""

# Test 2: Simulated Data Evaluation
echo "2. Testing Simulated Data Evaluation..."
python3 eval_cnn_sim.py --model_id pipeline_test > /tmp/eval_sim_test.log 2>&1
SIM_RESULT=$?
if [ $SIM_RESULT -eq 0 ]; then
    echo "   ✓ Simulated data evaluation completed"
    grep "EVALUATION COMPLETED" /tmp/eval_sim_test.log
else
    echo "   ✗ Simulated evaluation failed"
    exit 1
fi

echo ""

# Test 3: Real Data Evaluation
echo "3. Testing Real Data Evaluation..."
python3 eval_cnn_real.py --model_id pipeline_test > /tmp/eval_real_test.log 2>&1
REAL_RESULT=$?
if [ $REAL_RESULT -eq 0 ]; then
    echo "   ✓ Real data evaluation completed"
    grep "COMPLETED" /tmp/eval_real_test.log
else
    echo "   ✗ Real data evaluation failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "✓ FULL PIPELINE TEST PASSED"
echo "=========================================="
echo ""
echo "Summary:"
echo "  • Training: 1 epoch completed"
echo "  • Simulated evaluation: 10 test samples"
echo "  • Real evaluation: 1 sample"
echo "  • Total time: ~10 minutes"
echo ""
echo "Model saved to: model_result/pipeline_test_cnn_spatial/"
