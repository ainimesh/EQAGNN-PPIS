#!/bin/bash
# Run multiple experiments

echo "Starting E(Q)AGNN-PPIS experiments..."

# Experiment 1: Test_60 validation
echo "Experiment 1: Training with Test_60 validation"
python train.py --epochs 50 --val_dataset test_60 --exp_name exp1_test60

# Experiment 2: Test_315 validation
echo "Experiment 2: Training with Test_315 validation"
python train.py --epochs 50 --val_dataset test_315 --exp_name exp2_test315

# Experiment 3: Different learning rates
echo "Experiment 3: Learning rate comparison"
for lr in 0.001 0.0005 0.0001; do
    echo "Training with learning rate: $lr"
    python train.py --epochs 30 --lr $lr --exp_name exp3_lr_${lr}
done

echo "All experiments completed!"