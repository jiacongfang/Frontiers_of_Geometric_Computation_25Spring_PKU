#!/bin/bash

# Training script for MLP with Fourier features
python train.py --data_root ./data --save_path ./checkpoint_fourier --lambda_gradient 0.5 \
    --lambda_eikonal 0  --lambda_sdf 2.0 --mix_dataset --use_fourier --learning_rate 2e-4 \
    --fourier_mapping_size 64 --fourier_scale 5