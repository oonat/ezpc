#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Set your target data directory
DATA_ROOT="./"

echo "========================================="
echo "Starting EZPC Automated Dataset Downloads"
echo "Target Directory: $DATA_ROOT"
echo "========================================="

# CIFAR-100
echo "Fetching CIFAR-100..."
python download_dataset.py --dataset CIFAR-100 --dataset_root "$DATA_ROOT"

# CUB-200-2011
echo "Fetching CUB-200-2011..."
python download_dataset.py --dataset CUB-200-2011 --dataset_root "$DATA_ROOT"

# Places365
echo "Fetching Places365..."
python download_dataset.py --dataset Places365 --dataset_root "$DATA_ROOT"

# ImageNet and ImageNet-100
echo "Fetching ImageNet Information..."
python download_dataset.py --dataset ImageNet --dataset_root "$DATA_ROOT"

echo "========================================="
echo "All automated download scripts completed!"
echo "Please complete the manual ImageNet steps if needed."
echo "========================================="