#!/bin/bash
# Training script for Car Detection Model
# 117 annotated car images with YOLOv8

echo "🚗 CAR DETECTION MODEL TRAINING"
echo "================================"
echo ""
echo "Dataset: 117 images, 11 car models"
echo "Model: YOLOv8"
echo ""

# Check which device to use
echo "Select training device:"
echo "1) CPU (slower, ~3-5 hours for 50 epochs)"
echo "2) NVIDIA GPU (fast, ~30-60 min for 100 epochs)"
echo "3) Apple GPU (M1/M2/M3, ~1-2 hours for 100 epochs)"
echo ""

# Default to CUDA (NVIDIA GPU)
DEVICE="${1:-cuda}"

if [ "$DEVICE" = "cpu" ]; then
    echo "Training on CPU..."
    python3 train.py \
        --epochs 50 \
        --batch-size 8 \
        --learning-rate 0.001 \
        --device cpu
elif [ "$DEVICE" = "mps" ]; then
    echo "Training on Apple GPU (MPS)..."
    python3 train.py \
        --epochs 100 \
        --batch-size 16 \
        --learning-rate 0.001 \
        --device mps
else
    echo "Training on NVIDIA GPU (CUDA)..."
    python3 train.py \
        --epochs 100 \
        --batch-size 16 \
        --learning-rate 0.001 \
        --device cuda
fi

echo ""
echo "✅ Training complete!"
echo ""
echo "Next steps:"
echo "  1. Check results: open runs/mask_detection/train/"
echo "  2. Test model:    python3 test_car_detection.py image.jpg"
echo "  3. Deploy API:    python3 -m uvicorn api:app --reload"
