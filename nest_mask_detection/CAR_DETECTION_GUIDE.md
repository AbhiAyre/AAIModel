# 🚗 Car Model Detection - Complete Training Guide

## Overview

Transform your car images into a production-ready detection model:
- **Input**: Car images + YOLO labels
- **Process**: Organize → Train → Test
- **Output**: Trained YOLOv8 model that detects 15 car models

---

## 📊 Workflow Steps

### Step 1: Gather & Organize Images (1-2 hours)

**Create folder structure:**
```bash
cd nest_mask_detection/

# Already created:
# SampleImages/
# ├── train/
# │   ├── images/
# │   └── labels/
# ├── val/
# │   ├── images/
# │   └── labels/
# └── test/
#     ├── images/
#     └── labels/
```

**Add your car images:**

Option A - Manual upload:
1. Download car images from internet (Google Images, Unsplash, etc.)
2. Save to `SampleImages/` directory
3. At least 100-500 images total (5-30 per car model)

Option B - Collect from camera/dataset:
- Use your own photos
- Use public datasets (Kaggle, COCO)

**Required: 15 Car Model Classes**
```
0  → Toyota          8  → Tesla
1  → Honda           9  → Nissan
2  → BMW             10 → Hyundai
3  → Mercedes-Benz   11 → Kia
4  → Audi            12 → Mazda
5  → Volkswagen      13 → Subaru
6  → Ford            14 → Lexus
7  → Chevrolet
```

---

### Step 2: Create YOLO Labels (2-3 hours for 100-500 images)

**Recommended Tool: Roboflow** (easiest, free tier available)

1. Go to https://roboflow.com
2. Create a new project
3. Set format: **YOLO v8**
4. Upload all your images
5. Draw bounding boxes for each car
6. Assign correct class ID (0-14)
7. Export as YOLO format

**Expected output:**
```
SampleImages/
├── image_1.jpg
├── image_1.txt       ← Label file
├── image_2.jpg
├── image_2.txt
└── ...
```

**Label file format (one car per line):**
```
class_id x_center y_center width height

Example:
2 0.45 0.52 0.35 0.48    (BMW centered at 45%, 52%, 35% width, 48% height)
```

---

### Step 3: Split into Train/Val/Test

**Run the setup script:**
```bash
python3 setup_car_dataset.py --print-info
```

**Output:**
- 70% → `train/` (training)
- 15% → `val/` (validation)
- 15% → `test/` (testing)

Creates `dataset.yaml` automatically.

---

### Step 4: Install ML Dependencies

```bash
# PyTorch (Deep Learning Framework)
python3 -m pip install torch torchvision

# YOLOv8 (Object Detection Model)
python3 -m pip install ultralytics

# Verify
python3 -c "import torch; from ultralytics import YOLO; print('✓ Ready to train')"
```

---

### Step 5: Train the Model

**Basic training (CPU):**
```bash
python3 train.py \
  --epochs 50 \
  --batch-size 8 \
  --device cpu
```

**Faster training (GPU - NVIDIA only):**
```bash
python3 train.py \
  --epochs 100 \
  --batch-size 16 \
  --device cuda
```

**Custom parameters:**
```bash
python3 train.py \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --model yolov8n \
  --device cuda \
  --save-dir models/car_detection_v1
```

**Training output:**
- Logs in `runs/mask_detection/train/`
- Model saved to `models/yolov8n_trained.pt`
- Training typically takes:
  - CPU: 2-4 hours (50 epochs)
  - GPU: 15-30 min (50 epochs)

---

### Step 6: Test Your Model

**Test on single image:**
```bash
python3 test_car_detection.py SampleImages/test_image.jpg

# Output:
# ✓ Toyota (0.92 confidence) at [100, 150, 250, 300]
# ✓ BMW (0.87 confidence) at [280, 50, 450, 200]
# ✓ Annotated image saved: SampleImages/test_image_annotated.jpg
```

**Test on all test images:**
```bash
for img in SampleImages/test/images/*.jpg; do
  python3 test_car_detection.py "$img"
done
```

---

### Step 7: Evaluate Performance

Check training metrics:
```bash
# View training logs
open runs/mask_detection/train/results.csv

# Metrics:
# - mAP (mean Average Precision) - overall accuracy
# - Precision - of detections, how many correct
# - Recall - what % of cars found
# - F1 Score - balance of precision/recall
```

**Good metrics:**
- mAP > 0.7 (70%)
- Precision > 0.8
- Recall > 0.75

---

### Step 8: Connect to Google Nest (Later)

Once you're happy with accuracy:

```bash
# Start Pub/Sub consumer
python3 consumer.py

# It will:
# 1. Listen for Nest camera events
# 2. Download event images
# 3. Run car detection
# 4. Store results in database
```

---

## 📋 Complete Command Reference

### Setup
```bash
# Show setup instructions
python3 setup_car_dataset.py --print-info

# Organize images (interactive)
python3 setup_car_dataset.py --source SampleImages
```

### Training
```bash
# Basic training
python3 train.py

# Advanced training
python3 train.py \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --device cuda \
  --model yolov8m

# Setup only (no training)
python3 train.py --setup-only
```

### Testing
```bash
# Test single image
python3 test_car_detection.py image.jpg

# Test with custom model
python3 test_car_detection.py image.jpg --model models/custom_model.pt
```

### API
```bash
# Start API server
python3 -m uvicorn api:app --reload

# Make prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@test_image.jpg"

# View stats
curl http://localhost:8000/stats

# View docs (in browser)
# http://localhost:8000/docs
```

---

## 🎯 Expected Accuracy Progression

| Stage | mAP | Images | Time |
|-------|-----|--------|------|
| Day 1 - Setup | - | 100-200 | 2-3h |
| Day 2 - First training | 0.30-0.45 | 200-300 | 2-3h |
| Day 3 - Refine | 0.50-0.65 | 300-400 | 2-3h |
| Day 4-5 - Production | 0.70-0.85+ | 400-500 | 2-3h |

---

## 💡 Tips for Better Results

### Data Collection
✅ Collect diverse images:
- Different car brands/models
- Various angles (front, back, side)
- Different lighting (day, night, cloudy)
- Different distances (close, far, medium)
- Different weather

❌ Avoid:
- Low quality/blurry images
- Images where car is too small/occluded
- Only one angle per model
- Poor lighting

### Training
✅ Do:
- Start with small batch size (8-16) if OOM errors
- Use data augmentation (YOLOv8 does this automatically)
- Train longer for better accuracy (100+ epochs)
- Use GPU if available (10x faster)

❌ Don't:
- Train on too few images
- Use very old model (yolov8n best for speed, yolov8x for accuracy)
- Stop training too early

---

## ⚠️ Troubleshooting

### "CUDA out of memory"
```bash
# Reduce batch size
python3 train.py --batch-size 4 --device cuda
```

### "No images found"
```bash
# Verify images in SampleImages/
ls -la SampleImages/images/*.jpg

# Setup directory structure first
python3 setup_car_dataset.py
```

### "Low accuracy (mAP < 0.5)"
```bash
# Solutions:
# 1. Add more images (200+ minimum)
# 2. Better labels (check annotation quality)
# 3. Train longer (100+ epochs)
# 4. Use larger model (yolov8m instead of yolov8n)

python3 train.py --epochs 150 --model yolov8m --batch-size 16 --device cuda
```

### Model not detecting anything
```bash
# Check:
# 1. Model is actual trained: ls -la models/
# 2. Test image has cars: open SampleImages/test/images/*.jpg
# 3. Confidence threshold not too high (default 0.5 is good)
```

---

## 📦 Next: Google Nest Integration

Once your model achieves **mAP > 0.70**:

1. Update `.env` with Nest credentials
2. Run event processor: `python3 consumer.py`
3. Model will auto-detect cars from live Nest feed
4. Results stored in database

---

## 🚀 Summary

| Step | Time | Command |
|------|------|---------|
| 1. Gather images | 1-2h | Collect 100-500 car images |
| 2. Add labels | 2-3h | Use Roboflow or LabelImg |
| 3. Split data | 5min | `python3 setup_car_dataset.py` |
| 4. Train model | 30min-2h | `python3 train.py --epochs 100` |
| 5. Test accuracy | 10min | `python3 test_car_detection.py image.jpg` |
| 6. Connect Nest | 10min | Update `.env` and run `python3 consumer.py` |

**Total: 6-8 hours to production-ready car detection!**

