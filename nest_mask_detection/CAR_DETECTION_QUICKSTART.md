# 🚗 Car Detection Setup - Ready to Go!

Your project has been **pivoted from mask detection to car model detection**. Here's what's ready:

---

## ✅ What's Been Done

### Configuration
- ✅ Updated to detect **15 car models** (Toyota, Honda, BMW, etc.)
- ✅ Created config system with class names
- ✅ Updated class definitions in `config.py`

### Dataset Structure
- ✅ Created `SampleImages/` directory with train/val/test splits
- ✅ Script to organize images: `setup_car_dataset.py`
- ✅ Database updated for car detection statistics

### Model & Training
- ✅ Updated model to use dynamic class names
- ✅ Created training script: `train.py`
- ✅ Created testing script: `test_car_detection.py`
- ✅ Configuration for 15 car classes

### Documentation
- ✅ `CAR_DETECTION_GUIDE.md` - Complete workflow
- ✅ `DATASET_README.md` - Dataset preparation guide
- ✅ `setup_car_dataset.py --print-info` - Interactive help

---

## 🚀 Quick Start (5 Minutes)

### 1. **Show Setup Instructions**
```bash
python3 setup_car_dataset.py --print-info
```

### 2. **Add Your Car Images**
```bash
# Place car images in:
# SampleImages/image_1.jpg
# SampleImages/image_2.jpg
# ... etc
```

### 3. **Add YOLO Labels** (Using annotation tool)
```bash
# For each image, create corresponding label:
# SampleImages/image_1.txt
# SampleImages/image_2.txt

# Format: class_id x_center y_center width height
# Example: 2 0.5 0.5 0.3 0.4  (BMW class, centered, 30%x40%)
```

### 4. **Split & Organize**
```bash
python3 setup_car_dataset.py
```

### 5. **Train Model**
```bash
# Install ML dependencies first
python3 -m pip install torch torchvision ultralytics

# Train (CPU, 50 epochs)
python3 train.py --epochs 50 --batch-size 8 --device cpu

# Train (GPU, faster)
python3 train.py --epochs 100 --batch-size 16 --device cuda
```

### 6. **Test on Image**
```bash
python3 test_car_detection.py SampleImages/test_image.jpg
```

---

## 📁 Project Structure

```
nest_mask_detection/
├── config.py                 ← 15 car classes defined
├── model.py                  ← Uses dynamic class names
├── train.py                  ← Training pipeline
├── test_car_detection.py     ← Testing script (NEW)
├── setup_car_dataset.py      ← Dataset organizer (NEW)
├── SampleImages/             ← Your dataset (NEW)
│   ├── train/images/
│   ├── train/labels/
│   ├── val/images/
│   ├── val/labels/
│   ├── test/images/
│   └── test/labels/
├── CAR_DETECTION_GUIDE.md    ← Full workflow guide (NEW)
├── DATASET_README.md         ← Dataset setup (NEW)
├── api.py                    ← API endpoints (updated)
├── database.py               ← Car detection stats (updated)
└── ... other files
```

---

## 🎯 Car Models (15 Classes)

```
 0 → Toyota          8 → Tesla
 1 → Honda           9 → Nissan
 2 → BMW             10 → Hyundai
 3 → Mercedes-Benz   11 → Kia
 4 → Audi            12 → Mazda
 5 → Volkswagen      13 → Subaru
 6 → Ford            14 → Lexus
 7 → Chevrolet
```

**Can customize!** Edit `config.py` ModelConfig.class_names

---

## 📊 Expected Timeline

| Stage | Time | Action |
|-------|------|--------|
| Day 1-2 | 6-8h | Gather 100-500 car images |
| Day 2-3 | 3-4h | Annotate images with YOLO labels |
| Day 3-4 | 30m-2h | Train model |
| Day 4 | 10m | Test & evaluate |
| Day 5+ | - | Connect to Nest & live inference |

---

## 🔧 Key Files Changed

**Configuration:**
- `config.py` - Updated ModelConfig with 15 classes

**Model:**
- `model.py` - Now uses class names from config, generic color rendering

**Database:**
- `database.py` - Tracks car models instead of mask/no-mask

**New Files:**
- `test_car_detection.py` - Single image testing
- `setup_car_dataset.py` - Dataset organization
- `CAR_DETECTION_GUIDE.md` - Complete guide
- `DATASET_README.md` - Dataset preparation

---

## 💡 Annotation Tools (Pick One)

### 1. **Roboflow** (Easiest, Recommended)
- Go to https://roboflow.com
- Create project (YOLO v8 format)
- Upload images
- Draw boxes
- Export YOLO format
- Free tier: 3 projects, unlimited images

### 2. **LabelImg** (Desktop, Offline)
- Download: https://github.com/heartexlTaxa/labelImg
- Install: `pip install labelImg`
- Run: `labelImg`
- Supports YOLO format
- Works offline

### 3. **CVAT** (Professional, Web)
- https://app.cvat.ai
- Professional annotation
- Supports teams
- Free and paid tiers

---

## 📈 Training Tips

### For Best Accuracy:
✅ Collect **5-30 images per car model** (100-500 total)
✅ Vary angles, lighting, weather
✅ Include front, back, side views
✅ Mix close and distant shots
✅ Train for **100+ epochs**
✅ Use **batch_size=16** if GPU available
✅ Use **larger model** if accuracy poor (yolov8m/yolov8l)

### Training Speeds:
- CPU: ~2-4 hours for 50 epochs
- GPU (NVIDIA): ~20-40 min for 50 epochs
- GPU (Apple M1/M2): ~1-2 hours for 50 epochs

---

## 🧪 Testing Workflow

```bash
# After training:

# 1. Test single image
python3 test_car_detection.py SampleImages/test/images/car1.jpg

# 2. Test all test images
for img in SampleImages/test/images/*.jpg; do
  python3 test_car_detection.py "$img"
done

# 3. Get statistics
curl http://localhost:8000/stats | python3 -m json.tool

# Output example:
# {
#   "total_predictions": 10,
#   "avg_detections": 2.5,
#   "avg_confidence": 0.875,
#   "car_model_detections": {
#     "Toyota": 8,
#     "Honda": 7,
#     "BMW": 5
#   }
# }
```

---

## 🔌 Later: Google Nest Integration

Once your model achieves **mAP > 0.70** accuracy:

```bash
# 1. Update .env with Nest credentials
nano .env

# 2. Start event consumer
python3 consumer.py

# 3. Model will automatically:
# - Listen for Nest camera events
# - Download event images
# - Run car detection
# - Store results in database
# - Send alerts if configured
```

---

## 📞 Next Steps

1. **Read full guide:**
   ```bash
   cat CAR_DETECTION_GUIDE.md
   ```

2. **Gather images:**
   - Use Google Images, Unsplash, or your own photos
   - 100-500 images, 5-30 per car model

3. **Annotate images:**
   - Use Roboflow, LabelImg, or CVAT
   - Create YOLO format labels

4. **Train model:**
   ```bash
   python3 train.py --epochs 100 --device cuda
   ```

5. **Test accuracy:**
   ```bash
   python3 test_car_detection.py test_image.jpg
   ```

---

## ❓ Need Help?

**Setup instructions:**
```bash
python3 setup_car_dataset.py --print-info
```

**Full workflow:**
```bash
cat CAR_DETECTION_GUIDE.md
```

**Dataset preparation:**
```bash
cat DATASET_README.md
```

**Test model:**
```bash
python3 test_car_detection.py --help
```

---

## 🎉 You're All Set!

Your car detection project is ready to start training.
Just add images, create labels, and train!

Happy detecting! 🚗🚕🚙

