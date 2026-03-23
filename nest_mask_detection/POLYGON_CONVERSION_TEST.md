# 🚗 Polygon Annotation to YOLO - Complete Test Workflow

## Your Sample File Test Results

### ✅ Success! Conversion Complete

**Input:** `Screenshot 2026-03-22 at 5.json` (Polygon format)
**Output:** `Screenshot_converted.txt` (YOLO format)

---

## 📊 What Was Converted

### Original Annotation (Polygon JSON)
```json
[
  {
    "content": [
      {"x": 68, "y": 339},
      {"x": 65, "y": 317},
      ... (17 points total forming a polygon)
    ],
    "rectMask": {
      "xMin": 65,
      "yMin": 268,
      "width": 170,
      "height": 90
    },
    "labels": {
      "labelName": "Honda",
      "labelColor": "#ff0000"
    },
    "contentType": "polygon"
  }
]
```

### Converted Output (YOLO Format)
```
1 0.234375 0.652083 0.265625 0.187500
```

**Meaning:**
- `1` = Class ID (Honda)
- `0.234375` = x_center (normalized)
- `0.652083` = y_center (normalized)
- `0.265625` = width (normalized)
- `0.187500` = height (normalized)

---

## 🔄 Conversion Formula

```
From Polygon Annotation:
├─ xMin = 65, yMin = 268
├─ width = 170, height = 90
└─ image_size = 640×480

To YOLO Format:
├─ x_center = (65 + 170/2) / 640 = 0.234375
├─ y_center = (268 + 90/2) / 480 = 0.652083
├─ width_norm = 170 / 640 = 0.265625
└─ height_norm = 90 / 480 = 0.187500
```

---

## 🎯 Next Steps for Training

### Step 1: Have You Got the Image File?

You need the corresponding **image file**. Let me check what image this annotation came from:

```bash
# The annotation filename suggests it's a screenshot
# Look for the image file:
ls SampleImages/train/images/

# Expected files:
# - Screenshot_2026-03-22_at_5.jpg  OR
# - Screenshot.jpg  OR
# - Similar name
```

**Do you have the actual car image?** If yes, please provide:
- Image filename
- Image dimensions (width × height)

### Step 2: Verify Image Dimensions

The conversion used **640×480** as assumed dimensions. If your image is different, we need to reconvert:

```bash
# Reconvert with correct dimensions
python3 polygon_converter.py \
  "SampleImages/train/labels/Screenshot 2026-03-22 at 5.json" \
  --width 1920 \
  --height 1080 \
  --output SampleImages/train/labels/Screenshot_converted.txt
```

### Step 3: Organize Files

Once you have the image:

```
SampleImages/
└── train/
    ├── images/
    │   └── screenshot.jpg      ← Your car image
    └── labels/
        └── screenshot.txt       ← YOLO format label
```

### Step 4: Split Dataset

```bash
python3 setup_car_dataset.py \
  --source SampleImages/
```

### Step 5: Train Model

```bash
# First, install dependencies
python3 -m pip install torch torchvision ultralytics

# Train
python3 train.py \
  --epochs 50 \
  --batch-size 8 \
  --device cpu
```

---

## 🔧 Using the Polygon Converter

### Single File Conversion

```bash
python3 polygon_converter.py <json_file> --width <W> --height <H> [--output <txt_file>]

# Example:
python3 polygon_converter.py \
  SampleImages/train/labels/Screenshot\ 2026-03-22\ at\ 5.json \
  --width 640 \
  --height 480 \
  --output SampleImages/train/labels/screenshot.txt
```

### Batch Conversion (Multiple Files)

```python
from polygon_converter import PolygonAnnotationConverter
from pathlib import Path

# Define image dimensions for each file
image_dims = {
    "screenshot_001": (640, 480),
    "screenshot_002": (1920, 1080),
    "screenshot_003": (1280, 720),
}

# Convert all
PolygonAnnotationConverter.batch_convert_directory(
    json_dir=Path("SampleImages/train/labels/"),
    output_dir=Path("SampleImages/train/labels/"),
    image_dimensions=image_dims
)
```

---

## 📋 Supported Annotation Formats

The `polygon_converter.py` supports:

| Format | Source | Example |
|--------|--------|---------|
| Polygon | CVAT, Roboflow, Custom | Multiple (x,y) points + rect mask |
| Rectangle | CVAT, Roboflow | Single rect mask |
| Polygon w/ Label | LabelImg Export | JSON with "labelName" |

---

## 🎨 Car Model Classes

Your annotation has **Honda** (class_id = 1).

All 15 supported models:

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

## ⚠️ Important Notes

### Image Dimensions Matter!
The conversion accuracy depends on correct image dimensions. If you use wrong dimensions:
- Normalized coordinates will be incorrect
- Model training will perform poorly

### Get Real Image Dimensions

**Option 1: Using Python**
```python
from PIL import Image
img = Image.open("screenshot.jpg")
width, height = img.size
print(f"Image size: {width}×{height}")
```

**Option 2: Using CV2**
```python
import cv2
img = cv2.imread("screenshot.jpg")
height, width = img.shape[:2]
print(f"Image size: {width}×{height}")
```

**Option 3: File Properties**
```bash
# macOS
mdls screenshot.jpg | grep Dimensions

# Linux
file screenshot.jpg
identify screenshot.jpg  # ImageMagick

# Windows
python -c "from PIL import Image; img=Image.open('screenshot.jpg'); print(img.size)"
```

---

## 🧪 Troubleshooting

### "Unknown car model"
```
⚠️  Error: Unknown car model: "Honda"
```
**Solution:** Car model name must match exactly (case-sensitive):
- Correct: "Honda"
- Wrong: "honda", "HONDA", "Honda Car"

### "No dimensions provided"
```
❌ Error: No dimensions for screenshot, skipping
```
**Solution:** Provide image dimensions:
```bash
python3 polygon_converter.py file.json --width 1920 --height 1080
```

### Coordinates out of range
```
⚠️  Warning: Normalized coordinate > 1.0
```
**Cause:** Image dimensions too small
**Solution:** Use actual image dimensions, not assumed ones

---

## 📚 Complete Workflow Checklist

- [ ] Have polygon annotation JSON files
- [ ] Know image dimensions (height × width)
- [ ] Have corresponding car images
- [ ] Run polygon converter
- [ ] Verify YOLO files created (*.txt)
- [ ] Organize into train/val/test splits
- [ ] Install ML dependencies (torch, ultralytics)
- [ ] Train model with `python3 train.py`
- [ ] Test on images with `python3 test_car_detection.py`
- [ ] Deploy or integrate with Nest cameras

---

## 🚀 Quick Command Reference

```bash
# Convert single file with correct dimensions
python3 polygon_converter.py \
  SampleImages/train/labels/screenshot.json \
  --width 1920 \
  --height 1080 \
  --output SampleImages/train/labels/screenshot.txt

# Setup dataset
python3 setup_car_dataset.py

# Train model
python3 train.py --epochs 100 --batch-size 16 --device cuda

# Test on image
python3 test_car_detection.py path/to/test_image.jpg

# Get statistics
curl http://localhost:8000/stats
```

---

## ❓ Next Question

**What are the dimensions of the image associated with this annotation?**

Once you provide the dimensions (or the image file), I can:
1. Reconvert with correct dimensions
2. Create the complete training dataset
3. Show you the exact training command

Just tell me! 📸

