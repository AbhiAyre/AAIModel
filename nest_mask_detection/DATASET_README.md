# Car Model Detection - Dataset Setup Guide

## Car Models List (15 Classes)

Map your car images to these 15 models:

```
0  → Toyota
1  → Honda
2  → BMW
3  → Mercedes-Benz
4  → Audi
5  → Volkswagen
6  → Ford
7  → Chevrolet
8  → Tesla
9  → Nissan
10 → Hyundai
11 → Kia
12 → Mazda
13 → Subaru
14 → Lexus
```

**Can customize this list!** Edit the `CLASSES` variable in `car_dataset_setup.py`

---

## Directory Structure

```
SampleImages/
├── train/
│   ├── images/          # Training images (70%)
│   └── labels/          # YOLO format labels
├── val/
│   ├── images/          # Validation images (15%)
│   └── labels/
└── test/
    ├── images/          # Test images (15%)
    └── labels/
```

---

## How to Add Your Images

### Option 1: Manual Organization (If you have labeled images)

1. **Copy your car images** to SampleImages/
2. **Create subdirectories** for each car model:
```bash
mkdir -p SampleImages/{toyota,honda,bmw,mercedes}/{images,labels}
```

3. **Place images** in respective model folders
4. **Add YOLO labels** (see format below)

### Option 2: Automatic Setup (Using the script we'll create)

1. Place all raw images in `SampleImages/raw_images/`
2. Run the setup script
3. Script automatically creates train/val/test splits

---

## YOLO Label Format

Each image needs a `.txt` file with the same name in the `labels/` folder.

**Example:** `image_001.jpg` → `image_001.txt`

**Label format (one line per object):**
```
<class_id> <x_center> <y_center> <width> <height>
```

Where:
- `class_id`: 0-14 (from list above)
- `x_center`, `y_center`: Center of car (0-1, normalized)
- `width`, `height`: Car dimensions (0-1, normalized)

**Example label file (image_001.txt):**
```
2 0.5 0.5 0.3 0.4
```
(BMW at center of image, 30% width, 40% height)

---

## Annotation Tools

**Easy tools to create YOLO labels:**

1. **Roboflow** (Web, Free, Recommended)
   - Upload images
   - Draw boxes
   - Auto-export as YOLO format
   - https://roboflow.com

2. **LabelImg** (Desktop, Free)
   - Download: https://github.com/heartexlabs/labelImg
   - Supports YOLO format
   - Works offline

3. **CVAT** (Web, Free)
   - https://app.cvat.ai
   - Professional annotation tool

---

## Image Requirements

- **Format**: JPG, PNG
- **Size**: 640x640 recommended (will be resized anyway)
- **Per model**: At least 50-100 images per car model
- **Variety**: Different angles, lighting, backgrounds

---

## Next Steps

1. Gather car images (or download from internet)
2. Organize into SampleImages structure
3. Create YOLO labels using annotation tool
4. Run `python3 setup_car_dataset.py`
5. Train model: `python3 train.py --device cpu`
6. Test on images: `python3 test_car_detection.py`

---

## Tips for Better Dataset

✅ **Do:**
- Mix front, side, back views
- Include different weather conditions
- Vary distances and angles
- Include partial car images

❌ **Avoid:**
- All images from same angle
- Low quality/blurry images
- Images too small to see car details
- Only daytime images

