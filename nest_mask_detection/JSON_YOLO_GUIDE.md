# JSON ↔ YOLO Annotation Format Converter

Convert between **human-friendly JSON annotations** and **YOLO training format**.

---

## 📊 Format Comparison

### JSON Format (Human-readable)
```json
{
  "image": {
    "path": "car_photo.jpg",
    "width": 1920,
    "height": 1440
  },
  "annotations": [
    {
      "id": 0,
      "car_model": "BMW",
      "bbox": {
        "x": 100,
        "y": 150,
        "width": 200,
        "height": 180
      },
      "bbox_normalized": {
        "x_center": 0.1563,
        "y_center": 0.1667,
        "width": 0.1042,
        "height": 0.125
      }
    }
  ]
}
```

### YOLO Format (Training)
```
2 0.1563 0.1667 0.1042 0.125
0 0.4219 0.1111 0.0938 0.1111
```

---

## 🔄 Workflow

```
Your Annotations (Multiple Formats)
           ↓
JSON Converter (annotation_converter.py)
           ↓
    YOLO Format (.txt files)
           ↓
    YOLOv8 Training
           ↓
Trained Car Detection Model
```

---

## 🚀 Quick Start

### 1. View Examples
```bash
python3 annotation_examples.py workflow
```

### 2. Convert JSON → YOLO
```bash
python3 annotation_converter.py json2yolo \
  --json-dir annotations/ \
  --output-dir SampleImages/labels/
```

### 3. Process & Split Dataset
```bash
python3 annotation_converter.py process \
  --json-dir annotations/ \
  --output-dir SampleImages/
```

### 4. Train Model
```bash
python3 train.py --data-dir SampleImages/
```

---

## 📝 Detailed Usage

### Option A: Start with JSON Annotations

**Step 1: Collect annotations in JSON format**

File: `annotations/car_photo_001.json`
```json
{
  "image": {
    "path": "car_photo_001.jpg",
    "width": 1920,
    "height": 1440
  },
  "annotations": [
    {
      "id": 0,
      "car_model": "BMW",
      "bbox": {"x": 100, "y": 150, "width": 200, "height": 180},
      "bbox_normalized": {
        "x_center": 0.156,
        "y_center": 0.167,
        "width": 0.104,
        "height": 0.125
      }
    }
  ]
}
```

**Step 2: Convert all JSON → YOLO**
```bash
python3 annotation_converter.py json2yolo \
  --json-dir annotations/ \
  --output-dir SampleImages/train/labels/
```

Creates: `SampleImages/train/labels/car_photo_001.txt`
```
2 0.156 0.167 0.104 0.125
```

**Step 3: Copy images to same directory**
```bash
cp annotations/*.jpg SampleImages/train/images/
```

**Step 4: Train**
```bash
python3 train.py --data-dir SampleImages/
```

---

### Option B: Organize Mixed Annotations

**Step 1: Place images + JSON labels in one directory**
```
annotations/
├── car_001.jpg
├── car_001.json
├── car_002.jpg
├── car_002.json
└── ...
```

**Step 2: Process entire dataset**
```bash
python3 annotation_converter.py process \
  --json-dir annotations/ \
  --output-dir SampleImages/
```

Automatically:
- Reads JSON annotations
- Converts to YOLO format
- Splits into train/val/test (70/15/15)
- Organizes into: `SampleImages/train/labels/`, etc.

**Step 3: Copy images to correct splits**
```bash
# Script to move images based on train/val/test splits
python3 << 'EOF'
import shutil
from pathlib import Path

annotations_dir = Path("annotations")
sample_dir = Path("SampleImages")

for json_file in annotations_dir.glob("*.json"):
    image_file = annotations_dir / f"{json_file.stem}.jpg"
    if not image_file.exists():
        continue

    # Copy to train (you can customize logic for val/test)
    split = "train"  # Default to train
    dest = sample_dir / split / "images" / image_file.name
    shutil.copy2(image_file, dest)
    print(f"Copied {image_file.name} to {split}/")
EOF
```

**Step 4: Train**
```bash
python3 train.py --data-dir SampleImages/
```

---

### Option C: Convert YOLO Back to JSON

Useful for viewing/editing YOLO labels:

```python
from annotation_converter import YOLOAnnotation

# Load YOLO file
yolo_content = open("sample.txt").read()

# Convert to JSON
json_ann = YOLOAnnotation.yolo_to_json(
    yolo_content,
    image_path="sample.jpg",
    image_width=1920,
    image_height=1440
)

# Save as JSON
import json
with open("sample.json", "w") as f:
    json.dump(json_ann, f, indent=2)
```

---

## 🎯 Car Model Classes

JSON uses **car model names**, automatically converted to class IDs:

```
Toyota        → 0     Tesla       → 8
Honda         → 1     Nissan      → 9
BMW           → 2     Hyundai     → 10
Mercedes-Benz → 3     Kia         → 11
Audi          → 4     Mazda       → 12
Volkswagen    → 5     Subaru      → 13
Ford          → 6     Lexus       → 14
Chevrolet     → 7
```

---

## 💻 Python API

### Create Annotations Programmatically

```python
from annotation_converter import JSONAnnotation, YOLOAnnotation
from pathlib import Path

# 1. Create JSON annotation
detections = [
    {"model": "BMW", "x": 100, "y": 150, "width": 200, "height": 180},
    {"model": "Toyota", "x": 350, "y": 100, "width": 180, "height": 160},
]

annotation = JSONAnnotation.create_annotation(
    image_path="car.jpg",
    image_width=1920,
    image_height=1440,
    detections=detections
)

# 2. Save JSON
JSONAnnotation.save_json(annotation, Path("car.json"))

# 3. Convert to YOLO
yolo_content = YOLOAnnotation.json_to_yolo(annotation)

# 4. Save YOLO
YOLOAnnotation.save_yolo(yolo_content, Path("car.txt"))
```

### Batch Convert Directory

```python
from annotation_converter import json_directory_to_yolo

json_directory_to_yolo(
    json_dir=Path("annotations/"),
    yolo_dir=Path("SampleImages/train/labels/")
)
```

### Process & Split Automatically

```python
from annotation_converter import process_json_dataset

process_json_dataset(
    json_dir=Path("annotations/"),
    output_dir=Path("SampleImages/"),
    train_split=0.7,
    val_split=0.15
)
```

---

## 📋 Bbox Format Explained

### Pixel Coordinates (JSON)
```
"bbox": {
  "x": 100,      # Top-left X coordinate
  "y": 150,      # Top-left Y coordinate
  "width": 200,  # Box width
  "height": 180  # Box height
}
```

### Normalized Coordinates (JSON & YOLO)
```
"bbox_normalized": {
  "x_center": 0.1563,  # Center X (0-1)
  "y_center": 0.1667,  # Center Y (0-1)
  "width": 0.1042,     # Width ratio (0-1)
  "height": 0.125      # Height ratio (0-1)
}

# YOLO .txt:
2 0.1563 0.1667 0.1042 0.125
└ class_id
   └ x_center  └ y_center  └ width  └ height
```

### Conversion

**Pixel → Normalized:**
```
x_center = (x + width/2) / image_width
y_center = (y + height/2) / image_height
width_norm = width / image_width
height_norm = height / image_height
```

**Normalized → Pixel:**
```
x = (x_center - width/2) * image_width
y = (y_center - height/2) * image_height
width = width_norm * image_width
height = height_norm * image_height
```

---

## 🔍 Examples

### Run Examples
```bash
# Show JSON structure
python3 annotation_examples.py json

# Show workflow
python3 annotation_examples.py workflow

# Run all examples
python3 annotation_examples.py all
```

### Example Output
```
JSON Annotation Format:
{
  "image": {
    "path": "sample_image.jpg",
    "width": 640,
    "height": 480
  },
  "annotations": [
    {
      "id": 0,
      "car_model": "BMW",
      "bbox": {
        "x": 100,
        "y": 150,
        "width": 200,
        "height": 180
      },
      "bbox_normalized": {
        "x_center": 0.234375,
        "y_center": 0.3125,
        "width": 0.3125,
        "height": 0.375
      }
    }
  ]
}

YOLO Format (.txt file content):
2 0.234375 0.3125 0.3125 0.375
0 0.617188 0.302083 0.351562 0.364583
```

---

## ✅ Complete Workflow Example

```bash
# 1. Create annotations (using Roboflow or LabelImg)
# Place in: annotations/car_001.json, annotations/car_001.jpg, etc.

# 2. View structure
python3 annotation_examples.py json

# 3. Process dataset
python3 annotation_converter.py process \
  --json-dir annotations/ \
  --output-dir SampleImages/

# 4. Verify YOLO labels created
ls SampleImages/train/labels/
# car_001.txt   car_002.txt   car_003.txt ...

# 5. Move images to correct locations
cp annotations/*.jpg SampleImages/train/images/
# (split manually or use custom script)

# 6. Train model
python3 train.py --epochs 100 --batch-size 16 --device cuda

# 7. Test
python3 test_car_detection.py SampleImages/test_image.jpg

# Done! 🎉
```

---

## 🎯 Integration with Annotation Tools

### Roboflow Export
1. Create project on https://roboflow.com
2. Set format: **YOLO v8**
3. Export → Download ZIP
4. Extract and convert to intermediate JSON:
   ```bash
   python3 << 'EOF'
   import json
   from pathlib import Path

   # Read YOLO files and convert to JSON
   # ...custom script based on your filenames
   EOF
   ```

### LabelImg Export
1. Use LabelImg to annotate
2. Export format: **YOLO**
3. Creates .txt files in YOLO format
4. Convert if needed to JSON for viewing

---

## 🆘 Troubleshooting

### "Unknown car model"
```
⚠️  Car model must be from the 15 classes:
   Toyota, Honda, BMW, Mercedes-Benz, Audi,
   Volkswagen, Ford, Chevrolet, Tesla, Nissan,
   Hyundai, Kia, Mazda, Subaru, Lexus
```

**Fix:** Edit annotation JSON, use correct car model name (case-sensitive)

### "Mismatch in dimensions"
```
❌ Error: Image dimensions don't match
```

**Fix:** Ensure `image_width` and `image_height` in JSON match actual image

### YOLO file has wrong values
```
❌ Normalized coordinates should be 0-1
```

**Fix:** Check bbox_normalized values in JSON are between 0 and 1

---

## 📚 Reference

| Task | Command |
|------|---------|
| View examples | `python3 annotation_examples.py all` |
| Convert JSON→YOLO | `python3 annotation_converter.py json2yolo --json-dir X --output-dir Y` |
| Process dataset | `python3 annotation_converter.py process --json-dir X --output-dir Y` |
| Create annotation | See Python API section |
| Convert back to JSON | See Python API section |

---

## 🌟 Summary

- **JSON**: Safe, human-readable format for storage/editing
- **YOLO**: Training format for YOLOv8
- **Converter**: Seamless conversion both ways
- **Workflow**: JSON → YOLO → Training → Done!

Ready to annotate! 🚗📸

