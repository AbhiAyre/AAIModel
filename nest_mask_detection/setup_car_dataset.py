"""Setup and organize car detection dataset."""
import os
import shutil
from pathlib import Path
import argparse
import logging
import yaml
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Car model classes (matches config.py)
CLASSES = {
    0: "Toyota", 1: "Honda", 2: "BMW", 3: "Mercedes-Benz", 4: "Audi",
    5: "Volkswagen", 6: "Ford", 7: "Chevrolet", 8: "Tesla", 9: "Nissan",
    10: "Hyundai", 11: "Kia", 12: "Mazda", 13: "Subaru", 14: "Lexus"
}


def create_dataset_structure(base_dir: Path):
    """Create train/val/test directory structure."""
    base_dir = Path(base_dir)

    dirs = [
        base_dir / "train" / "images",
        base_dir / "train" / "labels",
        base_dir / "val" / "images",
        base_dir / "val" / "labels",
        base_dir / "test" / "images",
        base_dir / "test" / "labels",
    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {d}")

    return dirs


def split_dataset(
    source_dir: Path,
    dest_dir: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42
):
    """
    Split images from source into train/val/test.

    Args:
        source_dir: Source directory with images and labels
        dest_dir: Destination base directory
        train_split: Train ratio
        val_split: Val ratio
        test_split: Test ratio
        seed: Random seed
    """
    np.random.seed(seed)

    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    # Find all images
    image_files = sorted([
        f for f in source_dir.glob("*")
        if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
    ])

    if not image_files:
        logger.warning(f"No images found in {source_dir}")
        return

    # Shuffle and split
    np.random.shuffle(image_files)
    n = len(image_files)
    train_count = int(n * train_split)
    val_count = int(n * val_split)

    train_imgs = image_files[:train_count]
    val_imgs = image_files[train_count:train_count + val_count]
    test_imgs = image_files[train_count + val_count:]

    logger.info(f"Total images: {n}")
    logger.info(f"  Train: {len(train_imgs)} ({train_split*100:.0f}%)")
    logger.info(f"  Val: {len(val_imgs)} ({val_split*100:.0f}%)")
    logger.info(f"  Test: {len(test_imgs)} ({test_split*100:.0f}%)")

    # Copy to destination
    splits = {
        "train": train_imgs,
        "val": val_imgs,
        "test": test_imgs
    }

    for split_name, img_list in splits.items():
        for img_path in img_list:
            # Copy image
            dest_img = dest_dir / split_name / "images" / img_path.name
            shutil.copy2(img_path, dest_img)

            # Copy label if exists
            label_path = img_path.parent / (img_path.stem + ".txt")
            if label_path.exists():
                dest_label = dest_dir / split_name / "labels" / label_path.name
                shutil.copy2(label_path, dest_label)

        logger.info(f"✓ Copied {len(img_list)} images to {split_name}/")


def organize_by_model(
    source_dir: Path,
    dest_dir: Path,
    model_mapping: dict
):
    """
    Organize images by car model folders.

    Args:
        source_dir: Source directory with model subfolders
        dest_dir: Destination base directory
        model_mapping: Dict mapping folder -> class_id
    """
    source_dir = Path(source_dir)
    dest_dir = Path(dest_dir)

    for model_folder, class_id in model_mapping.items():
        model_path = source_dir / model_folder

        if not model_path.exists():
            logger.warning(f"Model folder not found: {model_path}")
            continue

        logger.info(f"Processing {model_folder} (class_id={class_id})...")
        split_dataset(model_path, dest_dir)


def create_dataset_yaml(dest_dir: Path, yaml_path: Path = None):
    """Create dataset.yaml for YOLOv8 training."""
    if yaml_path is None:
        yaml_path = Path(dest_dir).parent / "dataset.yaml"

    yaml_content = {
        "path": str(Path(dest_dir).absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(CLASSES),
        "names": {i: name for i, name in CLASSES.items()}
    }

    with open(yaml_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    logger.info(f"✓ Created dataset.yaml at {yaml_path}")
    return yaml_path


def print_instructions():
    """Print setup instructions."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    CAR MODEL DETECTION - DATASET SETUP                       ║
╚══════════════════════════════════════════════════════════════════════════════╝

📁 Default Car Model Classes (15 total):
""")
    for class_id, name in CLASSES.items():
        print(f"   {class_id:2d} → {name}")

    print(f"""
📁 Expected Directory Structure:

    SampleImages/
    ├── train/
    │   ├── images/       (70% of images)
    │   └── labels/       (YOLO .txt files)
    ├── val/
    │   ├── images/       (15% of images)
    │   └── labels/
    └── test/
        ├── images/       (15% of images)
        └── labels/

🏷️  YOLO Label Format (.txt files):

    Each image → corresponding .txt file with same name
    Format: classId x_center y_center width height
    Example: 2 0.5 0.5 0.3 0.4

    (all coordinates normalized 0-1)

📋 Usage:

1️⃣  Copy your car images to SampleImages/
2️⃣  Create YOLO labels using annotation tool:
    - Roboflow: https://roboflow.com
    - LabelImg: https://github.com/heartexlabs/labelImg
    - CVAT: https://app.cvat.ai

3️⃣  Run this script to organize into train/val/test:
    python3 setup_car_dataset.py

4️⃣  Start training:
    python3 train.py --epochs 50 --batch-size 8 --device cpu

5️⃣  Test on image:
    python3 test_car_detection.py path/to/test_image.jpg

🎯 Tips:

✅ Minimum 50-100 images per car model
✅ Mix different angles and lighting
✅ Include various weather conditions
✅ Use images with clear car visibility
❌ Avoid blurry or unclear images
❌ Don't use same image multiple times
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Setup car detection dataset")
    parser.add_argument(
        "--source",
        type=str,
        default="SampleImages",
        help="Source directory with images",
    )
    parser.add_argument(
        "--print-info",
        action="store_true",
        help="Print setup instructions and exit",
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Training split ratio",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation split ratio",
    )

    args = parser.parse_args()

    if args.print_info:
        print_instructions()
        exit(0)

    # Create structure
    logger.info("Creating dataset structure...")
    source_dir = Path(args.source)

    if not source_dir.exists():
        logger.error(f"Source directory not found: {source_dir}")
        print_instructions()
        exit(1)

    # Setup train/val/test directories
    create_dataset_structure(source_dir)

    # Copy and split images
    logger.info("Splitting and organizing images...")
    split_dataset(
        source_dir,
        source_dir,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=1.0 - args.train_split - args.val_split
    )

    # Create dataset.yaml
    create_dataset_yaml(source_dir)

    logger.info("✓ Dataset setup complete!")
    logger.info(f"Next step: python3 train.py --data-dir {source_dir}")
