"""Convert between JSON and YOLO annotation formats."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Car model classes
CLASSES = {
    0: "Toyota",
  1: "Honda",
  2: "BMW",
  3: "Mercedes",
  4: "Audi",
  5: "Volkswagen",
  6: "Ford",
  7: "Chevy",
  8: "Tesla",
  9: "Nissan",
  10: "Hyundai",
  11: "Kia",
  12: "Pizza",
  13: "Truck",
  14: "Bus",
  15: "Police",
  16: "USPS",
  17: "FedEx",
  18: "LandRover",
  19: "Porsche",
  20: "Jeep",
  21: "Lexus",
  22: "Volvo"
}

# Reverse map for class name to ID
CLASS_ID_MAP = {v: k for k, v in CLASSES.items()}


# =============================================================================
# JSON FORMAT (Human-readable, for annotation)
# =============================================================================

class JSONAnnotation:
    """JSON annotation format for car detections."""

    @staticmethod
    def create_annotation(
        image_path: str,
        image_width: int,
        image_height: int,
        detections: List[Dict] = None
    ) -> Dict:
        """
        Create JSON annotation structure.

        Args:
            image_path: Path to image file
            image_width: Image width in pixels
            image_height: Image height in pixels
            detections: List of detection dicts with keys:
                {model: str, x: int, y: int, width: int, height: int}

        Returns:
            JSON-serializable annotation dict
        """
        if detections is None:
            detections = []

        return {
            "image": {
                "path": str(image_path),
                "width": image_width,
                "height": image_height,
            },
            "annotations": [
                {
                    "id": i,
                    "car_model": det["model"],
                    "bbox": {
                        "x": det["x"],
                        "y": det["y"],
                        "width": det["width"],
                        "height": det["height"],
                    },
                    "bbox_normalized": {
                        "x_center": (det["x"] + det["width"] / 2) / image_width,
                        "y_center": (det["y"] + det["height"] / 2) / image_height,
                        "width": det["width"] / image_width,
                        "height": det["height"] / image_height,
                    }
                }
                for i, det in enumerate(detections)
            ]
        }

    @staticmethod
    def save_json(annotation: Dict, output_path: Path):
        """Save annotation to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(annotation, f, indent=2)

        logger.info(f"Saved JSON annotation: {output_path}")

    @staticmethod
    def load_json(json_path: Path) -> Dict:
        """Load annotation from JSON file."""
        with open(json_path) as f:
            return json.load(f)


# =============================================================================
# YOLO FORMAT (For training)
# =============================================================================

class YOLOAnnotation:
    """YOLO annotation format for car detections."""

    @staticmethod
    def json_to_yolo(json_annotation: Dict) -> str:
        """
        Convert JSON annotation to YOLO format.

        Args:
            json_annotation: Annotation dict from JSON

        Returns:
            YOLO format string (ready for .txt file)
        """
        lines = []

        for ann in json_annotation["annotations"]:
            bbox_norm = ann["bbox_normalized"]
            class_name = ann["car_model"]
            class_id = CLASS_ID_MAP.get(class_name)

            if class_id is None:
                logger.warning(f"Unknown car model: {class_name}")
                continue

            # YOLO format: class_id x_center y_center width height
            yolo_line = (
                f"{class_id} "
                f"{bbox_norm['x_center']:.6f} "
                f"{bbox_norm['y_center']:.6f} "
                f"{bbox_norm['width']:.6f} "
                f"{bbox_norm['height']:.6f}"
            )
            lines.append(yolo_line)

        return "\n".join(lines)

    @staticmethod
    def yolo_to_json(
        yolo_content: str,
        image_path: str,
        image_width: int,
        image_height: int
    ) -> Dict:
        """
        Convert YOLO format to JSON annotation.

        Args:
            yolo_content: Content of YOLO .txt file
            image_path: Path to image
            image_width: Image width
            image_height: Image height

        Returns:
            JSON annotation dict
        """
        detections = []

        for line in yolo_content.strip().split("\n"):
            if not line.strip():
                continue

            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            x_center_norm = float(parts[1])
            y_center_norm = float(parts[2])
            width_norm = float(parts[3])
            height_norm = float(parts[4])

            # Convert normalized to pixel coordinates
            x_center = x_center_norm * image_width
            y_center = y_center_norm * image_height
            width = width_norm * image_width
            height = height_norm * image_height

            # Top-left corner
            x = int(x_center - width / 2)
            y = int(y_center - height / 2)

            car_model = CLASSES.get(class_id, "Unknown")

            detections.append({
                "model": car_model,
                "x": x,
                "y": y,
                "width": int(width),
                "height": int(height),
            })

        return JSONAnnotation.create_annotation(
            image_path, image_width, image_height, detections
        )

    @staticmethod
    def save_yolo(yolo_content: str, output_path: Path):
        """Save annotation to YOLO .txt file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            f.write(yolo_content)

        logger.info(f"Saved YOLO annotation: {output_path}")

    @staticmethod
    def load_yolo(yolo_path: Path) -> str:
        """Load annotation from YOLO .txt file."""
        with open(yolo_path) as f:
            return f.read()


# =============================================================================
# BATCH CONVERSION
# =============================================================================

def json_directory_to_yolo(
    json_dir: Path,
    yolo_dir: Path,
    image_dimensions: Optional[Dict[str, Tuple[int, int]]] = None
):
    """
    Convert all JSON annotations in directory to YOLO format.

    Args:
        json_dir: Directory with JSON annotation files
        yolo_dir: Output directory for YOLO .txt files
        image_dimensions: Dict mapping image_name -> (width, height)
    """
    json_dir = Path(json_dir)
    yolo_dir = Path(yolo_dir)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    logger.info(f"Converting {len(json_files)} JSON files to YOLO format...")

    for json_file in json_files:
        json_ann = JSONAnnotation.load_json(json_file)

        yolo_content = YOLOAnnotation.json_to_yolo(json_ann)

        yolo_file = yolo_dir / (json_file.stem + ".txt")
        YOLOAnnotation.save_yolo(yolo_content, yolo_file)

        logger.info(f"✓ {json_file.name} → {yolo_file.name}")

    logger.info(f"Conversion complete! {len(json_files)} files converted.")


def yolo_directory_to_json(
    yolo_dir: Path,
    json_dir: Path,
    image_dir: Path,
    image_dimensions: Dict[str, Tuple[int, int]]
):
    """
    Convert all YOLO annotations in directory to JSON format.

    Args:
        yolo_dir: Directory with YOLO .txt files
        json_dir: Output directory for JSON files
        image_dir: Directory with image files
        image_dimensions: Dict mapping image_name -> (width, height)
    """
    yolo_dir = Path(yolo_dir)
    json_dir = Path(json_dir)
    json_dir.mkdir(parents=True, exist_ok=True)

    yolo_files = sorted(yolo_dir.glob("*.txt"))
    logger.info(f"Converting {len(yolo_files)} YOLO files to JSON format...")

    for yolo_file in yolo_files:
        image_name = yolo_file.stem
        dims = image_dimensions.get(image_name)

        if dims is None:
            logger.warning(f"Missing dimensions for {image_name}, skipping")
            continue

        width, height = dims

        yolo_content = YOLOAnnotation.load_yolo(yolo_file)
        json_ann = YOLOAnnotation.yolo_to_json(
            yolo_content,
            str(image_dir / f"{image_name}.jpg"),
            width,
            height
        )

        json_file = json_dir / (yolo_file.stem + ".json")
        JSONAnnotation.save_json(json_ann, json_file)

        logger.info(f"✓ {yolo_file.name} → {json_file.name}")

    logger.info(f"Conversion complete! {len(yolo_files)} files converted.")


# =============================================================================
# BATCH PROCESSING
# =============================================================================

def process_json_dataset(
    json_dir: Path,
    output_dir: Path,
    train_split: float = 0.7,
    val_split: float = 0.15,
):
    """
    Process JSON annotations and organize into train/val/test with YOLO labels.

    Args:
        json_dir: Directory with JSON annotation files
        output_dir: Output directory for organized dataset
        train_split: Training split ratio
        val_split: Validation split ratio
    """
    import numpy as np

    json_dir = Path(json_dir)
    output_dir = Path(output_dir)

    # Create structure
    for split in ["train", "val", "test"]:
        (output_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    json_files = sorted(json_dir.glob("*.json"))
    n = len(json_files)

    # Shuffle and split
    np.random.seed(42)
    indices = np.random.permutation(n)
    train_count = int(n * train_split)
    val_count = int(n * val_split)

    splits = {
        "train": indices[:train_count],
        "val": indices[train_count:train_count + val_count],
        "test": indices[train_count + val_count:],
    }

    logger.info(f"Processing {n} JSON annotations...")
    logger.info(f"  Train: {len(splits['train'])}")
    logger.info(f"  Val: {len(splits['val'])}")
    logger.info(f"  Test: {len(splits['test'])}")

    for split_name, split_indices in splits.items():
        for idx in split_indices:
            json_file = json_files[idx]
            json_ann = JSONAnnotation.load_json(json_file)

            # Convert to YOLO
            yolo_content = YOLOAnnotation.json_to_yolo(json_ann)

            # Save YOLO label
            yolo_file = output_dir / split_name / "labels" / (json_file.stem + ".txt")
            YOLOAnnotation.save_yolo(yolo_content, yolo_file)

        logger.info(f"✓ {split_name}/labels: {len(split_indices)} files")

    logger.info("Dataset processing complete!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert between JSON and YOLO annotations")
    parser.add_argument("command", choices=["json2yolo", "yolo2json", "process"])
    parser.add_argument("--json-dir", type=str, help="JSON directory")
    parser.add_argument("--yolo-dir", type=str, help="YOLO directory")
    parser.add_argument("--output-dir", type=str, help="Output directory")

    args = parser.parse_args()

    if args.command == "json2yolo":
        json_directory_to_yolo(Path(args.json_dir), Path(args.output_dir))
    elif args.command == "yolo2json":
        logger.error("yolo2json requires image dimensions (not yet implemented in CLI)")
    elif args.command == "process":
        process_json_dataset(Path(args.json_dir), Path(args.output_dir))
