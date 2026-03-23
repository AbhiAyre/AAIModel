"""Convert polygon annotation format (CVAT/Roboflow) to YOLO format."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Car model classes
CLASSES = {
    "Toyota": 0, "Honda": 1, "BMW": 2, "Mercedes-Benz": 3, "Audi": 4,
    "Volkswagen": 5, "Ford": 6, "Chevrolet": 7, "Tesla": 8, "Nissan": 9,
    "Hyundai": 10, "Kia": 11, "Mazda": 12, "Subaru": 13, "Lexus": 14
}


class PolygonAnnotationConverter:
    """Convert polygon annotations to YOLO format."""

    @staticmethod
    def parse_polygon_json(json_file: Path) -> List[Dict]:
        """
        Parse polygon annotation JSON file.

        Args:
            json_file: Path to JSON file with polygon annotations

        Returns:
            List of annotations
        """
        with open(json_file) as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            # Try common keys
            if "annotations" in data:
                return data["annotations"]
            elif "objects" in data:
                return data["objects"]
            else:
                return [data]
        else:
            return []

    @staticmethod
    def polygon_to_yolo(
        annotation: Dict,
        image_width: int,
        image_height: int
    ) -> Optional[str]:
        """
        Convert single polygon annotation to YOLO format.

        Args:
            annotation: Single annotation dict
            image_width: Image width in pixels
            image_height: Image height in pixels

        Returns:
            YOLO format string or None if error
        """
        try:
            # Extract car model label
            if "labels" in annotation:
                car_model = annotation["labels"].get("labelName", "").strip()
            else:
                logger.warning("No labels found in annotation")
                return None

            if car_model not in CLASSES:
                logger.warning(f"Unknown car model: {car_model}")
                return None

            class_id = CLASSES[car_model]

            # Use rect mask (bounding box) if available
            if "rectMask" in annotation:
                rect = annotation["rectMask"]
                x_min = rect["xMin"]
                y_min = rect["yMin"]
                width = rect["width"]
                height = rect["height"]

                # Convert to center coordinates
                x_center = (x_min + width / 2) / image_width
                y_center = (y_min + height / 2) / image_height
                width_norm = width / image_width
                height_norm = height / image_height

                # YOLO format
                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                return yolo_line

            # Fallback: calculate bbox from polygon points
            elif "content" in annotation:
                points = annotation["content"]
                if not points:
                    return None

                # Get bounding box from points
                xs = [p["x"] for p in points]
                ys = [p["y"] for p in points]

                x_min, x_max = min(xs), max(xs)
                y_min, y_max = min(ys), max(ys)

                width = x_max - x_min
                height = y_max - y_min

                # Convert to center coordinates (normalized)
                x_center = (x_min + width / 2) / image_width
                y_center = (y_min + height / 2) / image_height
                width_norm = width / image_width
                height_norm = height / image_height

                yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {width_norm:.6f} {height_norm:.6f}"
                return yolo_line

        except Exception as e:
            logger.error(f"Error converting polygon to YOLO: {e}")
            return None

    @staticmethod
    def convert_polygon_file(
        json_file: Path,
        image_width: int,
        image_height: int,
        output_file: Optional[Path] = None
    ) -> str:
        """
        Convert entire polygon JSON file to YOLO format.

        Args:
            json_file: Input JSON file
            image_width: Image width
            image_height: Image height
            output_file: Optional output file path

        Returns:
            YOLO format content (all lines)
        """
        annotations = PolygonAnnotationConverter.parse_polygon_json(json_file)

        yolo_lines = []
        for ann in annotations:
            yolo_line = PolygonAnnotationConverter.polygon_to_yolo(
                ann, image_width, image_height
            )
            if yolo_line:
                yolo_lines.append(yolo_line)

        yolo_content = "\n".join(yolo_lines)

        if output_file:
            output_file = Path(output_file)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w") as f:
                f.write(yolo_content)
            logger.info(f"Saved YOLO annotations to {output_file}")

        return yolo_content

    @staticmethod
    def batch_convert_directory(
        json_dir: Path,
        output_dir: Path,
        image_dimensions: Dict[str, Tuple[int, int]]
    ):
        """
        Convert all polygon JSON files in directory to YOLO format.

        Args:
            json_dir: Directory with JSON annotation files
            output_dir: Output directory for YOLO .txt files
            image_dimensions: Dict mapping filename -> (width, height)
        """
        json_dir = Path(json_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        json_files = sorted(json_dir.glob("*.json"))
        logger.info(f"Converting {len(json_files)} polygon JSON files...")

        for json_file in json_files:
            filename = json_file.stem
            dims = image_dimensions.get(filename)

            if dims is None:
                logger.warning(f"No dimensions for {filename}, skipping")
                continue

            width, height = dims

            output_file = output_dir / (filename + ".txt")
            PolygonAnnotationConverter.convert_polygon_file(
                json_file, width, height, output_file
            )

        logger.info(f"✓ Conversion complete! {len(json_files)} files processed.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert polygon annotations to YOLO format"
    )
    parser.add_argument("json_file", type=str, help="JSON file to convert")
    parser.add_argument("--width", type=int, required=True, help="Image width")
    parser.add_argument("--height", type=int, required=True, help="Image height")
    parser.add_argument("--output", type=str, help="Output file path")

    args = parser.parse_args()

    result = PolygonAnnotationConverter.convert_polygon_file(
        Path(args.json_file),
        args.width,
        args.height,
        Path(args.output) if args.output else None
    )

    print("\nYOLO Format Output:")
    print(result)
