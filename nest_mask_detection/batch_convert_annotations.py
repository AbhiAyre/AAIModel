"""Batch convert all polygon annotations to YOLO format."""
import json
import logging
from pathlib import Path
from polygon_converter import PolygonAnnotationConverter

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Image dimensions (fixed size for all annotations)
IMAGE_WIDTH = 825
IMAGE_HEIGHT = 463

def batch_convert_all_annotations():
    """Convert all JSON annotations to YOLO format."""

    json_dir = Path("SampleImages/train/labels")
    output_dir = json_dir  # Save in same location

    json_files = sorted(json_dir.glob("*.json"))

    logger.info(f"\n{'='*80}")
    logger.info(f"BATCH CONVERTING {len(json_files)} POLYGON ANNOTATIONS TO YOLO FORMAT")
    logger.info(f"{'='*80}\n")
    logger.info(f"📐 Image dimensions: {IMAGE_WIDTH} × {IMAGE_HEIGHT} pixels")
    logger.info(f"📁 Input directory: {json_dir}")
    logger.info(f"📁 Output directory: {output_dir}")
    logger.info(f"\n{'Processing...'}\n")

    success_count = 0
    error_count = 0

    for i, json_file in enumerate(json_files, 1):
        try:
            # Convert
            yolo_content = PolygonAnnotationConverter.convert_polygon_file(
                json_file,
                IMAGE_WIDTH,
                IMAGE_HEIGHT,
                output_file=None  # Don't save yet
            )

            if yolo_content:
                # Save YOLO file
                txt_file = output_dir / (json_file.stem + ".txt")
                with open(txt_file, "w") as f:
                    f.write(yolo_content)

                # Show progress
                status = "✓"
                success_count += 1

                # Get car model
                try:
                    ann_data = json.load(open(json_file))
                    if isinstance(ann_data, list) and ann_data:
                        car_model = ann_data[0].get("labels", {}).get("labelName", "Unknown")
                    else:
                        car_model = "Unknown"
                except:
                    car_model = "Unknown"

                logger.info(f"{status} [{i:3d}/{len(json_files)}] {json_file.stem:50s} → {car_model}")
            else:
                error_count += 1
                logger.info(f"✗ [{i:3d}/{len(json_files)}] {json_file.stem:50s} → No detections")

        except Exception as e:
            error_count += 1
            logger.info(f"✗ [{i:3d}/{len(json_files)}] {json_file.stem:50s} → ERROR: {str(e)[:40]}")

    logger.info(f"\n{'='*80}")
    logger.info(f"CONVERSION COMPLETE!")
    logger.info(f"{'='*80}")
    logger.info(f"✓ Successfully converted: {success_count}/{len(json_files)}")
    logger.info(f"✗ Errors: {error_count}")
    logger.info(f"")

    # Count YOLO files created
    txt_files = list(json_dir.glob("*.txt"))
    logger.info(f"📊 YOLO .txt files created: {len(txt_files)}")
    logger.info(f"   Ready for training! ✅")
    logger.info(f"")

if __name__ == "__main__":
    batch_convert_all_annotations()
