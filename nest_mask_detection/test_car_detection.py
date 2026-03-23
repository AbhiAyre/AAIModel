"""Test car detection on single image."""
import logging
import argparse
from pathlib import Path
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_car_detection(image_path: str, model_path: str = "models/yolov8n_cars.pt"):
    """
    Test car detection on an image.

    Args:
        image_path: Path to test image
        model_path: Path to trained model
    """
    try:
        from model import MaskDetectionModel
        from config import config
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install: python3 -m pip install torch ultralytics opencv-python")
        return

    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info(f"Train a model first: python3 train.py")
        return

    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return

    logger.info(f"Loading model: {model_path}")
    model = MaskDetectionModel(
        model_name=config.model.model_name,
        device=config.model.device
    )
    model.load(model_path)

    logger.info("Running inference...")
    detections, annotated_img = model.predict(
        image,
        conf_threshold=config.model.confidence_threshold,
        iou_threshold=config.model.iou_threshold,
    )

    # Print results
    logger.info(f"\n{'='*80}")
    logger.info(f"DETECTION RESULTS")
    logger.info(f"{'='*80}")
    logger.info(f"Total detections: {len(detections)}")

    if detections:
        logger.info(f"\n{'Class':<20} {'Confidence':<15} {'BBox':<30}")
        logger.info("-" * 65)

        for det in detections:
            class_name = det["class_name"]
            confidence = det["confidence"]
            bbox = det["bbox"]
            logger.info(
                f"{class_name:<20} {confidence:<15.3f} "
                f"({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})"
            )

        # Summary by class
        logger.info(f"\n{'Car Model Summary':<30}")
        logger.info("-" * 50)
        class_counts = {}
        for det in detections:
            class_name = det["class_name"]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1

        for class_name, count in sorted(class_counts.items()):
            logger.info(f"{class_name:<30} : {count} detected")

    else:
        logger.warning("No cars detected in image")

    # Save annotated image
    output_path = image_path.parent / f"{image_path.stem}_annotated.jpg"
    cv2.imwrite(str(output_path), annotated_img)
    logger.info(f"\n✓ Annotated image saved: {output_path}")

    logger.info(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description="Test car detection model")
    parser.add_argument(
        "image",
        type=str,
        help="Path to test image or directory of images",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n_cars.pt",
        help="Path to trained model",
    )

    args = parser.parse_args()

    image_path = Path(args.image)

    # If directory, test all images in it
    if image_path.is_dir():
        logger.info(f"Testing all images in directory: {image_path}")
        image_files = list(image_path.glob("*.jpg")) + list(image_path.glob("*.png"))

        if not image_files:
            logger.error(f"No images found in {image_path}")
            return

        logger.info(f"Found {len(image_files)} images")

        for img_file in sorted(image_files):
            logger.info(f"\n{'='*80}")
            logger.info(f"Testing: {img_file.name}")
            logger.info(f"{'='*80}")
            test_car_detection(str(img_file), args.model)
    else:
        # Single image
        test_car_detection(args.image, args.model)


if __name__ == "__main__":
    main()
