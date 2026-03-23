"""Example usage of JSON annotation format."""
import json
from pathlib import Path
from annotation_converter import JSONAnnotation, YOLOAnnotation


def example_create_json_annotation():
    """Create a sample JSON annotation file."""
    print("=" * 80)
    print("EXAMPLE 1: Create JSON Annotation")
    print("=" * 80)

    # Sample detections
    detections = [
        {"model": "BMW", "x": 100, "y": 150, "width": 200, "height": 180},
        {"model": "Toyota", "x": 350, "y": 100, "width": 180, "height": 160},
    ]

    # Create annotation
    annotation = JSONAnnotation.create_annotation(
        image_path="sample_image.jpg",
        image_width=640,
        image_height=480,
        detections=detections
    )

    # Display
    print("\nJSON Annotation Format:")
    print(json.dumps(annotation, indent=2))

    # Save
    json_file = Path("sample_annotations.json")
    JSONAnnotation.save_json(annotation, json_file)
    print(f"\n✓ Saved to: {json_file}")

    return annotation


def example_json_to_yolo(json_annotation):
    """Convert JSON annotation to YOLO format."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: Convert JSON to YOLO Format")
    print("=" * 80)

    # Convert to YOLO
    yolo_content = YOLOAnnotation.json_to_yolo(json_annotation)

    print("\nYOLO Format (.txt file content):")
    print(yolo_content)

    # Save
    yolo_file = Path("sample_annotations.txt")
    YOLOAnnotation.save_yolo(yolo_content, yolo_file)
    print(f"\n✓ Saved to: {yolo_file}")

    return yolo_content


def example_yolo_to_json(yolo_content):
    """Convert YOLO format back to JSON."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Convert YOLO Back to JSON")
    print("=" * 80)

    # Convert back to JSON
    json_annotation = YOLOAnnotation.yolo_to_json(
        yolo_content,
        image_path="sample_image.jpg",
        image_width=640,
        image_height=480
    )

    print("\nJSON Annotation (reconstructed from YOLO):")
    print(json.dumps(json_annotation, indent=2))

    # Verify detections
    print("\nDetections:")
    for i, ann in enumerate(json_annotation["annotations"], 1):
        bbox = ann["bbox"]
        car_model = ann["car_model"]
        print(f"  {i}. {car_model} at pixel ({bbox['x']}, {bbox['y']}) "
              f"size {bbox['width']}x{bbox['height']}")


def example_workflow():
    """Show complete workflow."""
    print("\n" + "=" * 80)
    print("WORKFLOW: JSON → YOLO → Training")
    print("=" * 80)

    print("""
STEP 1: Collect Annotations in JSON Format
   - Use annotation tool or create manually
   - Store in: annotations/
   - Format: {"image": {...}, "annotations": [...]}

STEP 2: Convert JSON to YOLO Format
   - Command: python annotation_converter.py json2yolo --json-dir annotations/
   - Creates YOLO .txt files in output directory
   - Format: class_id x_center y_center width height

STEP 3: Organize for Training
   - Command: python annotation_converter.py process --json-dir annotations/
   - Automatically splits: train/val/test
   - Ready for YOLOv8 training!

STEP 4: Train Model
   - Command: python train.py --data-dir SampleImages/
   - Model trains on YOLO format labels
   - Done!
    """)


def example_json_structure():
    """Show JSON structure in detail."""
    print("\n" + "=" * 80)
    print("JSON ANNOTATION STRUCTURE")
    print("=" * 80)

    example = {
        "image": {
            "path": "SampleImages/train/images/car_photo_001.jpg",
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
            },
            {
                "id": 1,
                "car_model": "Toyota",
                "bbox": {
                    "x": 350,
                    "y": 100,
                    "width": 180,
                    "height": 160
                },
                "bbox_normalized": {
                    "x_center": 0.4219,
                    "y_center": 0.1111,
                    "width": 0.0938,
                    "height": 0.1111
                }
            }
        ]
    }

    print("\nExample JSON file (car_photo_001.json):")
    print(json.dumps(example, indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1]
        if command == "json":
            example_json_structure()
        elif command == "workflow":
            example_workflow()
        elif command == "all":
            example_json_structure()
            ann = example_create_json_annotation()
            yolo = example_json_to_yolo(ann)
            example_yolo_to_json(yolo)
            example_workflow()
    else:
        print("Usage: python annotation_examples.py [json|workflow|all]")
        print("\nExamples:")
        print("  json     - Show JSON structure")
        print("  workflow - Show complete workflow")
        print("  all      - Run all examples")
