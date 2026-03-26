"""Test car detection on single image."""
import logging
import argparse
from pathlib import Path
import cv2
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_car_detection(image_path: str, model_path: str = "models/yolov8n_cars.pt"):
    """
    Test car detection on an image.

    Args:
        image_path: Path to test image
        model_path: Path to trained model
    
    Returns:
        Dictionary with detection results
    """
    try:
        from model import MaskDetectionModel
        from config import config
    except ImportError as e:
        logger.error(f"Missing dependencies: {e}")
        logger.error("Install: python3 -m pip install torch ultralytics opencv-python")
        return None

    image_path = Path(image_path)
    model_path = Path(model_path)

    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        return None

    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.info(f"Train a model first: python3 train.py")
        return None

    logger.info(f"Loading image: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        logger.error(f"Failed to load image: {image_path}")
        return None

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

    # Return results for PDF report
    return {
        "image_name": image_path.name,
        "detections": detections,
        "total_detections": len(detections),
        "annotated_image_path": str(output_path)
    }


def generate_pdf_report(results_list, output_dir, metrics=None):
    """
    Generate a PDF report with all detection results.
    
    Args:
        results_list: List of detection results dictionaries
        output_dir: Directory to save the PDF report
        metrics: Accuracy metrics to include in the report
    """
    try:
        from reportlab.lib.pagesizes import letter, A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak, Image
        from reportlab.lib import colors
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
    except ImportError:
        logger.error("Missing reportlab: pip install reportlab")
        return

    output_dir = Path(output_dir)
    pdf_path = output_dir / f"detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"

    logger.info(f"Generating PDF report: {pdf_path}")

    doc = SimpleDocTemplate(str(pdf_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()

    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=TA_CENTER
    )
    story.append(Paragraph("Car Detection Test Report", title_style))
    story.append(Spacer(1, 0.3*inch))

    # Report metadata
    meta_data = [
        f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total Images Tested: {len(results_list)}",
        f"Total Detections: {sum(r['total_detections'] for r in results_list)}"
    ]

    for meta in meta_data:
        story.append(Paragraph(meta, styles['Normal']))
    story.append(Spacer(1, 0.3*inch))

    # Results table
    table_data = [["Image Name", "Total Detections", "Detected Classes"]]
    
    for result in results_list:
        image_name = result["image_name"]
        total_det = result["total_detections"]
        
        # Get unique classes with confidence scores
        class_info = []
        if result["detections"]:
            for det in result["detections"]:
                class_name = det["class_name"]
                confidence = det["confidence"]
                class_info.append(f"{class_name} ({confidence:.2%})")
        
        classes_text = ", ".join(class_info) if class_info else "No detections"
        table_data.append([image_name, str(total_det), classes_text])

    # Create table
    table = Table(table_data, colWidths=[2*inch, 1.2*inch, 2.3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1f77b4')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.lightgrey]),
    ]))

    story.append(table)
    story.append(Spacer(1, 0.5*inch))

    # Detailed results per image
    story.append(Paragraph("Detailed Detection Results", styles['Heading2']))
    story.append(Spacer(1, 0.2*inch))

    for result in results_list:
        story.append(Paragraph(f"<b>{result['image_name']}</b>", styles['Heading3']))
        
        if result["detections"]:
            det_data = [["Class", "Confidence", "BBox Coordinates"]]
            for det in result["detections"]:
                bbox = det["bbox"]
                det_data.append([
                    det["class_name"],
                    f"{det['confidence']:.2%}",
                    f"({bbox[0]}, {bbox[1]}) - ({bbox[2]}, {bbox[3]})"
                ])
            
            det_table = Table(det_data, colWidths=[1.5*inch, 1.5*inch, 2.5*inch])
            det_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ]))
            story.append(det_table)
        else:
            story.append(Paragraph("No detections found", styles['Normal']))

        story.append(Spacer(1, 0.2*inch))

    # If metrics are provided, add them to the report
    if metrics:
        story.append(Paragraph("Accuracy Metrics", styles['Heading2']))
        story.append(Spacer(1, 0.2*inch))

        for metric_name, metric_value in metrics.items():
            story.append(Paragraph(f"<b>{metric_name}:</b> {metric_value}", styles['Normal']))
        
        story.append(Spacer(1, 0.5*inch))

    # Build PDF
    doc.build(story)
    logger.info(f"✓ PDF report saved: {pdf_path}")


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
    results_list = []

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
            result = test_car_detection(str(img_file), args.model)
            if result:
                results_list.append(result)
        
        # Compute accuracy metrics
        labels_dir = image_path.parent / "labels"
        if labels_dir.exists():
            from accuracy_metrics import compute_accuracy_metrics, print_accuracy_report

            metrics = compute_accuracy_metrics(results_list, labels_dir)
            print_accuracy_report(metrics)
            
            # Generate PDF report with metrics
            if results_list:
                generate_pdf_report(results_list, image_path, metrics)
        else:
            logger.warning(f"Labels directory not found: {labels_dir}")
            if results_list:
                generate_pdf_report(results_list, image_path)
    else:
        # Single image
        result = test_car_detection(args.image, args.model)
        if result:
            results_list.append(result)
            generate_pdf_report(results_list, image_path.parent)


if __name__ == "__main__":
    main()