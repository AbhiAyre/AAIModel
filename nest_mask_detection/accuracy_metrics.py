"""Compute accuracy metrics by comparing model predictions with ground truth labels."""
import logging
from pathlib import Path
from typing import List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Car model classes
CLASSES = {
    "Toyota": 0, "Honda": 1, "BMW": 2, "Mercedes": 3, "Audi": 4,
    "Volkswagen": 5, "Ford": 6, "Chevy": 7, "Tesla": 8, "Nissan": 9,
    "Hyundai": 10, "Kia": 11, "Pizza": 12, "Truck": 13, "Bus": 14
}


def compute_accuracy_metrics(results_list: List[Dict], labels_dir: Path) -> Dict:
    """
    Compute accuracy metrics by comparing predictions with ground truth labels.
    
    Args:
        results_list: List of detection results from model
        labels_dir: Path to directory containing ground truth .txt labels
    
    Returns:
        Dictionary with accuracy metrics
    """
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    labels_dir = Path(labels_dir)
    
    # Lists to store predictions and ground truth
    all_predictions = []
    all_ground_truth = []
    
    logger.info(f"\n{'='*80}")
    logger.info("COMPUTING ACCURACY METRICS")
    logger.info(f"{'='*80}\n")
    
    for result in results_list:
        image_name = result["image_name"]
        label_file = labels_dir / f"{Path(image_name).stem}.txt"
        
        # Read ground truth labels
        ground_truth_classes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        class_id = int(parts[0])
                        ground_truth_classes.append(class_id)
        
        # Get predicted classes
        predicted_classes = []
        if result["detections"]:
            for det in result["detections"]:
                class_name = det["class_name"]
                # Get class ID from CLASSES mapping
                class_id = next((v for k, v in CLASSES.items() if k == class_name), None)
                if class_id is not None:
                    predicted_classes.append(class_id)
        
        logger.info(f"Image: {image_name}")
        logger.info(f"  Ground Truth Classes: {ground_truth_classes}")
        logger.info(f"  Predicted Classes: {predicted_classes}")
        
        # Compare predictions with ground truth
        if ground_truth_classes:
            all_ground_truth.extend(ground_truth_classes)
            
            # Pad predictions to match ground truth length
            if len(predicted_classes) < len(ground_truth_classes):
                predicted_classes.extend([-1] * (len(ground_truth_classes) - len(predicted_classes)))
            elif len(predicted_classes) > len(ground_truth_classes):
                predicted_classes = predicted_classes[:len(ground_truth_classes)]
            
            all_predictions.extend(predicted_classes)
            logger.info(f"  Match: {predicted_classes == ground_truth_classes}\n")
        else:
            logger.warning(f"  No ground truth labels found for {image_name}\n")
    
    # Compute overall metrics
    metrics = {
        "total_images": len(results_list),
        "total_predictions": sum(len(r["detections"]) for r in results_list),
        "class_metrics": {}
    }
    
    if all_ground_truth and all_predictions:
        # Overall metrics
        metrics["precision"] = precision_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
        metrics["recall"] = recall_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
        metrics["f1_score"] = f1_score(all_ground_truth, all_predictions, average='weighted', zero_division=0)
        
        # Per-class metrics
        unique_classes = sorted(set(all_ground_truth))
        for class_id in unique_classes:
            class_name = next((k for k, v in CLASSES.items() if v == class_id), f"Class_{class_id}")
            
            class_gt = [1 if x == class_id else 0 for x in all_ground_truth]
            class_pred = [1 if x == class_id else 0 for x in all_predictions]
            
            metrics["class_metrics"][class_name] = {
                "precision": precision_score(class_gt, class_pred, zero_division=0),
                "recall": recall_score(class_gt, class_pred, zero_division=0),
                "f1_score": f1_score(class_gt, class_pred, zero_division=0),
                "support": sum(class_gt)
            }
    else:
        logger.warning("Not enough data to compute metrics")
    
    return metrics


def print_accuracy_report(metrics: Dict) -> None:
    """Print formatted accuracy metrics report."""
    logger.info(f"\n{'='*80}")
    logger.info("ACCURACY METRICS REPORT")
    logger.info(f"{'='*80}\n")
    
    logger.info(f"Total Images Tested: {metrics['total_images']}")
    logger.info(f"Total Predictions: {metrics['total_predictions']}\n")
    
    if "precision" in metrics:
        logger.info(f"Overall Metrics:")
        logger.info(f"  Precision: {metrics['precision']:.4f}")
        logger.info(f"  Recall: {metrics['recall']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}\n")
        
        if metrics["class_metrics"]:
            logger.info(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<8}")
            logger.info("-" * 64)
            
            for class_name, scores in sorted(metrics["class_metrics"].items()):
                logger.info(
                    f"{class_name:<20} {scores['precision']:<12.4f} "
                    f"{scores['recall']:<12.4f} {scores['f1_score']:<12.4f} "
                    f"{scores['support']:<8}"
                )
    else:
        logger.warning("Could not compute metrics - check label files")