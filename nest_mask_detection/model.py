"""Model utilities for mask detection."""
import logging
from pathlib import Path
from typing import Tuple, List
import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger(__name__)


class MaskDetectionModel:
    """YOLO-based mask detection model."""

    def __init__(self, model_name: str = "yolov8n", device: str = "cpu"):
        """
        Initialize model.

        Args:
            model_name: YOLOv8 model variant (n=nano, s=small, m=medium, l=large, x=xlarge)
            device: Device to use (cpu, cuda)
        """
        self.model_name = model_name
        self.device = device
        self.model = YOLO(f"{model_name}.pt")
        self.model.to(device)
        logger.info(f"Loaded {model_name} on {device}")

    def predict(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.45,
    ) -> Tuple[List[dict], np.ndarray]:
        """
        Run inference on image.

        Args:
            image: Input image (BGR format)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold

        Returns:
            Tuple of (detections, annotated_image) where detections is list of dicts with keys:
            - class_id: int
            - class_name: str (car model name)
            - confidence: float
            - bbox: [x1, y1, x2, y2] in pixel coordinates
            - center: [x_center, y_center]
        """
        from config import config

        results = self.model(
            image,
            conf=conf_threshold,
            iou=iou_threshold,
            device=self.device
        )

        detections = []
        annotated_img = image.copy()

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                x_center = (x1 + x2) // 2
                y_center = (y1 + y2) // 2

                # Get class name from config
                class_names = config.model.class_names
                class_name = class_names[class_id] if class_id < len(class_names) else "unknown"

                detections.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2],
                    "center": [x_center, y_center],
                })

                # Draw bounding box with color
                color = (0, 255, 0)  # Green for cars
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_img,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2
                )

        return detections, annotated_img

    def train(
        self,
        data_yaml: Path,
        epochs: int = 100,
        img_size: int = 640,
        batch_size: int = 16,
        patience: int = 20,
        lr0: float = 0.001,
    ) -> dict:
        """
        Train model on dataset.

        Args:
            data_yaml: Path to dataset YAML file
            epochs: Number of epochs
            img_size: Image size
            batch_size: Batch size
            patience: Early stopping patience
            lr0: Initial learning rate

        Returns:
            Training results dict
        """
        logger.info(f"Starting training on {self.device}")

        results = self.model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=img_size,
            batch=batch_size,
            patience=patience,
            lr0=lr0,
            device=self.device,
            save=True,
            project="runs/mask_detection",
            name="train",
        )

        logger.info("Training completed")
        return results

    def save(self, path: Path):
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(str(path))
        logger.info(f"Model saved to {path}")

    def load(self, path: Path):
        """Load model."""
        self.model = YOLO(str(path))
        self.model.to(self.device)
        logger.info(f"Model loaded from {path}")

    def get_model_stats(self) -> dict:
        """Get model statistics."""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "parameters": sum(p.numel() for p in self.model.model.parameters()),
            "trainable_parameters": sum(
                p.numel() for p in self.model.model.parameters() if p.requires_grad
            ),
        }
