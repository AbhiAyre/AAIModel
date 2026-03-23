"""Pub/Sub consumer for Nest camera events."""
import logging
import cv2
import numpy as np
import io
import tempfile
from pathlib import Path
from datetime import datetime

from config import config
from model import MaskDetectionModel
from database import DatabaseClient
from nest_integration import NestDeviceAccess, NestPubSubListener
from monitoring import metrics, alerts

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class NestEventProcessor:
    """Process Nest camera events."""

    def __init__(self):
        """Initialize processor."""
        self.model = MaskDetectionModel(
            model_name=config.model.model_name,
            device=config.model.device
        )
        self.db = DatabaseClient()
        self.nest = NestDeviceAccess(
            project_id=config.nest.project_id,
            api_key=config.nest.api_key,
            device_id=config.nest.device_id,
        )
        logger.info("Event processor initialized")

    def process_event(self, event: dict):
        """
        Process Nest event.

        Args:
            event: Event dict with keys: event_id, timestamp, url
        """
        try:
            event_id = event.get("event_id")
            image_url = event.get("url")

            logger.info(f"Processing event {event_id}")

            # Download event image
            image_data = self.nest.get_event_image(image_url)
            if image_data is None:
                logger.error(f"Failed to download image for event {event_id}")
                return

            # Decode image
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                logger.error(f"Failed to decode image for event {event_id}")
                return

            # Run inference
            logger.info(f"Running inference on {event_id}")
            start = datetime.utcnow()
            detections, annotated_img = self.model.predict(
                img,
                conf_threshold=config.model.confidence_threshold,
                iou_threshold=config.model.iou_threshold,
            )
            elapsed = (datetime.utcnow() - start).total_seconds()

            # Calculate stats
            mask_count = sum(1 for d in detections if d["class_id"] == 0)
            no_mask_count = sum(1 for d in detections if d["class_id"] == 1)
            confidences = [d["confidence"] for d in detections]
            confidence_avg = sum(confidences) / len(confidences) if confidences else 0.0

            # Record metrics
            metrics.record_inference_time(elapsed, len(detections))
            metrics.record_prediction(mask_count, no_mask_count, confidence_avg)

            # Check alerts
            alert = alerts.check_prediction(mask_count, no_mask_count, confidence_avg)

            # Save to database
            record_id = self.db.save_prediction(
                event_id=event_id,
                device_id=config.nest.device_id,
                detections=detections,
                image_url=image_url,
            )

            # Save annotated image
            self._save_annotated_image(event_id, annotated_img)

            logger.info(
                f"Event {event_id} processed: "
                f"detections={len(detections)}, "
                f"mask={mask_count}, "
                f"no_mask={no_mask_count}, "
                f"confidence={confidence_avg:.3f}, "
                f"time={elapsed:.2f}s"
            )

        except Exception as e:
            logger.error(f"Error processing event: {e}", exc_info=True)

    def _save_annotated_image(self, event_id: str, annotated_img: np.ndarray):
        """
        Save annotated image.

        Args:
            event_id: Event ID
            annotated_img: Annotated image
        """
        try:
            output_dir = Path("output") / "annotated_images"
            output_dir.mkdir(parents=True, exist_ok=True)

            filename = output_dir / f"{event_id}_annotated.jpg"
            cv2.imwrite(str(filename), annotated_img)
            logger.debug(f"Saved annotated image to {filename}")
        except Exception as e:
            logger.error(f"Failed to save annotated image: {e}")


def main():
    """Main entry point."""
    logger.info("=" * 80)
    logger.info("NEST MASK DETECTION - PUB/SUB CONSUMER")
    logger.info("=" * 80)

    # Initialize processor
    processor = NestEventProcessor()

    # Initialize Pub/Sub listener
    listener = NestPubSubListener(
        project_id=config.nest.project_id,
        subscription_id=config.nest.pubsub_subscription,
        callback=processor.process_event,
    )

    # Start listening
    try:
        listener.start_listening()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        listener.stop_listening()


if __name__ == "__main__":
    main()
