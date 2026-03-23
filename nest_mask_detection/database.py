"""Database models and utilities for storing predictions."""
import logging
from datetime import datetime
from typing import List, Optional
from sqlalchemy import create_engine, Column, String, Integer, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

from config import config

logger = logging.getLogger(__name__)

Base = declarative_base()


class PredictionRecord(Base):
    """Database model for storing car detection predictions."""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    event_id = Column(String, unique=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    device_id = Column(String, nullable=False)
    image_url = Column(String)
    num_detections = Column(Integer)
    detected_models = Column(JSON)  # List of detected car models with confidence
    model_summary = Column(JSON)  # Count of each car model {Toyota: 2, Honda: 1, ...}
    avg_confidence = Column(Float)
    detections_json = Column(JSON)  # Store full detection data


class DatabaseClient:
    """Database client for predictions."""

    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize database client.

        Args:
            db_url: Database URL (defaults from config)
        """
        if db_url is None:
            db_url = config.database.db_url

        self.db_url = db_url
        self.engine = create_engine(db_url, echo=config.database.echo)
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )

        # Create tables
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Connected to database: {db_url}")

    def get_session(self) -> Session:
        """Get database session."""
        return self.SessionLocal()

    def save_prediction(
        self,
        event_id: str,
        device_id: str,
        detections: List[dict],
        image_url: Optional[str] = None,
    ) -> Optional[int]:
        """
        Save prediction to database.

        Args:
            event_id: Unique event ID
            device_id: Device ID
            detections: List of detection dicts
            image_url: URL of the image

        Returns:
            Record ID or None
        """
        try:
            session = self.get_session()

            # Calculate stats
            confidences = [d["confidence"] for d in detections]
            confidence_avg = sum(confidences) / len(confidences) if confidences else 0.0

            # Build model summary (count of each detected car model)
            model_summary = {}
            detected_models = []
            for d in detections:
                class_name = d["class_name"]
                model_summary[class_name] = model_summary.get(class_name, 0) + 1
                detected_models.append({
                    "model": class_name,
                    "confidence": d["confidence"],
                    "bbox": d["bbox"]
                })

            # Create record
            record = PredictionRecord(
                event_id=event_id,
                device_id=device_id,
                image_url=image_url,
                num_detections=len(detections),
                detected_models=detected_models,
                model_summary=model_summary,
                avg_confidence=confidence_avg,
                detections_json=detections,
            )

            session.add(record)
            session.commit()
            record_id = record.id

            logger.info(f"Saved prediction {event_id} to database (ID: {record_id})")
            logger.info(f"  Models: {model_summary}")
            session.close()

            return record_id
        except Exception as e:
            logger.error(f"Failed to save prediction: {e}")
            return None

    def get_prediction(self, event_id: str) -> Optional[PredictionRecord]:
        """Get prediction by event ID."""
        try:
            session = self.get_session()
            record = session.query(PredictionRecord).filter_by(
                event_id=event_id
            ).first()
            session.close()
            return record
        except Exception as e:
            logger.error(f"Failed to get prediction: {e}")
            return None

    def get_predictions_by_device(
        self,
        device_id: str,
        limit: int = 100,
    ) -> List[PredictionRecord]:
        """Get recent predictions for a device."""
        try:
            session = self.get_session()
            records = session.query(PredictionRecord).filter_by(
                device_id=device_id
            ).order_by(
                PredictionRecord.timestamp.desc()
            ).limit(limit).all()
            session.close()
            return records
        except Exception as e:
            logger.error(f"Failed to get predictions: {e}")
            return []

    def get_stats(self, device_id: Optional[str] = None) -> dict:
        """
        Get prediction statistics.

        Args:
            device_id: Filter by device (optional)

        Returns:
            Statistics dict
        """
        try:
            session = self.get_session()
            query = session.query(PredictionRecord)

            if device_id:
                query = query.filter_by(device_id=device_id)

            total = query.count()
            avg_confidence = 0
            avg_detections = 0
            car_model_counts = {}

            if total > 0:
                records = query.all()
                avg_confidence = sum(r.avg_confidence for r in records) / total
                avg_detections = sum(r.num_detections for r in records) / total

                # Aggregate car model detections
                for record in records:
                    if record.model_summary:
                        for model, count in record.model_summary.items():
                            car_model_counts[model] = car_model_counts.get(model, 0) + count

            session.close()

            return {
                "total_predictions": total,
                "avg_detections": round(avg_detections, 2),
                "avg_confidence": round(avg_confidence, 4),
                "car_model_detections": car_model_counts,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {}
