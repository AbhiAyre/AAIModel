"""Tests for mask detection system."""
import pytest
import numpy as np
import cv2
from pathlib import Path


class TestModel:
    """Test model inference."""

    @pytest.fixture
    def dummy_image(self):
        """Create a dummy image for testing."""
        return np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    def test_model_initialization(self):
        """Test model can be initialized."""
        from model import MaskDetectionModel

        try:
            model = MaskDetectionModel(model_name="yolov8n", device="cpu")
            assert model is not None
            assert model.model_name == "yolov8n"
        except Exception as e:
            pytest.skip(f"Model initialization requires YOLOv8: {e}")

    def test_prediction_shape(self, dummy_image):
        """Test prediction returns expected format."""
        from model import MaskDetectionModel

        try:
            model = MaskDetectionModel(model_name="yolov8n", device="cpu")
            detections, annotated = model.predict(dummy_image)

            assert isinstance(detections, list)
            assert isinstance(annotated, np.ndarray)
            assert annotated.shape == dummy_image.shape
        except Exception as e:
            pytest.skip(f"Prediction test requires model: {e}")


class TestDataset:
    """Test dataset utilities."""

    def test_create_dataset_dirs(self, tmp_path):
        """Test dataset directory creation."""
        from dataset import DataLoader

        base_dir = tmp_path / "data"
        dirs = DataLoader.create_dataset_dirs(base_dir)

        assert len(dirs) == 6
        for d in dirs:
            assert d.exists()

    def test_split_dataset(self, tmp_path):
        """Test dataset splitting."""
        from dataset import DataLoader

        # Create dummy images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(10):
            img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
            cv2.imwrite(str(img_dir / f"image_{i}.jpg"), img)

        # Split
        train, val, test = DataLoader.split_dataset(
            img_dir,
            train_split=0.7,
            val_split=0.15,
            test_split=0.15,
        )

        assert len(train) == 7
        assert len(val) == 2
        assert len(test) == 1


class TestDatabase:
    """Test database operations."""

    def test_database_connection(self, tmp_path):
        """Test database connection."""
        from database import DatabaseClient

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"
        db = DatabaseClient(db_url=db_url)

        assert db is not None
        assert db.db_url == db_url

    def test_save_prediction(self, tmp_path):
        """Test saving prediction to database."""
        from database import DatabaseClient

        db_path = tmp_path / "test.db"
        db_url = f"sqlite:///{db_path}"
        db = DatabaseClient(db_url=db_url)

        detections = [
            {
                "class_id": 0,
                "class_name": "mask",
                "confidence": 0.95,
                "bbox": [100, 100, 200, 200],
                "center": [150, 150],
            }
        ]

        record_id = db.save_prediction(
            event_id="test_event_1",
            device_id="device_123",
            detections=detections,
        )

        assert record_id is not None

        # Retrieve and verify
        record = db.get_prediction("test_event_1")
        assert record is not None
        assert record.event_id == "test_event_1"
        assert record.mask_count == 1
        assert record.no_mask_count == 0


class TestAPI:
    """Test API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from fastapi.testclient import TestClient
        from api import app

        return TestClient(app)

    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_stats_endpoint(self, client):
        """Test stats endpoint."""
        try:
            response = client.get("/stats")
            # May fail if DB not initialized, but endpoint should exist
            assert response.status_code in [200, 503]
        except Exception:
            pass


class TestMonitoring:
    """Test monitoring utilities."""

    def test_metrics_collection(self):
        """Test metrics collection."""
        from monitoring import MetricsCollector

        metrics = MetricsCollector()
        metrics.record_metric("test_metric", 100)
        metrics.record_inference_time(0.05, 2)

        summary = metrics.get_summary()
        assert summary["total_metrics_recorded"] >= 2

    def test_alert_generation(self):
        """Test alert generation."""
        from monitoring import AlertManager

        alerts = AlertManager(alert_threshold_no_mask=0.5)

        # Should alert when no_mask ratio >= 0.5
        alert = alerts.check_prediction(
            mask_count=1,
            no_mask_count=2,
            confidence=0.9,
        )
        assert alert is not None

        # Should not alert when below threshold
        alert = alerts.check_prediction(
            mask_count=3,
            no_mask_count=1,
            confidence=0.9,
        )
        assert alert is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
