"""Monitoring and metrics utilities."""
import logging
import time
from datetime import datetime
from typing import Optional, Dict, Any
from functools import wraps
import json

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collect and log metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Any] = {}
        self.start_time = datetime.utcnow()

    def record_metric(self, name: str, value: Any, tags: Optional[Dict] = None):
        """
        Record a metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags dict
        """
        timestamp = datetime.utcnow().isoformat()
        key = f"{name}:{timestamp}"

        self.metrics[key] = {
            "name": name,
            "value": value,
            "timestamp": timestamp,
            "tags": tags or {},
        }

        logger.info(f"Metric: {name}={value} {json.dumps(tags or {})}")

    def record_inference_time(self, elapsed: float, num_detections: int):
        """Record inference time metric."""
        self.record_metric(
            "inference_time_ms",
            round(elapsed * 1000, 2),
            {"detections": num_detections}
        )

    def record_prediction(self, mask_count: int, no_mask_count: int, confidence: float):
        """Record prediction metric."""
        self.record_metric(
            "prediction",
            {
                "mask_count": mask_count,
                "no_mask_count": no_mask_count,
                "confidence": round(confidence, 4),
            },
            {"total": mask_count + no_mask_count}
        )

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        return {
            "uptime_seconds": uptime,
            "total_metrics_recorded": len(self.metrics),
            "start_time": self.start_time.isoformat(),
        }


def measure_time(func):
    """Decorator to measure function execution time."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.time() - start
            logger.info(f"{func.__name__} took {elapsed:.3f}s")
    return wrapper


class AlertManager:
    """Manage alerts for anomalies."""

    def __init__(self, alert_threshold_no_mask: float = 0.5):
        """
        Initialize alert manager.

        Args:
            alert_threshold_no_mask: Alert if no_mask ratio exceeds this
        """
        self.alert_threshold_no_mask = alert_threshold_no_mask
        self.alerts = []

    def check_prediction(
        self,
        mask_count: int,
        no_mask_count: int,
        confidence: float,
    ) -> Optional[str]:
        """
        Check prediction and generate alert if needed.

        Args:
            mask_count: Number of people with masks
            no_mask_count: Number of people without masks
            confidence: Average confidence

        Returns:
            Alert message or None
        """
        total = mask_count + no_mask_count
        if total == 0:
            return None

        no_mask_ratio = no_mask_count / total

        if no_mask_ratio >= self.alert_threshold_no_mask:
            alert = (
                f"ALERT: High no-mask ratio ({no_mask_ratio:.2%}) - "
                f"mask_count={mask_count}, no_mask_count={no_mask_count}"
            )
            self.alerts.append({
                "timestamp": datetime.utcnow().isoformat(),
                "message": alert,
                "mask_count": mask_count,
                "no_mask_count": no_mask_count,
                "no_mask_ratio": no_mask_ratio,
            })
            logger.warning(alert)
            return alert

        return None

    def get_recent_alerts(self, limit: int = 10) -> list:
        """Get recent alerts."""
        return self.alerts[-limit:]


# Global instances
metrics = MetricsCollector()
alerts = AlertManager()
