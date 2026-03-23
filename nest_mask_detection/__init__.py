"""Nest Mask Detection - Real-time COVID-19 mask detection for Google Nest cameras."""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Real-time mask detection for Google Nest cameras using YOLOv8"

from config import config
from model import MaskDetectionModel
from database import DatabaseClient
from nest_integration import NestDeviceAccess, NestPubSubListener, NestCloudStorage
from monitoring import MetricsCollector, AlertManager, measure_time

__all__ = [
    "config",
    "MaskDetectionModel",
    "DatabaseClient",
    "NestDeviceAccess",
    "NestPubSubListener",
    "NestCloudStorage",
    "MetricsCollector",
    "AlertManager",
    "measure_time",
]
