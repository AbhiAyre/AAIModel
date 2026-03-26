"""Configuration management for Nest mask detection system."""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional


@dataclass
class GoogleNestConfig:
    """Google Nest Device Access configuration."""
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT_ID", "")
    device_id: str = os.getenv("NEST_DEVICE_ID", "")
    api_key: str = os.getenv("GOOGLE_HOME_API_KEY", "")
    pubsub_topic: str = os.getenv("PUBSUB_TOPIC", "")
    pubsub_subscription: str = os.getenv("PUBSUB_SUBSCRIPTION", "")


@dataclass
class OAuth2Config:
    """OAuth 2.0 configuration."""
    client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    client_secret: str = os.getenv("GOOGLE_CLIENT_SECRET", "")
    redirect_uri: str = os.getenv("GOOGLE_REDIRECT_URI", "http://localhost:8000/auth/callback")
    scopes: list = None

    def __post_init__(self):
        """Initialize default scopes."""
        if self.scopes is None:
            self.scopes = [
                "https://www.googleapis.com/auth/sdm.service",
                "https://www.googleapis.com/auth/pubsub",
            ]


@dataclass
class ModelConfig:
    """ML model configuration."""
    model_name: str = "yolov8n"  # nano for faster inference on edge
    confidence_threshold: float = 0.5
    iou_threshold: float = 0.45
    img_size: int = 640
    num_classes: int = 15  # Car models (15 classes)
    device: str = "cpu"  # or "cuda" for GPU

    # Car model classes
    class_names: list = None

    def __post_init__(self):
        """Initialize class names."""
        if self.class_names is None:
            self.class_names = [
                 "Toyota", "Honda", "BMW", "Mercedes", "Audi", "Volkswagen", "Ford", "Chevy", "Tesla", "Nissan", "Hyundai", "Kia", "Pizza", "Truck", "Bus", "Police", "USPS", "FedEx", "LandRover", "Porsche", "Jeep", "Lexus","Volvo" ]

@dataclass
class DatasetConfig:
    """Dataset configuration."""
    data_dir: Path = Path("/Users/abhiayre/MLModel/AAIModel/nest_mask_detection/SampleImages")
    train_dir: Path = Path("/Users/abhiayre/MLModel/AAIModel/nest_mask_detection/SampleImages/train")
    val_dir: Path = Path("/Users/abhiayre/MLModel/AAIModel/nest_mask_detection/SampleImages/val")
    test_dir: Path = Path("/Users/abhiayre/MLModel/AAIModel/nest_mask_detection/SampleImages/test")
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    batch_size: int = 16
    num_workers: int = 4


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 0.0005
    momentum: float = 0.937
    warmup_epochs: int = 3
    patience: int = 20  # early stopping
    seed: int = 42


@dataclass
class APIConfig:
    """FastAPI configuration."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    max_workers: int = 4


@dataclass
class DatabaseConfig:
    """Database configuration."""
    db_url: str = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")
    echo: bool = False


@dataclass
class Config:
    """Main configuration class."""
    nest: GoogleNestConfig
    oauth: OAuth2Config
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    api: APIConfig
    database: DatabaseConfig


    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            nest=GoogleNestConfig(),
            oauth=OAuth2Config(),
            model=ModelConfig(),
            dataset=DatasetConfig(),
            training=TrainingConfig(),
            api=APIConfig(),
            database=DatabaseConfig(),
        )


# Global config instance
config = Config.from_env()
