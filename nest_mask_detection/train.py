"""Training script for car model detection."""
import logging
import json
from pathlib import Path
from typing import Optional
import argparse
import yaml

from config import config
from model import MaskDetectionModel
from dataset import DataLoader

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_dataset_yaml(data_dir: Path, output_path: Path):
    """
    Create YOLO format dataset.yaml file.

    Args:
        data_dir: Path to data directory
        output_path: Path to save dataset.yaml
    """
    yaml_content = {
        "path": str(data_dir.absolute()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 23,
        "names": {
            0: "Toyota", 1: "Honda", 2: "BMW", 3: "Mercedes", 4: "Audi",
            5: "Volkswagen", 6: "Ford", 7: "Chevy", 8: "Tesla", 9: "Nissan",
            10: "Hyundai", 11: "Kia", 12: "Pizza", 13: "Truck", 14: "Bus", 15: "Police", 16: "USPS", 17: "FedEx", 18:"LandRover", 19:"Porsche", 20:"Jeep", 21:"Lexus", 22:"Volvo"
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)

    logger.info(f"Created dataset.yaml at {output_path}")


def setup_dataset(data_dir: Optional[Path] = None) -> Path:
    """
    Setup dataset directory structure and YAML file.

    Args:
        data_dir: Data directory path

    Returns:
        Path to dataset.yaml
    """
    if data_dir is None:
        data_dir = config.dataset.data_dir

    data_dir = Path(data_dir)
    logger.info(f"Setting up dataset in {data_dir}")

    # Create directory structure
    DataLoader.create_dataset_dirs(data_dir)

    # Create dataset.yaml
    yaml_path = data_dir / "dataset.yaml"
    create_dataset_yaml(data_dir, yaml_path)

    return yaml_path


def train(
    data_yaml: Path,
    epochs: int = 100,
    batch_size: int = 15,
    learning_rate: float = 0.001,
    model_name: str = "yolov8n",
    device: str = "cpu",
    save_dir: Optional[Path] = None,
):
    """
    Train mask detection model.

    Args:
        data_yaml: Path to dataset.yaml
        epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        model_name: YOLOv8 model variant
        device: Device to use (cpu, cuda)
        save_dir: Directory to save trained model
    """
    logger.info("=" * 80)
    logger.info("CAR MODEL DETECTION TRAINING")
    logger.info("=" * 80)

    # Initialize model
    logger.info(f"Initializing {model_name} model...")
    model = MaskDetectionModel(model_name=model_name, device=device)

    # Print model stats
    stats = model.get_model_stats()
    logger.info(f"Model Statistics: {json.dumps(stats, indent=2)}")

    # Start training
    logger.info("Starting training...")
    results = model.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        lr0=learning_rate,
        patience=config.training.patience,
        img_size=config.model.img_size,
    )

    # Save model
    if save_dir is None:
        save_dir = Path("models")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    model_path = save_dir / f"{model_name}_cars.pt"
    model.save(model_path)

    logger.info("=" * 80)
    logger.info(f"Training completed! Model saved to {model_path}")
    logger.info("=" * 80)

    return model_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Train car model detection")
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(config.dataset.data_dir),
        help="Data directory (default: from config)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=config.training.epochs,
        help="Number of epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.dataset.batch_size,
        help="Batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=config.training.learning_rate,
        help="Learning rate",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.model.model_name,
        help="YOLOv8 model variant (n, s, m, l, x)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=config.model.device,
        help="Device (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="models",
        help="Directory to save trained model",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Only setup dataset structure without training",
    )

    args = parser.parse_args()

    # Setup dataset
    yaml_path = setup_dataset(args.data_dir)

    if args.setup_only:
        logger.info("Dataset setup completed. Exiting.")
        return

    # Train
    train(
        data_yaml=yaml_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        model_name=args.model,
        device=args.device,
        save_dir=args.save_dir,
    )


if __name__ == "__main__":
    main()
