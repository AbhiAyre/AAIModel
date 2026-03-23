"""Dataset utilities for mask detection."""
import os
import json
from pathlib import Path
from typing import List, Tuple, Dict
import cv2
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms


class MaskDetectionDataset(Dataset):
    """Dataset for mask detection with YOLO format annotations."""

    def __init__(
        self,
        img_dir: Path,
        annotations_dir: Path,
        img_size: int = 640,
        augment: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            img_dir: Directory containing images
            annotations_dir: Directory containing YOLO format .txt annotations
            img_size: Target image size
            augment: Whether to apply augmentations
        """
        self.img_dir = Path(img_dir)
        self.annotations_dir = Path(annotations_dir)
        self.img_size = img_size
        self.augment = augment

        # List all images
        self.img_files = sorted([
            f for f in self.img_dir.glob("**/*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        if not self.img_files:
            raise ValueError(f"No images found in {img_dir}")

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ]) if not augment else None

    def __len__(self) -> int:
        return len(self.img_files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get image and annotations.

        Returns:
            Tuple of (image, bboxes) where bboxes is list of dicts with keys:
            - class_id: int (0=mask, 1=no_mask)
            - bbox: [x_center, y_center, width, height] normalized
        """
        img_path = self.img_files[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to read image: {img_path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        # Load annotations
        ann_path = self.annotations_dir / (img_path.stem + ".txt")
        bboxes = []

        if ann_path.exists():
            with open(ann_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bboxes.append({
                            "class_id": class_id,
                            "bbox": [x_center, y_center, width, height]
                        })

        # Resize image
        img_resized = cv2.resize(img, (self.img_size, self.img_size))

        if self.transform:
            img_resized = self.transform(img_resized)

        return img_resized, bboxes

    def get_class_names(self) -> List[str]:
        """Get class names."""
        return ["mask", "no_mask"]


class DataLoader:
    """Data loading utilities."""

    @staticmethod
    def split_dataset(
        img_dir: Path,
        train_split: float = 0.7,
        val_split: float = 0.15,
        test_split: float = 0.15,
        seed: int = 42,
    ) -> Tuple[List[Path], List[Path], List[Path]]:
        """
        Split dataset into train, val, test.

        Args:
            img_dir: Directory containing images
            train_split: Proportion for training
            val_split: Proportion for validation
            test_split: Proportion for testing
            seed: Random seed

        Returns:
            Tuple of (train_imgs, val_imgs, test_imgs)
        """
        np.random.seed(seed)
        img_dir = Path(img_dir)

        # Get all images
        all_imgs = sorted([
            f for f in img_dir.glob("**/*")
            if f.suffix.lower() in [".jpg", ".jpeg", ".png"]
        ])

        # Shuffle
        np.random.shuffle(all_imgs)

        # Split
        n = len(all_imgs)
        train_count = int(n * train_split)
        val_count = int(n * val_split)

        train_imgs = all_imgs[:train_count]
        val_imgs = all_imgs[train_count:train_count + val_count]
        test_imgs = all_imgs[train_count + val_count:]

        return train_imgs, val_imgs, test_imgs

    @staticmethod
    def create_dataset_dirs(base_dir: Path):
        """Create standard dataset directory structure."""
        base_dir = Path(base_dir)
        dirs = [
            base_dir / "train" / "images",
            base_dir / "train" / "labels",
            base_dir / "val" / "images",
            base_dir / "val" / "labels",
            base_dir / "test" / "images",
            base_dir / "test" / "labels",
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
        return dirs
