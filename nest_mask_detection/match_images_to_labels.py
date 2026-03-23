"""Match PNG images with YOLO label files by sorting."""
from pathlib import Path
import shutil
import os

def match_images_to_labels():
    """Match PNG images with YOLO label files."""

    images_dir = Path("SampleImages/train/images")
    labels_dir = Path("SampleImages/train/labels")

    # Get sorted list of label files
    label_files = sorted([f for f in labels_dir.glob("*.txt")])

    # Get sorted list of PNG files
    png_files = sorted([f for f in images_dir.glob("*.png")])

    print(f"📋 Found {len(png_files)} PNG images and {len(label_files)} labels")
    print(f"\n🔄 Matching images to labels (by sorted order)...\n")

    # Match by creating a mapping of sorted images to sorted labels
    matched_count = 0

    for i, (label_file, png_file) in enumerate(zip(label_files, png_files), 1):
        # New PNG filename to match label
        new_png_name = images_dir / (label_file.stem + ".png")

        # Only rename if different
        if png_file != new_png_name:
            try:
                # Copy (not move) to preserve original
                shutil.copy2(png_file, new_png_name)
                matched_count += 1

                if i <= 5 or i % 20 == 0:
                    print(f"✓ [{i:3d}] {png_file.name:50s} → {new_png_name.name}")
                elif i == 6:
                    print(f"      ... (matching continues) ...\n")
            except Exception as e:
                print(f"✗ [{i:3d}] Error: {e}")
        else:
            matched_count += 1

    print(f"\n{'='*80}")
    print(f"✅ Matching complete!")
    print(f"   Matched: {matched_count}/{len(label_files)} images")

    # Verify
    matched_images = sorted([f for f in images_dir.glob("*.png")])
    print(f"   Total images now: {len(matched_images)}")

    # Check if all labels have matching images
    missing = 0
    for label_file in label_files:
        image_file = images_dir / (label_file.stem + ".png")
        if not image_file.exists():
            print(f"   ⚠️  Missing image for: {label_file.name}")
            missing += 1

    if missing == 0:
        print(f"\n✅ ALL LABELS HAVE MATCHING IMAGES!")
        print(f"\n📊 Dataset Status:")
        print(f"   Images: {len(matched_images)}")
        print(f"   Labels: {len(label_files)}")
        print(f"   Status: ✅ READY FOR TRAINING!")
    else:
        print(f"\n⚠️  {missing} labels missing corresponding images")

if __name__ == "__main__":
    match_images_to_labels()
