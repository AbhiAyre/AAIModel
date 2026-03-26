from polygon_converter import PolygonAnnotationConverter
from pathlib import Path
from PIL import Image

# Paths
json_dir = Path("SampleImages/train/labels")
output_dir = Path("SampleImages/train/labels")
img_dir = Path("SampleImages/train/images")

# Get image dimensions
image_dimensions = {}
for img_file in sorted(img_dir.glob("*")):
    if img_file.suffix.lower() in ['.jpg', '.png', '.jpeg']:
        with Image.open(img_file) as im:
            image_dimensions[img_file.stem] = (im.width, im.height)

print(f"Found {len(image_dimensions)} images with dimensions")

# Batch convert all JSON files
PolygonAnnotationConverter.batch_convert_directory(
    json_dir, 
    output_dir, 
    image_dimensions
)