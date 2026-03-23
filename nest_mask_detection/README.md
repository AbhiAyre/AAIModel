# Nest Mask Detection

Real-time COVID-19 mask detection for Google Nest cameras using YOLOv8.

## Features

- **Real-time inference**: YOLOv8-based mask detection with sub-100ms latency
- **Event-driven**: Automatic processing of Nest camera events via Google Pub/Sub
- **REST API**: FastAPI server for on-demand inference
- **Database storage**: Track predictions and statistics over time
- **Monitoring**: Built-in metrics collection and anomaly detection
- **Docker deployment**: Ready-to-deploy containerized application
- **Google Cloud native**: Integrated with Google Cloud services

## Project Structure

```
nest_mask_detection/
├── config.py              # Configuration management
├── dataset.py             # Data loading and preprocessing
├── model.py               # YOLOv8 model wrapper
├── train.py               # Training pipeline
├── api.py                 # FastAPI inference server
├── consumer.py            # Pub/Sub event listener
├── nest_integration.py    # Google Nest integration
├── database.py            # Database models and utilities
├── monitoring.py          # Metrics and alerts
├── requirements.txt       # Python dependencies
├── Dockerfile             # Container image
├── docker-compose.yml     # Multi-container setup
├── .env.example           # Environment variables template
└── README.md              # This file
```

## Quick Start

### 1. Prerequisites

- Python 3.9+
- Google Cloud account with:
  - Device Access enabled
  - Pub/Sub topic and subscription created
  - Service account with appropriate permissions
- Docker & Docker Compose (optional, for containerized deployment)

### 2. Installation

```bash
# Clone or navigate to project
cd nest_mask_detection

# Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your Google Cloud credentials
# Required fields:
# - GOOGLE_CLOUD_PROJECT_ID
# - NEST_DEVICE_ID
# - GOOGLE_HOME_API_KEY
# - PUBSUB_TOPIC
# - PUBSUB_SUBSCRIPTION
```

### 3. Dataset Setup

```bash
# Create dataset directory structure
mkdir -p data/mask_detection

# Place your annotated images in YOLO format:
# data/mask_detection/train/images/ + labels/
# data/mask_detection/val/images/ + labels/
# data/mask_detection/test/images/ + labels/

# YOLO format: class_id x_center y_center width height (normalized)
```

### 4. Training

```bash
# Setup dataset structure
python train.py --setup-only

# Train model (requires GPU recommended)
python train.py \
  --epochs 100 \
  --batch-size 16 \
  --learning-rate 0.001 \
  --model yolov8n \
  --device cuda  # or 'cpu'
```

### 5. API Server

```bash
# Run inference server
python -m uvicorn api:app --host 0.0.0.0 --port 8000

# API will be available at http://localhost:8000
# Docs: http://localhost:8000/docs
```

### 6. Event Consumer

```bash
# Run Pub/Sub event listener (requires Google Cloud credentials)
python consumer.py
```

## API Endpoints

### Health Check
```bash
GET /health
```

### Run Inference
```bash
POST /predict
- file: Image file (JPEG/PNG)
- event_id: Optional Nest event ID
- device_id: Optional device ID
- image_url: Optional URL
```

### Get Statistics
```bash
GET /stats?device_id=<device_id>
```

### Model Info
```bash
GET /model-info
```

### Nest Device Info
```bash
GET /device-info
```

## Docker Deployment

### Build Image
```bash
docker build -t nest-mask-detection:latest .
```

### Docker Compose (Full Stack)
```bash
# Start all services (API, Database, Consumer)
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Docker Run (API Only)
```bash
docker run -p 8000:8000 \
  -e GOOGLE_CLOUD_PROJECT_ID=<project> \
  -e NEST_DEVICE_ID=<device> \
  -e GOOGLE_HOME_API_KEY=<api_key> \
  -e DATABASE_URL=sqlite:///./predictions.db \
  nest-mask-detection:latest
```

## Configuration

All configuration is managed via environment variables (see `.env.example`):

### Google Cloud
- `GOOGLE_CLOUD_PROJECT_ID`: GCP project ID
- `NEST_DEVICE_ID`: Nest device ID
- `GOOGLE_HOME_API_KEY`: Google Home API key
- `PUBSUB_TOPIC`: Pub/Sub topic
- `PUBSUB_SUBSCRIPTION`: Pub/Sub subscription

### Model
- `MODEL_NAME`: YOLOv8 variant (n, s, m, l, x)
- `CONFIDENCE_THRESHOLD`: Detection confidence threshold (0.0-1.0)
- `IOU_THRESHOLD`: IoU threshold for NMS (0.0-1.0)
- `IMG_SIZE`: Input image size (default: 640)
- `DEVICE`: Computation device (cpu, cuda, mps)

### API
- `API_HOST`: API host (default: 0.0.0.0)
- `API_PORT`: API port (default: 8000)
- `DEBUG`: Debug mode (default: false)

### Database
- `DATABASE_URL`: Database connection string
  - SQLite: `sqlite:///./predictions.db`
  - PostgreSQL: `postgresql://user:password@host:5432/mask_detection`

## Usage Examples

### Python Client
```python
import requests
from pathlib import Path

# Read image
with open("image.jpg", "rb") as f:
    files = {"file": f}
    response = requests.post(
        "http://localhost:8000/predict",
        files=files,
        params={
            "event_id": "nest_event_123",
            "device_id": "device_abc",
        }
    )

# Parse results
result = response.json()
print(f"Detections: {result['num_detections']}")
print(f"Masks: {result['mask_count']}")
print(f"No masks: {result['no_mask_count']}")
print(f"Avg confidence: {result['confidence_avg']:.3f}")
```

### cURL
```bash
# Send image for inference
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "event_id=event123"

# Get statistics
curl "http://localhost:8000/stats"

# Get model info
curl "http://localhost:8000/model-info"
```

## Database Schema

### predictions table
- `id`: Primary key
- `event_id`: Unique event identifier
- `timestamp`: Prediction timestamp
- `device_id`: Nest device ID
- `image_url`: URL of the event image
- `num_detections`: Total detections
- `mask_count`: People with masks
- `no_mask_count`: People without masks
- `confidence_avg`: Average detection confidence
- `detections_json`: Full detection data (JSON)

## Monitoring

The system includes built-in monitoring:

### Metrics Collected
- Inference time (milliseconds)
- Number of detections
- Mask vs no-mask counts
- Average confidence scores

### Alerts
- High no-mask ratio (configurable threshold)
- Inference failures
- API errors

### Accessing Metrics
```python
from monitoring import metrics, alerts

# Get summary
summary = metrics.get_summary()

# Get recent alerts
recent_alerts = alerts.get_recent_alerts()
```

## Performance

### YOLOv8 Model Variants
| Model | Parameters | Speed (ms) | mAP |
|-------|-----------|-----------|-----|
| Nano (n) | 3.2M | ~10-20 | 50% |
| Small (s) | 11.2M | ~20-40 | 61% |
| Medium (m) | 25.9M | ~40-70 | 68% |
| Large (l) | 43.7M | ~70-120 | 72% |

### Deployment Recommendations
- **Edge (Raspberry Pi, Coral): yolov8n**
- **CPU server: yolov8s or yolov8m**
- **GPU server: yolov8l or yolov8x**

## Development

### Running Tests
```bash
pytest tests/ -v --cov=.
```

### Code Style
```bash
black . --line-length 100
flake8 .
```

### Debugging
```bash
# Set DEBUG=true in .env
# Run API with logging
python -u api.py
```

## Troubleshooting

### Model loading fails
- Ensure CUDA/cuDNN installed if using GPU
- Check disk space for model downloads (~60-200MB)

### Pub/Sub events not received
- Verify credentials in .env
- Check Pub/Sub topic and subscription exist
- Ensure service account has pubsub.subscriber role

### API timeouts
- Increase batch processing time
- Use smaller model variant
- Check GPU memory if using CUDA

### Database connection errors
- Verify DATABASE_URL format
- For PostgreSQL: ensure database exists
- Check credentials in connection string

## References

- [Google Nest Device Access](https://developers.google.com/nest/device-access)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Google Pub/Sub](https://cloud.google.com/pubsub)

## License

[Add your license here]

## Contributing

[Add contribution guidelines here]
