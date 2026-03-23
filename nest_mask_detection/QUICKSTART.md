# Quick Start Guide - Nest Mask Detection

## 5-Minute Setup

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp .env.example .env
# Edit .env with your Google Cloud credentials:
# - GOOGLE_CLOUD_PROJECT_ID
# - NEST_DEVICE_ID
# - GOOGLE_HOME_API_KEY
# - PUBSUB_TOPIC
# - PUBSUB_SUBSCRIPTION
```

### 3. Start API Server
```bash
python -m uvicorn api:app --reload
# Server running at http://localhost:8000
```

### 4. Test Inference
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@your_image.jpg"
```

---

## Development Workflow

### Dataset Preparation
```bash
# 1. Organize your images in YOLO format:
mkdir -p data/mask_detection/{train,val,test}/{images,labels}

# 2. YOLO label format for each image:
# class_id x_center y_center width height (all normalized 0-1)
# Example: 0 0.5 0.5 0.3 0.4  (class=mask at center)
```

### Model Training
```bash
# Train on your dataset
python train.py \
  --data-dir data/mask_detection \
  --epochs 100 \
  --batch-size 16 \
  --device cuda

# Model saved to: models/yolov8n_trained.pt
```

### Event Processing
```bash
# Process real Nest camera events
python consumer.py

# Listen for Pub/Sub events and run inference automatically
```

---

## API Usage

### Predict on Image
```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@image.jpg" \
  -F "event_id=event123" \
  -F "device_id=device456"
```

**Response:**
```json
{
  "num_detections": 2,
  "mask_count": 1,
  "no_mask_count": 1,
  "confidence_avg": 0.9234,
  "detections": [
    {
      "class_id": 0,
      "class_name": "mask",
      "confidence": 0.95,
      "bbox": [100, 100, 200, 200],
      "center": [150, 150]
    }
  ]
}
```

### Get Statistics
```bash
curl "http://localhost:8000/stats?device_id=device456"
```

### Health Check
```bash
curl "http://localhost:8000/health"
```

---

## Docker Deployment

### Single Container
```bash
docker build -t mask-detection .
docker run -p 8000:8000 \
  -e GOOGLE_CLOUD_PROJECT_ID=your-project \
  -e NEST_DEVICE_ID=your-device \
  -e GOOGLE_HOME_API_KEY=your-key \
  mask-detection
```

### Full Stack (API + Database + Consumer)
```bash
docker-compose up -d
docker-compose logs -f
```

---

## Configuration Reference

```bash
# Model Performance
MODEL_NAME=yolov8n          # nano (fast), s, m, l, x (accurate)
CONFIDENCE_THRESHOLD=0.5    # 0-1, lower = more detections
IOU_THRESHOLD=0.45          # 0-1, NMS threshold
DEVICE=cpu                  # cpu, cuda, mps

# API
API_HOST=0.0.0.0
API_PORT=8000

# Database
DATABASE_URL=sqlite:///./predictions.db
# PostgreSQL: postgresql://user:pass@host:5432/dbname
```

---

## Troubleshooting

**Q: Model downloads fail?**
- A: First run requires downloading ~60MB model
- Ensure internet connection and disk space

**Q: GPU not detected?**
- A: Install CUDA: https://developer.nvidia.com/cuda-downloads
- Verify: python -c "import torch; print(torch.cuda.is_available())"

**Q: Pub/Sub events not received?**
- A: Verify credentials and permissions
- Check topic/subscription exist in Google Cloud Console

**Q: Slow inference?**
- A: Use smaller model (yolov8n) or enable GPU
- Check: DEVICE=cuda in .env

---

## Next Steps

1. **Collect Data**: Gather Nest camera images for training
2. **Annotate**: Label images using tools like Label Studio or Roboflow
3. **Fine-tune**: Train model on your specific use case
4. **Deploy**: Use docker-compose for production setup
5. **Monitor**: Check /stats endpoint for performance metrics
6. **Alert**: Configure alert thresholds for high no-mask ratios

---

## Resources

- API Docs: http://localhost:8000/docs (Swagger UI)
- YOLOv8 Guide: https://docs.ultralytics.com/
- Google Nest API: https://developers.google.com/nest/device-access
- FastAPI: https://fastapi.tiangolo.com/
