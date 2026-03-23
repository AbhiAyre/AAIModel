"""FastAPI inference server for mask detection."""
import logging
import io
import base64
from typing import Optional, List
from datetime import datetime
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel

from config import config
from auth import OAuth2Handler

# Optional imports for features that require heavy dependencies
try:
    from model import MaskDetectionModel
except ImportError:
    MaskDetectionModel = None

try:
    from database import DatabaseClient
except ImportError:
    DatabaseClient = None

try:
    from nest_integration import NestDeviceAccess
except ImportError:
    NestDeviceAccess = None

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Nest Mask Detection API",
    version="1.0.0",
    description="Real-time mask detection for Google Nest cameras"
)

# Initialize components
model = None
db = None
nest = None
oauth = OAuth2Handler()


@app.on_event("startup")
async def startup():
    """Initialize components on startup."""
    global model, db, nest

    logger.info("Starting Nest Mask Detection API...")

    # Load model (optional - requires torch/ultralytics)
    if MaskDetectionModel is not None:
        try:
            model = MaskDetectionModel(
                model_name=config.model.model_name,
                device=config.model.device
            )
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.warning(f"Model not available: {e}")
    else:
        logger.warning("torch/ultralytics not installed - model features disabled")

    # Initialize database (optional - can use SQLite)
    if DatabaseClient is not None:
        try:
            db = DatabaseClient()
            logger.info("Database initialized")
        except Exception as e:
            logger.warning(f"Database connection failed: {e}")
    else:
        logger.warning("sqlalchemy not available - database features disabled")

    # Initialize Nest client (optional - requires Google Cloud)
    if NestDeviceAccess is not None:
        try:
            nest = NestDeviceAccess(
                project_id=config.nest.project_id,
                api_key=config.nest.api_key,
                device_id=config.nest.device_id,
            )
            logger.info("Nest client initialized")
        except Exception as e:
            logger.warning(f"Nest client not available: {e}")
    else:
        logger.warning("Google Cloud libraries not available - Nest features disabled")


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    logger.info("Shutting down Nest Mask Detection API...")


# Pydantic models
class DetectionResult(BaseModel):
    """Single detection result."""
    class_id: int
    class_name: str
    confidence: float
    bbox: List[int]
    center: List[int]


class PredictionResponse(BaseModel):
    """Prediction response."""
    event_id: Optional[str] = None
    timestamp: str
    device_id: str
    image_url: Optional[str] = None
    num_detections: int
    mask_count: int
    no_mask_count: int
    confidence_avg: float
    detections: List[DetectionResult]
    annotated_image_base64: Optional[str] = None


class StatsResponse(BaseModel):
    """Statistics response."""
    total_predictions: int
    avg_mask_count: float
    avg_no_mask_count: float
    avg_confidence: float


# API endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "db_connected": db is not None,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    event_id: Optional[str] = Query(None),
    device_id: Optional[str] = Query(None),
    image_url: Optional[str] = Query(None),
):
    """
    Run inference on uploaded image.

    Args:
        file: Image file (JPEG/PNG)
        event_id: Optional Nest event ID
        device_id: Optional device ID
        image_url: Optional image URL

    Returns:
        Prediction results
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Read image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        # Run inference
        detections, annotated_img = model.predict(
            img,
            conf_threshold=config.model.confidence_threshold,
            iou_threshold=config.model.iou_threshold,
        )

        # Calculate stats
        mask_count = sum(1 for d in detections if d["class_id"] == 0)
        no_mask_count = sum(1 for d in detections if d["class_id"] == 1)
        confidences = [d["confidence"] for d in detections]
        confidence_avg = sum(confidences) / len(confidences) if confidences else 0.0

        # Encode annotated image
        _, buffer = cv2.imencode(".jpg", annotated_img)
        annotated_base64 = base64.b64encode(buffer).decode()

        # Save to database
        if db:
            db.save_prediction(
                event_id=event_id or f"event_{int(datetime.utcnow().timestamp())}",
                device_id=device_id or config.nest.device_id,
                detections=detections,
                image_url=image_url,
            )

        # Format response
        detection_results = [
            DetectionResult(
                class_id=d["class_id"],
                class_name=d["class_name"],
                confidence=d["confidence"],
                bbox=d["bbox"],
                center=d["center"],
            )
            for d in detections
        ]

        return PredictionResponse(
            event_id=event_id,
            timestamp=datetime.utcnow().isoformat(),
            device_id=device_id or config.nest.device_id,
            image_url=image_url,
            num_detections=len(detections),
            mask_count=mask_count,
            no_mask_count=no_mask_count,
            confidence_avg=confidence_avg,
            detections=detection_results,
            annotated_image_base64=annotated_base64,
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats", response_model=StatsResponse)
async def get_stats(device_id: Optional[str] = Query(None)):
    """
    Get prediction statistics.

    Args:
        device_id: Optional device ID filter

    Returns:
        Statistics
    """
    if db is None:
        raise HTTPException(status_code=503, detail="Database not connected")

    try:
        stats = db.get_stats(device_id=device_id)
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-info")
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        return model.get_model_stats()
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/device-info")
async def get_device_info():
    """Get Nest device information."""
    if nest is None:
        raise HTTPException(status_code=503, detail="Nest client not initialized")

    try:
        return nest.get_device_info()
    except Exception as e:
        logger.error(f"Device info error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# OAuth 2.0 Endpoints
@app.get("/auth/login")
async def oauth_login():
    """Redirect user to Google OAuth login."""
    auth_url = oauth.get_authorization_url()
    return RedirectResponse(url=auth_url)


@app.get("/auth/callback")
async def oauth_callback(code: str = Query(...), state: str = Query(...)):
    """Handle OAuth callback from Google."""
    try:
        if not oauth.exchange_code_for_token(code):
            raise HTTPException(
                status_code=400,
                detail="Failed to exchange authorization code"
            )

        return {
            "status": "success",
            "message": "Successfully authenticated with Google",
            "access_token": oauth.access_token[:20] + "...",  # Partial token for security
        }
    except Exception as e:
        logger.error(f"OAuth callback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/auth/status")
async def oauth_status():
    """Check OAuth authentication status."""
    access_token = oauth.get_access_token()
    return {
        "authenticated": access_token is not None,
        "has_refresh_token": oauth.refresh_token is not None,
        "token_expiry": oauth.token_expiry.isoformat() if oauth.token_expiry else None,
    }


@app.post("/auth/logout")
async def oauth_logout():
    """Clear cached tokens."""
    oauth.clear_cache()
    return {"status": "logged_out"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=config.api.host,
        port=config.api.port,
        log_level="info",
    )
