import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import uuid
from datetime import datetime
from typing import List, Dict, Optional

app = FastAPI(
    title="Autoencoder Defect Detection API",
    version="2.0.0",
    description="Complete production-ready defect detection API"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel-compatible directories
BASE_DIR = Path("/tmp/defect_api")
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

for directory in [UPLOAD_DIR, RESULT_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class Autoencoder(torch.nn.Module):
    """3-layer convolutional autoencoder"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU()
        )
        # Decoder
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(16, 3, kernel_size=3, stride=2, padding=1, output_padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

# Application state
app.state.model_loaded = False
app.state.autoencoder = None
app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.on_event("startup")
async def startup_event():
    """Initialize application state"""
    app.state.model_loaded = False
    app.state.autoencoder = None
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@app.post("/api/models")
async def load_model(
    model_file: UploadFile = File(...),
    force_cpu: bool = Form(False)
):
    """Load PyTorch model weights"""
    try:
        # Set device configuration
        if force_cpu:
            app.state.device = torch.device("cpu")
        
        # Save uploaded model
        model_path = MODEL_DIR / f"model_{uuid.uuid4().hex}.pth"
        with open(model_path, "wb") as f:
            contents = await model_file.read()
            f.write(contents)

        # Initialize model
        app.state.autoencoder = Autoencoder().to(app.state.device)
        app.state.autoencoder.load_state_dict(torch.load(model_path, map_location=app.state.device))
        app.state.autoencoder.eval()
        app.state.model_loaded = True

        # Cleanup GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return {
            "status": "success",
            "device": str(app.state.device),
            "model_size": f"{os.path.getsize(model_path)/1024/1024:.2f}MB"
        }

    except Exception as e:
        raise HTTPException(500, detail={
            "error": str(e),
            "message": "Model loading failed"
        })

@app.post("/api/detections")
async def detect_defects(
    image: UploadFile = File(...),
    threshold: float = Form(0.4, ge=0.1, le=0.9),
    min_area: int = Form(100, ge=10)
):
    """Detect defects in uploaded image"""
    if not app.state.model_loaded:
        raise HTTPException(400, detail="Model not loaded")
    
    try:
        # Read and validate image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid image file")

        # Preprocess
        h, w = img.shape[:2]
        resized = cv2.resize(img, (256, 256))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        tensor = tensor.to(app.state.device)

        # Model inference
        with torch.no_grad():
            recon = app.state.autoencoder(tensor)
            diff = torch.abs(tensor - recon).mean(1).squeeze().cpu().numpy()
            heatmap = cv2.resize(diff, (w, h))

        # Find defects
        _, mask = cv2.threshold((heatmap * 255).astype(np.uint8), int(threshold * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        defects = []
        result_img = img.copy()
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_area:
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append({
                    "x": x, "y": y, "width": w, "height": h,
                    "area": float(area),
                    "confidence": float(np.max(heatmap[y:y+h, x:x+w]))
                })
                # Draw rectangle
                cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)

        # Save and return results
        result_id = uuid.uuid4().hex
        result_path = RESULT_DIR / f"result_{result_id}.jpg"
        cv2.imwrite(str(result_path), result_img)

        return {
            "status": "success",
            "defects": defects,
            "result_url": f"/api/results/{result_id}",
            "image_size": f"{w}x{h}",
            "processing_time": datetime.now().isoformat()
        }

    except Exception as e:
        raise HTTPException(500, detail={
            "error": str(e),
            "message": "Detection failed"
        })

@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """Retrieve detection result image"""
    result_path = RESULT_DIR / f"result_{result_id}.jpg"
    if not result_path.exists():
        raise HTTPException(404, detail="Result not found")
    return FileResponse(result_path)

@app.get("/api/system")
async def system_info():
    """Get system information"""
    return {
        "python": sys.version,
        "torch": torch.__version__,
        "cuda": torch.cuda.is_available(),
        "device": str(app.state.device),
        "model_loaded": app.state.model_loaded
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api")
async def root():
    """API root endpoint"""
    return {
        "api": "Autoencoder Defect Detection",
        "version": app.version,
        "docs": "/docs",
        "endpoints": [
            {"method": "POST", "path": "/api/models", "desc": "Load model"},
            {"method": "POST", "path": "/api/detections", "desc": "Detect defects"},
            {"method": "GET", "path": "/api/results/{id}", "desc": "Get result image"}
        ]
    }