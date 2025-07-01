import os
import sys
import torch
import numpy as np
from pathlib import Path
import cv2
import uuid
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI(
    title="Autoencoder Defect Detection API",
    description="API for detecting defects using autoencoder model",
    version="1.0.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories (Vercel-compatible)
BASE_DIR = Path("/tmp/defect_detection")
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

for directory in [UPLOAD_DIR, RESULT_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class Autoencoder(torch.nn.Module):
    """Autoencoder model for defect detection"""
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

# Global application state
app.state.model_loaded = False
app.state.autoencoder = None
app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def process_image(image: np.ndarray, threshold: float = 0.4) -> tuple:
    """Process image through autoencoder and detect defects"""
    try:
        # Preprocess
        h, w = image.shape[:2]
        resized = cv2.resize(image, (256, 256))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        tensor = tensor.to(app.state.device)

        # Inference
        with torch.no_grad():
            recon = app.state.autoencoder(tensor)
            diff = torch.abs(tensor - recon).mean(1).squeeze().cpu().numpy()
            heatmap = cv2.resize(diff, (w, h))

        # Thresholding
        _, mask = cv2.threshold((heatmap * 255).astype(np.uint8), int(threshold * 255), 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        defects = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Minimum defect area
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append({
                    "x": x, "y": y, "width": w, "height": h,
                    "area": float(area)
                })

        return defects, heatmap

    except Exception as e:
        raise RuntimeError(f"Image processing failed: {str(e)}")

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
    """Endpoint to load the autoencoder model"""
    try:
        # Set device
        if force_cpu:
            app.state.device = torch.device("cpu")
        
        # Save model file
        model_path = MODEL_DIR / f"model_{uuid.uuid4().hex}.pth"
        with open(model_path, "wb") as buffer:
            buffer.write(await model_file.read())

        # Load model
        app.state.autoencoder = Autoencoder().to(app.state.device)
        app.state.autoencoder.load_state_dict(torch.load(model_path, map_location=app.state.device))
        app.state.autoencoder.eval()
        app.state.model_loaded = True

        return JSONResponse({
            "status": "success",
            "message": "Model loaded successfully",
            "device": str(app.state.device),
            "model_size": f"{os.path.getsize(model_path)/1024/1024:.2f}MB"
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Model loading failed: {str(e)}"
        )

@app.post("/api/detections")
async def detect_defects(
    image: UploadFile = File(...),
    threshold: float = Form(0.4, ge=0.1, le=0.9),
    visualize: bool = Form(True)
):
    """Endpoint to detect defects in an image"""
    if not app.state.model_loaded:
        raise HTTPException(status_code=400, detail="Model not loaded. Please load model first.")

    try:
        # Read and validate image
        contents = await image.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process image
        defects, heatmap = process_image(img, threshold)

        # Generate result
        result_id = uuid.uuid4().hex
        response = {
            "status": "success",
            "defect_count": len(defects),
            "defects": defects,
            "heatmap_size": f"{heatmap.nbytes/1024:.2f}KB"
        }

        # If visualization requested
        if visualize and defects:
            result_img = img.copy()
            for defect in defects:
                cv2.rectangle(result_img,
                            (defect["x"], defect["y"]),
                            (defect["x"] + defect["width"], defect["y"] + defect["height"]),
                            (0, 0, 255), 2)
            
            result_path = RESULT_DIR / f"result_{result_id}.jpg"
            cv2.imwrite(str(result_path), result_img)
            response["result_url"] = f"/api/results/{result_id}"

        return JSONResponse(response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """Endpoint to retrieve detection results"""
    result_path = RESULT_DIR / f"result_{result_id}.jpg"
    if not result_path.exists():
        raise HTTPException(status_code=404, detail="Result not found")
    return FileResponse(result_path)

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "model_loaded": app.state.model_loaded,
        "device": str(app.state.device),
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api")
async def root():
    """Root endpoint"""
    return {
        "message": "Autoencoder Defect Detection API",
        "version": app.version,
        "docs": "/docs",
        "redoc": "/redoc"
    }