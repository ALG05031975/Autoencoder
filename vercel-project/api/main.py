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

app = FastAPI(
    title="Autoencoder Defect Detection API",
    version="1.0.0",
    description="Complete defect detection API with all original features"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel-compatible directories (using /tmp)
BASE_DIR = Path("/tmp/defect_detection")
UPLOAD_DIR = BASE_DIR / "uploads"
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR = BASE_DIR / "models"

for directory in [UPLOAD_DIR, RESULT_DIR, MODEL_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

class Autoencoder(torch.nn.Module):
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

@app.post("/api/load_model")
async def load_model(
    model_file: UploadFile = File(...),
    force_cpu: bool = Form(False),
    debug_mode: bool = Form(False)
):
    """Load the autoencoder model"""
    try:
        # Set device configuration
        if force_cpu:
            app.state.device = torch.device("cpu")
        else:
            app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Save model file
        model_path = MODEL_DIR / f"model_{uuid.uuid4().hex}.pth"
        with open(model_path, "wb") as buffer:
            buffer.write(await model_file.read())

        # Load and initialize model
        app.state.autoencoder = Autoencoder().to(app.state.device)
        app.state.autoencoder.load_state_dict(torch.load(model_path, map_location=app.state.device))
        app.state.autoencoder.eval()
        app.state.model_loaded = True

        # Cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return JSONResponse({
            "status": "success",
            "message": f"Model loaded on {app.state.device}",
            "model_size": f"{os.path.getsize(model_path)/1024/1024:.2f}MB",
            "debug_mode": debug_mode
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Model loading failed",
                "device": str(app.state.device)
            }
        )

@app.post("/api/detect")
async def detect_defects(
    image: UploadFile = File(...),
    heatmap_threshold: float = Form(0.4, ge=0.1, le=0.9),
    min_defect_area: int = Form(100),
    visualize: bool = Form(True)
):
    """Detect defects in an image"""
    if not app.state.model_loaded:
        raise HTTPException(400, detail="Model not loaded. Please load model first.")

    try:
        # Read and validate image
        contents = await image.read()
        img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, detail="Invalid image file")

        # Process image
        h, w = img.shape[:2]
        resized = cv2.resize(img, (256, 256))
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).unsqueeze(0).float() / 255.0
        tensor = tensor.to(app.state.device)

        # Model inference
        with torch.no_grad():
            recon = app.state.autoencoder(tensor)
            diff = torch.abs(tensor - recon).mean(1).squeeze().cpu().numpy()
            heatmap = cv2.resize(diff, (w, h))

        # Thresholding and contour detection
        _, mask = cv2.threshold((heatmap * 255).astype(np.uint8), int(heatmap_threshold * 255), 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Process defects
        defects = []
        result_img = img.copy() if visualize else None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= min_defect_area:
                x, y, w, h = cv2.boundingRect(cnt)
                defects.append({
                    "x": x, "y": y, "width": w, "height": h,
                    "area": float(area),
                    "confidence": float(np.max(heatmap[y:y+h, x:x+w]))
                })
                if visualize:
                    cv2.rectangle(result_img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(result_img, f"{area:.0f}px", (x, y-5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

        # Prepare response
        response = {
            "status": "success",
            "defect_count": len(defects),
            "defects": defects,
            "heatmap_size": f"{heatmap.nbytes/1024:.2f}KB",
            "image_size": f"{len(contents)/1024:.2f}KB"
        }

        # Save visualization if requested
        if visualize and defects:
            result_id = uuid.uuid4().hex
            result_path = RESULT_DIR / f"result_{result_id}.jpg"
            cv2.imwrite(str(result_path), result_img)
            response["result_url"] = f"/api/results/{result_id}"

        return JSONResponse(response)

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "message": "Detection failed",
                "input_size": f"{len(contents)/1024:.2f}KB" if 'contents' in locals() else None
            }
        )

@app.get("/api/results/{result_id}")
async def get_result(result_id: str):
    """Retrieve saved detection results"""
    result_path = RESULT_DIR / f"result_{result_id}.jpg"
    if not result_path.exists():
        raise HTTPException(404, detail="Result not found or expired")
    return FileResponse(result_path)

@app.get("/api/system_info")
async def system_info():
    """Get system information"""
    return {
        "system": {
            "python": sys.version,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "device": str(app.state.device)
        },
        "model": {
            "loaded": app.state.model_loaded,
            "type": "Autoencoder"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "operational",
        "model_ready": app.state.model_loaded,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api")
async def root():
    """API root endpoint"""
    return {
        "message": "Autoencoder Defect Detection API",
        "version": app.version,
        "endpoints": {
            "load_model": "POST /api/load_model",
            "detect": "POST /api/detect",
            "results": "GET /api/results/{id}",
            "health": "GET /api/health",
            "docs": "/api/docs"
        }
    }