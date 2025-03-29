import logging
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from dotenv import load_dotenv

from model import ModelManager
from preprocessing import preprocess_image

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Skin Cancer Classification API",
    description="API for classifying skin cancer images using TensorFlow Lite",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Modify in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
    "Melanocytic nevi,Melanoma,Benign keratosis-like lesions,Basal cell carcinoma,Actinic keratoses,Vascular lesions,Dermatofibroma"
# Initialize model manager
model_path = os.getenv("MODEL_PATH", "./skin_cancer_model.tflite")
class_labels = os.getenv("CLASS_LABELS", "Melanocytic nevi,Melanoma,Benign keratosis-like lesions,Basal cell carcinoma,Actinic keratoses,Vascular lesions,Dermatofibroma").split(",")
model_manager = ModelManager(model_path, class_labels)

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    logger.info("Loading TFLite model...")
    try:
        model_manager.load_model()
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        # We'll continue running but log the error
        # In production, you might want to exit if model loading fails

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model_manager.is_model_loaded():
        return {"status": "healthy", "model_loaded": True}
    return {"status": "unhealthy", "model_loaded": False}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    """Classify skin image for cancer detection"""
    logger.info(f"Received classification request for file: {file.filename}")
    
    # Check if model is loaded
    if not model_manager.is_model_loaded():
        logger.error("Model not loaded")
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Validate file type
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="Only image files are accepted")
    
    try:
        # Read image file
        contents = await file.read()
        
        # Preprocess image
        processed_image = preprocess_image(contents)
        
        # Run inference
        prediction = model_manager.predict(processed_image)
        
        logger.info(f"Classification completed for {file.filename}")
        return prediction
    
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification error: {str(e)}")

if __name__ == "__main__":
    # For local development only
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

