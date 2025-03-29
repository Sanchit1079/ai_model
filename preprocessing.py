import io
import logging
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Default image size for the model
# These should match your model's expected input dimensions
IMG_SIZE = (224, 224)  # Common size for many models

def preprocess_image(image_bytes):
    """
    Preprocess image for model inference
    
    Args:
        image_bytes (bytes): Raw image bytes
        
    Returns:
        numpy.ndarray: Preprocessed image as numpy array
    """
    try:
        # Open image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (in case of RGBA or grayscale)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize image to expected dimensions
        image = image.resize(IMG_SIZE)
        
        # Convert to numpy array and normalize to [0,1]
        img_array = np.array(image).astype(np.float32) / 255.0
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Image preprocessed to shape: {img_array.shape}")
        return img_array
        
    except Exception as e:
        logger.error(f"Image preprocessing error: {str(e)}")
        raise ValueError(f"Failed to preprocess image: {str(e)}")

