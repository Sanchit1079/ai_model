import logging
import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages the TensorFlow Lite model for inference"""
    
    def __init__(self, model_path, class_labels):
        """
        Initialize the model manager
        
        Args:
            model_path (str): Path to the TFLite model file
            class_labels (list): List of class labels
        """
        self.model_path = model_path
        self.class_labels = class_labels
        self.interpreter = None
        self.input_details = None
        self.output_details = None
    
    def load_model(self):
        """Load the TFLite model into memory"""
        try:
            # Load TFLite model
            self.interpreter = tf.lite.Interpreter(model_path=self.model_path)
            self.interpreter.allocate_tensors()
            
            # Get input and output details
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            
            logger.info(f"Model loaded with input shape: {self.input_details[0]['shape']}")
            logger.info(f"Model output shape: {self.output_details[0]['shape']}")
            
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    
    def is_model_loaded(self):
        """Check if model is loaded"""
        return self.interpreter is not None
    
    def predict(self, image_array):
        """
        Run inference on preprocessed image
        
        Args:
            image_array (numpy.ndarray): Preprocessed image as numpy array
            
        Returns:
            dict: Prediction results with class probabilities
        """
        if not self.is_model_loaded():
            raise ValueError("Model not loaded")
        
        try:
            # Set input tensor
            self.interpreter.set_tensor(
                self.input_details[0]['index'], 
                image_array
            )
            
            # Run inference
            self.interpreter.invoke()
            
            # Get output tensor
            output_data = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )
            
            # Process results
            probabilities = output_data[0].tolist()
            
            # Create result dictionary
            result = {
                "predictions": [
                    {
                        "class": self.class_labels[i],
                        "probability": round(float(prob), 4)
                    } 
                    for i, prob in enumerate(probabilities)
                ],
                "top_prediction": {
                    "class": self.class_labels[np.argmax(probabilities)],
                    "probability": round(float(max(probabilities)), 4)
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise

