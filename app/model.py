import numpy as np
import tensorflow as tf

MODEL_PATH = "models/model.tflite"

lesion_type_dict = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions ",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}


class ModelHandler:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
        self.interpreter.allocate_tensors()
        self.class_keys = list(lesion_type_dict.keys())

    def predict(self, image_array: np.ndarray):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        self.interpreter.set_tensor(input_details[0]["index"], image_array)
        self.interpreter.invoke()
        output_data = self.interpreter.get_tensor(output_details[0]["index"])
        predicted_index = np.argmax(output_data)
        label = self.class_keys[predicted_index]
        return {
            "predicted_class": label,
            "description": lesion_type_dict[label],
            "confidence": float(output_data[0][predicted_index]),
        }


model_handler = ModelHandler()
