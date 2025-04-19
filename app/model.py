import numpy as np
from huggingface_hub import hf_hub_download
from tensorflow.keras.models import load_model

lesion_type_dict = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}


class ModelHandler:
    def __init__(self):
        model_path = hf_hub_download(
            repo_id="sanchiittt/skin_cancer", filename="skin_cancer_model.h5"
        )
        self.model = load_model(model_path)
        self.class_keys = list(lesion_type_dict.keys())

    def predict(self, image_array: np.ndarray, threshold: float = 0.2):
        if len(image_array.shape) == 3:
            image_array = np.expand_dims(image_array, axis=0)

        output_data = self.model.predict(image_array)[0]
        predicted_index = np.argmax(output_data)
        confidence = float(output_data[predicted_index])

        if confidence < threshold:
            return {
                "predicted_class": "unrelated",
                "description": "Not related to known skin lesion types.",
                "confidence": confidence,
            }

        label = self.class_keys[predicted_index]
        return {
            "predicted_class": label,
            "description": lesion_type_dict[label],
            "confidence": confidence,
        }


model_handler = ModelHandler()
