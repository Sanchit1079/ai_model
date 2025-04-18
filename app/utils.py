import httpx
import base64
import numpy as np
from PIL import Image
from io import BytesIO

async def load_image_from_url(url: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")

def load_image_from_base64(base64_str: str):
    image_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(image_data)).convert("RGB")

def preprocess_image(image: Image.Image, target_size=(128, 128)):
    image = image.resize(target_size)
    image_array = np.array(image) / 255.0
    return np.expand_dims(image_array.astype(np.float32), axis=0)
