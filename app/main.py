import uvicorn
from pydantic import BaseModel
from app.model import model_handler
from fastapi import FastAPI, HTTPException
from app.utils import load_image_from_base64, load_image_from_url, preprocess_image

app = FastAPI()

class PredictRequest(BaseModel):
    base64_image: str = None
    image_url: str = None

@app.post("/predict")
async def predict(payload: PredictRequest):
    try:
        if payload.image_url:
            image = await load_image_from_url(payload.image_url)
        elif payload.base64_image:
            image = load_image_from_base64(payload.base64_image)
        else:
            raise HTTPException(status_code=400, detail="No image provided.")

        image_array = preprocess_image(image)
        result = model_handler.predict(image_array)
        return {"success": True, "result": result}

    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
