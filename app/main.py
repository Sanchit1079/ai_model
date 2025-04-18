import os
import uvicorn
from datetime import datetime
from pydantic import BaseModel
from pymongo import MongoClient
from dotenv import load_dotenv
from app.model import model_handler
from fastapi import FastAPI, HTTPException
from app.utils import load_image_from_base64, load_image_from_url, preprocess_image

load_dotenv()

app = FastAPI()

MONGO_URI = os.environ.get("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["skin_cancer_app"]
collection = db["history"]


class PredictRequest(BaseModel):
    base64_image: str = None
    image_url: str = None
    patient_name: str = None
    patient_age: int = None
    contact_no: int = (None,)
    scar_duration: int = None


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

        # Store in MongoDB
        record = {
            "patient_name": payload.patient_name,
            "patient_age": payload.patient_age,
            "contact_no": payload.contact_no,
            "scar_duration": payload.scar_duration,
            "result": result,
            "timestamp": datetime.utcnow(),
        }
        collection.insert_one(record)

        return {"success": True, "result": result}
    except Exception as e:
        return {"success": False, "error": str(e)}


@app.get("/history")
def get_history():
    docs = list(collection.find({}, {"_id": 0}))
    return {"success": True, "records": docs}


if __name__ == "__main__":
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)
