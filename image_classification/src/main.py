from pathlib import Path
import io

import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from pydantic import BaseModel
from tensorflow.keras.models import load_model

app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using CNN model",
)

STATIC_DIR = Path(__file__).parent / "static"

class_labels = ["glaciers", "mountains", "forest", "buildings", "street", "sea"]


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    model_config = {
        "json_schema_extra": {
            "example": {
                "predicted_class": "mountains",
                "confidence": 0.95,
                "probabilities": {
                    "glaciers": 0.01,
                    "mountains": 0.95,
                    "forest": 0.01,
                    "buildings": 0.01,
                    "street": 0.01,
                    "sea": 0.01,
                },
            }
        }
    }


try:
    model = load_model("model/cnn_model.h5")
except Exception:
    raise Exception("Model file not found. Ensure the model is saved and in the correct location.")


def preprocess_image(image: Image.Image):
    image = image.convert("RGB").resize((150, 150))  # handles RGBA, grayscale, palette
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict image class",
    description="Upload an image to classify it into one of these categories: glaciers, mountains, forest, buildings, street, sea",
)
async def predict_image(
    file: UploadFile = File(..., description="The image file to classify. Must be in JPG, PNG, or JPEG format.")
):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File provided is not an image")
    try:
        image = Image.open(io.BytesIO(await file.read()))
        predictions = model.predict(preprocess_image(image), verbose=0)[0]
        top = int(np.argmax(predictions))
        return PredictionResponse(
            predicted_class=class_labels[top],
            confidence=float(predictions[top]),
            probabilities={class_labels[i]: float(p) for i, p in enumerate(predictions)},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api", summary="API info")
async def api_info():
    return {
        "message": "Image Classification API",
        "model": "CNN Classification Model",
        "input_shape": "150x150 RGB images",
        "available_classes": class_labels,
        "endpoints": {
            "predict": "/predict - POST (requires image file upload)",
            "docs": "/docs - API documentation",
        },
    }
