from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from PIL import Image
import io
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


app = FastAPI(
    title="Image Classification API",
    description="API for classifying images using CNN model"
)

# Define your class labels
class_labels = ["glaciers", "mountains", "forest", "buildings", "street", "sea"]

# Define response model
class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: dict[str, float]

    class Config:
        schema_extra = {
            "example": {
                "predicted_class": "mountains",
                "confidence": 0.95,
                "probabilities": {
                    "glaciers": 0.01,
                    "mountains": 0.95,
                    "forest": 0.01,
                    "buildings": 0.01,
                    "street": 0.01,
                    "sea": 0.01
                }
            }
        }

# Load your saved model
try:
    model = load_model('model/cnn_model.h5')
except:
    raise Exception("Model file not found. Ensure the model is saved and in the correct location.")

def preprocess_image(image):
    image = image.resize((150, 150))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/predict", 
    response_model=PredictionResponse,
    summary="Predict image class",
    description="Upload an image to classify it into one of these categories: glaciers, mountains, forest, buildings, street, sea")
async def predict_image(
    file: UploadFile = File(
        ...,
        description="The image file to classify. Must be in JPG, PNG, or JPEG format."
    )
):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File provided is not an image")
        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_class])
        
        return PredictionResponse(
            predicted_class=class_labels[predicted_class],
            confidence=confidence,
            probabilities={
                class_labels[i]: float(prob) 
                for i, prob in enumerate(predictions[0])
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    return {
        "message": "Image Classification API",
        "model": "CNN Classification Model",
        "input_shape": "150x150 RGB images",
        "available_classes": class_labels,
        "endpoints": {
            "predict": "/predict - POST (requires image file upload)",
            "docs": "/docs - API documentation"
        }
    }