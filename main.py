from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_model_components, symptoms_to_vector
import numpy as np
import logging

# Konfigurasi Logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Disease Prediction API",
    description="API ini memprediksi jenis penyakit berdasarkan gejala",
    version="1.0"
)

# Inisialisasi model dan data
MODEL_TYPES = ["bone", "digestive", "skin", "general"]
model_map = {}

for model_type in MODEL_TYPES:
    try:
        model, le, symptoms_dict, symptoms_keys = load_model_components("models", model_type)
        model_map[model_type] = {
            "model": model,
            "label_encoder": le,
            "symptoms_dict": symptoms_dict,
            "symptom_keys": symptoms_keys
        }
        logger.info(f"Model '{model_type}' berhasil dimuat.")
    except Exception as e:
        logger.error(f"Gagal memuat model '{model_type}': {e}")

# Model untuk request body
class Symptoms(BaseModel):
    symptoms: list[str]
    model_type: str

# Root Endpoint 
@app.get("/")
def home():
    return {"message": "Yeyy!!! Berhasil terhubung ke API Prediksi Penyakit"}

# Predict Endpoint 
@app.post("/predict")
def predict_disease(request: Symptoms):
    if request.model_type not in model_map:
        raise HTTPException(status_code=400, detail="Jenis model tidak dikenali. Gunakan: bone, digestive, skin, atau general")

    if not request.symptoms:
        raise HTTPException(status_code=400, detail="Gejala harus diisi terlebih dahulu")

    model_data = model_map[request.model_type]
    invalid_symptoms = [s for s in request.symptoms if s not in model_data["symptom_keys"]]

    if invalid_symptoms:
        raise HTTPException(status_code=400, detail=f"Gejala tidak valid: {invalid_symptoms}")

    input_vector = symptoms_to_vector(request.symptoms, model_data["symptom_keys"])
    prediction = model_data["model"].predict(np.array(input_vector), verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = model_data["label_encoder"].inverse_transform([predicted_class])[0]

    logger.info(f"Prediksi berhasil untuk model '{request.model_type}' dengan gejala: {request.symptoms} → {predicted_disease}")
    return {"prediction": predicted_disease}

# Symptoms List Endpoint 
@app.get("/symptoms/{model_type}")
def get_symptoms(model_type: str):
    if model_type not in model_map:
        raise HTTPException(status_code=400, detail="Jenis model tidak valid. Gunakan: bone, digestive, skin, atau general")
    return model_map[model_type]["symptoms_dict"]
