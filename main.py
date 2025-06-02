from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_model_components, symptoms_to_vector
import numpy as np
import logging

# Logger untuk catat info dan error
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Disease Prediction API", description="API prediksi penyakit berdasarkan gejala", version="1.0")

MODEL_TYPES = ["bone", "digestive", "skin", "general"]
model_map = {}

# Load model dan data gejala tiap tipe model
for model_type in MODEL_TYPES:
    try:
        model, le, symptoms_dict, symptoms_keys = load_model_components("models", model_type)
        model_map[model_type] = {"model": model, "label_encoder": le, "symptoms_dict": symptoms_dict, "symptom_keys": symptoms_keys}
        logger.info(f"Model '{model_type}' berhasil dimuat.")
    except Exception as e:
        logger.error(f"Gagal memuat model '{model_type}': {e}")

class Symptoms(BaseModel):
    symptoms: list[str]
    model_type: str

@app.get("/")
def home():
    return {"message": "Yeyy!!! Berhasil terhubung ke API Prediksi Penyakit"}

@app.post("/predict")
def predict_disease(request: Symptoms):
    # Validasi tipe model dan gejala input
    if request.model_type not in model_map:
        raise HTTPException(400, "Jenis model tidak dikenali. Gunakan: bone, digestive, skin, atau general")
    if not request.symptoms:
        raise HTTPException(400, "Gejala harus diisi terlebih dahulu")
    invalid = [s for s in request.symptoms if s not in model_map[request.model_type]["symptom_keys"]]
    if invalid:
        raise HTTPException(400, f"Gejala tidak valid: {invalid}")

    # Mengubah gejala ke vektor input model
    input_vector = symptoms_to_vector(request.symptoms, model_map[request.model_type]["symptom_keys"])

    # Memprediksi dan decode hasil
    prediction = model_map[request.model_type]["model"].predict(np.array(input_vector), verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = model_map[request.model_type]["label_encoder"].inverse_transform([predicted_class])[0]

    logger.info(f"Prediksi model '{request.model_type}' gejala: {request.symptoms} → {predicted_disease}")
    return {"prediction": predicted_disease}

@app.get("/symptoms/{model_type}")
def get_symptoms(model_type: str):
    # Validasi tipe model dan return daftar gejala
    if model_type not in model_map:
        raise HTTPException(400, "Jenis model tidak valid. Gunakan: bone, digestive, skin, atau general")
    return model_map[model_type]["symptoms_dict"]
