from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utils import load_model_components, symptoms_to_vector, load_disease_info
import numpy as np
import logging
import pandas as pd

# Memanggil fungsi load_disease_info untuk membaca data deskripsi gejala dan tindakan pencegahan
description_df, precaution_df = load_disease_info()

# Setup logger untuk mencatat informasi dan error saat runtime
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Inisialisasi FastAPI dengan metadata dasar
app = FastAPI(title="Disease Prediction API", description="API prediksi penyakit berdasarkan gejala", version="1.0")

# Daftar tipe model yang akan dimuat
MODEL_TYPES = ["bone", "digestive", "skin", "general"]
model_map = {}

# Memuat model, label encoder, dan data gejala untuk tiap tipe model
for model_type in MODEL_TYPES:
    try:
        model, le, symptoms_dict, symptoms_keys = load_model_components("models", model_type)
        model_map[model_type] = {"model": model, "label_encoder": le, "symptoms_dict": symptoms_dict, "symptom_keys": symptoms_keys}
        logger.info(f"Model '{model_type}' berhasil dimuat.")
    except Exception as e:
        logger.error(f"Gagal memuat model '{model_type}': {e}")

# Mendefinisikan struktur data input request dengan Pydantic
class Symptoms(BaseModel):
    symptoms: list[str]
    model_type: str

# Endpoint utama untuk cek API hidup
@app.get("/")
def home():
    return {"message": "Yeyy!!! Berhasil terhubung ke API Prediksi Penyakit"}

# Endpoint prediksi penyakit berdasarkan input gejala dan tipe model
@app.post("/predict")
def predict_disease(request: Symptoms):
    # Memvalidasi model tersedia dan gejala tidak kosong
    if request.model_type not in model_map or not model_map[request.model_type]:
        raise HTTPException(500, f"Model untuk tipe '{request.model_type}' tidak tersedia.")
    if not request.symptoms:
        raise HTTPException(400, "Gejala harus diisi terlebih dahulu")
    if len(request.symptoms) < 2:
        return {
        "prediction": "Tidak ada penyakit",
        "description": "-",
        "precautions": ["-"]
    }
    # Memvalidasi gejala input sesuai daftar gejala model
    invalid = [s for s in request.symptoms if s not in model_map[request.model_type]["symptom_keys"]]
    if invalid:
        raise HTTPException(400, f"Gejala tidak valid: {invalid}")

    # Mengkonversi gejala ke bentuk vektor input untuk model
    input_vector = symptoms_to_vector(request.symptoms, model_map[request.model_type]["symptom_keys"])

    # Memprediksi kelas penyakit dan decode hasil
    prediction = model_map[request.model_type]["model"].predict(np.array(input_vector), verbose=0)
    predicted_class = np.argmax(prediction, axis=1)[0]
    predicted_disease = model_map[request.model_type]["label_encoder"].inverse_transform([predicted_class])[0]

    # Mengambil deskripsi penyakit dari dataset
    desc_row = description_df[description_df["Disease"] == predicted_disease]
    description = desc_row["Description"].values[0] if not desc_row.empty else "Tidak tersedia"

    # Mengambil daftar tindakan pencegahan terkait penyakit
    precaution_row = precaution_df[precaution_df["Disease"] == predicted_disease]
    precautions = []
    if not precaution_row.empty:
        for i in range(1, 5):
            col = f"Precaution_{i}"
            if col in precaution_row.columns and pd.notna(precaution_row[col].values[0]):
                precautions.append(precaution_row[col].values[0])
    else:
        precautions = ["Tidak tersedia"]

    logger.info(f"Prediksi model '{request.model_type}' gejala: {request.symptoms} → {predicted_disease}")

    # Response berisi hasil prediksi dan info tambahan
    return {
        "prediction": predicted_disease,
        "description": description,
        "precautions": precautions
    }

# Endpoint untuk mengambil daftar gejala berdasarkan tipe model
@app.get("/symptoms/{model_type}")
def get_symptoms(model_type: str):
    if model_type not in model_map or not model_map[model_type]:
        raise HTTPException(400, f"Jenis model '{model_type}' tidak valid atau belum dimuat")
    return model_map[model_type]["symptoms_dict"]

# Endpoint untuk mengambil daftar model yang tersedia
@app.get("/models")
def get_models():
    available = [k for k, v in model_map.items() if v]  # Hanya model yang berhasil dimuat
    return {"available_models": available}
