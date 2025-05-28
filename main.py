from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import json
import logging
import os

# Konfigurasi logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Debugging: Cetak direktori kerja dan daftar file
logger.info(f"Current working directory: {os.getcwd()}")
try:
    logger.info(f"Files in current directory: {os.listdir('.')}")
    logger.info(f"Files in models directory: {os.listdir('models')}")
except Exception as e:
    logger.error(f"Error listing directories: {str(e)}")

app = FastAPI(title="Disease Prediction API", description="API untuk memprediksi penyakit berdasarkan gejala", version="1.0")

# Fungsi untuk memuat gejala dari JSON
def load_symptoms(dataset_name):
    path = os.path.join("models", dataset_name)
    logger.info(f"Attempting to load symptoms from: {path}")
    if not os.path.isfile(path):
        logger.error(f"File not found: {path}")
        raise FileNotFoundError(f"File not found: {path}")
    with open(path, "r") as f:
        return json.load(f)

# Muat gejala
try:
    bone_symptoms = load_symptoms("bone")
    digestive_symptoms = load_symptoms("digestive")
    skin_symptoms = load_symptoms("skin")
    general_symptoms = load_symptoms("general")
    logger.info("All symptom files loaded successfully")
except Exception as e:
    logger.error(f"Error loading symptoms: {str(e)}")
    raise

# Muat model dan label encoder
try:
    bone_model = tf.keras.models.load_model(os.path.join("models", "bone", "bone_disease_model.h5"))
    digestive_model = tf.keras.models.load_model(os.path.join("models", "digestive", "digestive_disease_model.h5"))
    skin_model = tf.keras.models.load_model(os.path.join("models", "skin", "skin_disease_model.h5"))
    general_model = tf.keras.models.load_model(os.path.join("models", "general", "general_disease_model.h5"))
    logger.info("All models loaded successfully")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    raise

try:
    bone_le = joblib.load(os.path.join("models", "bone", "bone_label_encoder.pkl"))
    digestive_le = joblib.load(os.path.join("models", "digestive", "digestive_label_encoder.pkl"))
    skin_le = joblib.load(os.path.join("models", "skin", "skin_label_encoder.pkl"))
    general_le = joblib.load(os.path.join("models", "general", "general_label_encoder.pkl"))
    logger.info("All label encoders loaded successfully")
except Exception as e:
    logger.error(f"Error loading label encoders: {str(e)}")
    raise

# Struktur input
class Symptoms(BaseModel):
    symptoms: list
    model_type: str  # String: 'bone', 'digestive', 'skin', atau 'general'

# Fungsi untuk mengubah gejala menjadi vektor
def symptoms_to_vector(symptoms, all_symptoms):
    symptom_vector = [0] * len(all_symptoms)
    for symptom in symptoms:
        if symptom in all_symptoms:
            symptom_vector[all_symptoms.index(symptom)] = 1
    return np.array(symptom_vector).reshape(1, -1)

# Endpoint utama
@app.get("/")
def home():
    try:
        return {"message": "Selamat datang di Disease Prediction API 🚀"}
    except Exception as e:
        logger.error(f"Error in home endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint untuk prediksi
@app.post("/predict")
def predict_disease(request: Symptoms):
    try:
        # Validasi input
        if not request.symptoms:
            raise HTTPException(status_code=400, detail="Daftar gejala tidak boleh kosong")
        
        # Pilih model, label encoder, dan gejala berdasarkan model_type
        model_map = {
            "bone": (bone_model, bone_le, bone_symptoms),
            "digestive": (digestive_model, digestive_le, digestive_symptoms),
            "skin": (skin_model, skin_le, skin_symptoms),
            "general": (general_model, general_le, general_symptoms)
        }
        
        if request.model_type not in model_map:
            raise HTTPException(status_code=400, detail="Tipe model tidak valid. Pilih: bone, digestive, skin, atau general")
        
        model, le, symptom_list = model_map[request.model_type]
        
        # Validasi gejala
        invalid_symptoms = [s for s in request.symptoms if s not in symptom_list]
        if invalid_symptoms:
            raise HTTPException(status_code=400, detail=f"Gejala tidak valid untuk model {request.model_type}: {invalid_symptoms}")
        
        # Ubah gejala menjadi vektor
        input_vector = symptoms_to_vector(request.symptoms, symptom_list)
        
        # Prediksi
        prediction = model.predict(input_vector, verbose=0)
        predicted_class = np.argmax(prediction, axis=1)[0]
        predicted_disease = le.inverse_transform([predicted_class])[0]
        probabilities = prediction[0].tolist()
        
        # Format output probabilitas
        prob_dict = {le.classes_[i]: prob for i, prob in enumerate(probabilities)}
        
        return {
            "prediction": predicted_disease,
            "probabilities": prob_dict
        }
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
