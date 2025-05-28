from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import joblib
import numpy as np
import json

app = FastAPI(title="Disease Prediction API", description="API untuk memprediksi penyakit berdasarkan gejala", version="1.0")

# Muat daftar gejala dari file JSON
with open("models/bone/bone_symptoms.json", "r") as f:
    bone_symptoms = json.load(f)
with open("models/digestive/digestive_symptoms.json", "r") as f:
    digestive_symptoms = json.load(f)
with open("models/skin/skin_symptoms.json", "r") as f:
    skin_symptoms = json.load(f)
with open("models/general/general_symptoms.json", "r") as f:
    general_symptoms = json.load(f)

# Muat model dan label encoder
bone_model = tf.keras.models.load_model("models/bone/bone_disease_model.h5")
digestive_model = tf.keras.models.load_model("models/digestive/digestive_disease_model.h5")
skin_model = tf.keras.models.load_model("models/skin/skin_disease_model.h5")
general_model = tf.keras.models.load_model("models/general/general_disease_model.h5")

bone_le = joblib.load("models/bone/bone_label_encoder.pkl")
digestive_le = joblib.load("models/digestive/digestive_label_encoder.pkl")
skin_le = joblib.load("models/skin/skin_label_encoder.pkl")
general_le = joblib.load("models/general/general_label_encoder.pkl")

# Struktur input
class Symptoms(BaseModel):
    symptoms: list
    model_type: str  # Pilih model: 'bone', 'digestive', 'skin', atau 'general'

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
    return {"message": "Selamat datang di Disease Prediction API 🚀"}

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
        raise HTTPException(status_code=500, detail=str(e))
