import os
import tensorflow as tf
import joblib
import json
from typing import Tuple

def load_model_components(base_path: str, model_type: str) -> Tuple:
    # Menggabungkan path dasar dengan tipe model untuk akses file
    path = os.path.join(base_path, model_type)
    
    # Load model Keras yang sudah disimpan
    model = tf.keras.models.load_model(os.path.join(path, f"{model_type}_disease_model.h5"))
    # Load label encoder yang disimpan dengan joblib
    label_encoder = joblib.load(os.path.join(path, f"{model_type}_label_encoder.pkl"))
    
    # Load gejala dari file JSON
    with open(os.path.join(path, f"{model_type}_symptoms.json"), "r") as f:
        symptoms_dict = json.load(f)

    # Membuat list key gejala untuk referensi
    symptoms_keys = list(symptoms_dict.keys())
    return model, label_encoder, symptoms_dict, symptoms_keys

def symptoms_to_vector(selected_symptoms, all_symptoms):
    # Membuat vektor binary dengan panjang sama seperti total gejala (all_symptoms)
    vector = [0] * len(all_symptoms)
    
    # Menandai posisi gejala yang dipilih user dengan 1
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            vector[idx] = 1
            
    # Return dalam bentuk list of list (karena biasanya input model berbentuk batch)
    return [vector]
