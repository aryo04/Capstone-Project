import os
import pandas as pd
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
    # Membuat vektor biner panjangnya sama dengan jumlah gejala keseluruhan
    vector = [0] * len(all_symptoms)
    not_found = []

    # Menandai posisi gejala yang dipilih dengan 1 dan mencatat gejala yang tidak ditemukan
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            idx = all_symptoms.index(symptom)
            vector[idx] = 1
        else:
            not_found.append(symptom)

    # Jika ada gejala yang tidak dikenal, memberi peringatan di log
    if not_found:
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"Gejala tidak ditemukan dalam daftar model: {not_found}")
    
    # mengembalikan vektor dalam bentuk list 
    return [vector]

def load_disease_info(data_path="data"):
    # Path file CSV untuk deskripsi gejala dan tindakan pencegahan
    desc_path = os.path.join(data_path, "symptom_description.csv")
    precaution_path = os.path.join(data_path, "symptom_precaution.csv")
    
    # Membaca CSV ke dataframe pandas
    description_df = pd.read_csv(desc_path)
    precaution_df = pd.read_csv(precaution_path)
    
    # Mengembalikan dua dataframe
    return description_df, precaution_df
