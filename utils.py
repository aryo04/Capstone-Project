import os
import tensorflow as tf
import joblib
import json
from typing import Tuple

def load_model_components(base_path: str, model_type: str) -> Tuple:
    path = os.path.join(base_path, model_type)
    
    model = tf.keras.models.load_model(os.path.join(path, f"{model_type}_disease_model.h5"))
    label_encoder = joblib.load(os.path.join(path, f"{model_type}_label_encoder.pkl"))
    
    with open(os.path.join(path, f"{model_type}_symptoms.json"), "r") as f:
        symptoms_dict = json.load(f)

    symptoms_keys = list(symptoms_dict.keys())
    return model, label_encoder, symptoms_dict, symptoms_keys


def symptoms_to_vector(symptoms: list, all_symptoms: list):
    return [[1 if symptom in symptoms else 0 for symptom in all_symptoms]]
