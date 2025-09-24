import joblib
import numpy as np
from features import extract_audio_features  # asegúrate de tener esta función en features.py

def predict_new_audio(audio_path, model_path, scaler_path):

    # Cargar modelo y scaler
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Extraer características
    features = extract_audio_features(audio_path)
    if features is None:
        return None, None

    # Convertir a array y normalizar
    feature_values = np.array(list(features.values())).reshape(1, -1)
    feature_values_scaled = scaler.transform(feature_values)

    # Predecir
    prediction = model.predict(feature_values_scaled)[0]
    probability = model.predict_proba(feature_values_scaled)[0]

    return prediction, probability