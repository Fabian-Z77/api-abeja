from flask import Flask, request, jsonify
import os
from predictor import predict_new_audio

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/predict', methods=['POST'])
def predict_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400

    file = request.files['file']
    file_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(file_path)

    # Realiza predicción
    model_path = "bee_queen_detector.pkl"
    scaler_path = "bee_queen_detector_scaler.pkl"
    prediction, probability = predict_new_audio(file_path, model_path, scaler_path)

    if prediction is None:
        return jsonify({"error": "No se pudo procesar el audio"}), 500

    return jsonify({
        "prediccion": "Con reina" if prediction == 1 else "Sin reina",
        "probabilidad_con_reina": round(probability[1], 3),
        "probabilidad_sin_reina": round(probability[0], 3)
    })

if __name__ == "__main__":
    app.run(debug=True, port=5000)
