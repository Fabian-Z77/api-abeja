import os
import pickle
import numpy as np

# Configuraciones de entorno para optimizar numba y librosa
os.environ['NUMBA_CACHE_DIR'] = '/tmp/numba_cache'
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'  # Limitar threads para evitar sobrecarga

print("🔄 Inicializando aplicación...")

# Pre-cargar librosa para compilar funciones numba
print("📦 Cargando librosa y compilando numba...")
import librosa
import librosa.feature

# Compilar funciones de numba con audio dummy para evitar delays en primera request
try:
    dummy_audio = np.random.random(22050)  # 1 segundo de audio
    _ = librosa.stft(dummy_audio)
    _ = librosa.feature.mfcc(y=dummy_audio, sr=22050, n_mfcc=13)
    print("✅ Librosa y numba compilados correctamente")
except Exception as e:
    print(f"⚠️ Warning al compilar numba: {e}")

from flask import Flask, request, jsonify
from predictor import predict_new_audio

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variables globales para modelos (se cargan una sola vez)
model = None
scaler = None

def load_models():
    """Carga los modelos una sola vez al iniciar la aplicación"""
    global model, scaler
    
    model_path = "bee_queen_detector.pkl"
    scaler_path = "bee_queen_detector_scaler.pkl"
    
    try:
        print("🤖 Cargando modelo de machine learning...")
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        print("📊 Cargando scaler...")
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
            
        print("✅ Modelos cargados correctamente")
        return True
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return False

# Cargar modelos al inicializar
models_loaded = load_models()

@app.route('/')
def health_check():
    """Endpoint de health check"""
    status = "✅ healthy" if models_loaded else "❌ unhealthy"
    return jsonify({
        "status": status,
        "message": "Bee Queen Detector API",
        "version": "1.0",
        "models_loaded": models_loaded
    })

@app.route('/health')
def health():
    """Endpoint adicional de health para monitoring"""
    return jsonify({
        "status": "healthy" if models_loaded else "unhealthy",
        "models_loaded": models_loaded
    })

@app.route('/predict', methods=['POST'])
def predict_audio():
    """Endpoint principal para predicción de audio"""
    
    if not models_loaded:
        return jsonify({"error": "Modelos no están cargados correctamente"}), 500
    
    if 'file' not in request.files:
        return jsonify({"error": "No se envió archivo"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No se seleccionó archivo"}), 400
    
    # Validar tipo de archivo
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        return jsonify({
            "error": f"Tipo de archivo no soportado. Use: {', '.join(allowed_extensions)}"
        }), 400

    try:
        # Guardar archivo
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        
        print(f"🎵 Procesando archivo: {file.filename}")

        # Realizar predicción usando modelos globales
        model_path = "bee_queen_detector.pkl"
        scaler_path = "bee_queen_detector_scaler.pkl"
        prediction, probability = predict_new_audio(file_path, model_path, scaler_path)

        # Limpiar archivo temporal
        try:
            os.remove(file_path)
        except:
            pass  # No crítico si no se puede eliminar

        if prediction is None:
            return jsonify({"error": "No se pudo procesar el audio"}), 500

        result = {
            "prediccion": "Con reina" if prediction == 1 else "Sin reina",
            "probabilidad_con_reina": round(float(probability[1]), 3),
            "probabilidad_sin_reina": round(float(probability[0]), 3),
            "confianza": round(float(max(probability)), 3),
            "archivo_procesado": file.filename
        }
        
        print(f"✅ Predicción completada: {result['prediccion']}")
        return jsonify(result)

    except Exception as e:
        print(f"❌ Error en predicción: {str(e)}")
        
        # Limpiar archivo en caso de error
        try:
            if 'file_path' in locals():
                os.remove(file_path)
        except:
            pass
            
        return jsonify({
            "error": "Error interno del servidor al procesar audio",
            "details": str(e)
        }), 500

@app.errorhandler(413)
def too_large(e):
    """Manejo de archivos muy grandes"""
    return jsonify({
        "error": "El archivo es demasiado grande. Máximo permitido: 16MB"
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Manejo de errores internos"""
    return jsonify({
        "error": "Error interno del servidor"
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)