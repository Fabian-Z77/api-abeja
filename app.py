import os
import pickle
import numpy as np
import uuid
import json
import time
import threading
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

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

from predictor import predict_new_audio

app = Flask(__name__)
CORS(app)  # Habilitar CORS para todas las rutas
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Variables globales para modelos (se cargan una sola vez)
model = None
scaler = None

# Diccionario para almacenar el progreso de cada tarea
progress_store = {}

import pickle
import joblib

def load_models():
    """Carga los modelos una sola vez al iniciar la aplicación"""
    global model, scaler
    
    model_path = "bee_queen_detector.pkl"
    scaler_path = "bee_queen_detector_scaler.pkl"
    
    try:
        print("🤖 Cargando modelo de machine learning...")
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception:
            # Si falla con pickle, probar con joblib
            model = joblib.load(model_path)
        
        print("📊 Cargando scaler...")
        try:
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        except Exception:
            scaler = joblib.load(scaler_path)
            
        print("✅ Modelos cargados correctamente")
        return True
    except Exception as e:
        print(f"❌ Error cargando modelos: {e}")
        return False


def update_progress(task_id, stage, progress):
    """Actualiza el progreso de una tarea específica"""
    progress_store[task_id] = {
        'stage': stage,
        'progress': progress,
        'timestamp': time.time()
    }
    print(f"📊 Task {task_id[:8]}: {stage} ({progress}%)")

def process_audio_with_progress(file_path, task_id):
    """Procesa el audio y actualiza el progreso en tiempo real"""
    try:
        update_progress(task_id, "Preparando archivo...", 10)
        
        # Verificar que el archivo existe
        if not os.path.exists(file_path):
            raise Exception("Archivo no encontrado")
        
        update_progress(task_id, "Extrayendo características de audio...", 30)
        
        # Usar tu función existente de extracción de características
        from features import extract_audio_features
        features = extract_audio_features(file_path)
        
        if features is None:
            raise Exception("No se pudieron extraer características del audio")
        
        print(f"🔍 Debug - Features type: {type(features)}, shape: {getattr(features, 'shape', 'No shape')}")
        print(f"🔍 Debug - Features content (first 5): {features[:5] if hasattr(features, '__getitem__') else features}")
        
        update_progress(task_id, "Normalizando datos...", 60)
        
        # Normalizar características usando el scaler global
        # Asegurar que features sea un array 2D
        if hasattr(features, 'shape'):
            if len(features.shape) == 1:
                features = features.reshape(1, -1)
        else:
            # Si features no es numpy array, convertirlo
            features = np.array(features).reshape(1, -1)
        
        features_scaled = scaler.transform(features)
        print(f"🔍 Debug - Features scaled shape: {features_scaled.shape}")
        
        update_progress(task_id, "Procesando con modelo de IA...", 80)
        
        # Hacer predicción usando el modelo global
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0]
        
        print(f"🔍 Debug - Raw prediction: {prediction}")
        print(f"🔍 Debug - Raw probability: {probability}")
        print(f"🔍 Debug - Probability type: {type(probability)}")
        print(f"🔍 Debug - Probability shape: {getattr(probability, 'shape', 'No shape')}")
        
        update_progress(task_id, "Finalizando análisis...", 95)
        
        # Versión más segura con manejo de errores para probabilidades
        try:
            # Convertir probabilidades a lista si es numpy array
            if hasattr(probability, 'tolist'):
                prob_list = probability.tolist()
            else:
                prob_list = list(probability)
            
            print(f"🔍 Debug - Probability list: {prob_list}")
            
            # Extraer probabilidades de manera segura
            prob_sin_reina = float(prob_list[0]) if len(prob_list) > 0 else 0.0
            prob_con_reina = float(prob_list[1]) if len(prob_list) > 1 else 0.0
            confianza = float(max(prob_list)) if len(prob_list) > 0 else 0.0
            
            result = {
                "prediccion": "Con reina" if prediction == 1 else "Sin reina",
                "probabilidad_con_reina": round(prob_con_reina, 3),
                "probabilidad_sin_reina": round(prob_sin_reina, 3),
                "confianza": round(confianza, 3),
                "archivo_procesado": os.path.basename(file_path)
            }
            
            print(f"✅ Debug - Final result: {result}")
            
        except Exception as prob_error:
            print(f"❌ Error procesando probabilidades: {prob_error}")
            print(f"🔍 Probability original: {probability}")
            
            # Fallback: resultado básico sin probabilidades detalladas
            result = {
                "prediccion": "Con reina" if prediction == 1 else "Sin reina",
                "probabilidad_con_reina": 0.5,  # Default fallback
                "probabilidad_sin_reina": 0.5,  # Default fallback
                "confianza": 0.5,
                "archivo_procesado": os.path.basename(file_path),
                "warning": f"Error procesando probabilidades: {str(prob_error)}"
            }
        
        update_progress(task_id, "¡Análisis completado!", 100)
        progress_store[task_id]['result'] = result
        progress_store[task_id]['completed'] = True
        
        print(f"✅ Task {task_id[:8]} completada: {result['prediccion']}")
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error en task {task_id[:8]}: {error_msg}")
        print(f"🔍 Error traceback:", exc_info=True)
        progress_store[task_id]['error'] = error_msg
        progress_store[task_id]['completed'] = True
    
    finally:
        # Limpiar archivo temporal
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"🧹 Archivo temporal eliminado: {file_path}")
        except Exception as e:
            print(f"⚠️ No se pudo eliminar archivo temporal: {e}")

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
        "models_loaded": models_loaded,
        "active_tasks": len(progress_store)
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
    """Inicia el procesamiento de audio y retorna un task_id"""
    
    print(f"📨 Recibida request POST a /predict")
    print(f"📁 Files en request: {list(request.files.keys())}")
    print(f"📋 Form data: {list(request.form.keys())}")
    
    if not models_loaded:
        error_msg = "Modelos no están cargados correctamente"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 500
    
    if 'file' not in request.files:
        error_msg = "No se envió archivo"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 400

    file = request.files['file']
    print(f"📎 Archivo recibido: {file.filename}")
    
    if file.filename == '':
        error_msg = "No se seleccionó archivo"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 400
    
    # Validar tipo de archivo
    allowed_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    file_ext = os.path.splitext(file.filename)[1].lower()
    
    if file_ext not in allowed_extensions:
        error_msg = f"Tipo de archivo no soportado. Use: {', '.join(allowed_extensions)}"
        print(f"❌ {error_msg}")
        return jsonify({"error": error_msg}), 400

    # Generar ID único para esta tarea
    task_id = str(uuid.uuid4())
    print(f"🆔 Generado task_id: {task_id}")
    
    try:
        # Guardar archivo con nombre único
        safe_filename = f"{task_id}_{file.filename}"
        file_path = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(file_path)
        print(f"💾 Archivo guardado en: {file_path}")
        
        print(f"🎵 Iniciando procesamiento de {file.filename} (Task: {task_id[:8]})")
        
        # Inicializar progreso
        update_progress(task_id, "Iniciando análisis...", 5)
        
        # Iniciar procesamiento en hilo separado
        thread = threading.Thread(
            target=process_audio_with_progress, 
            args=(file_path, task_id),
            daemon=True  # El hilo se cierra cuando se cierra la app
        )
        thread.start()
        print(f"🧵 Hilo de procesamiento iniciado para task {task_id[:8]}")
        
        response_data = {
            "task_id": task_id,
            "message": "Procesamiento iniciado",
            "progress_url": f"/progress/{task_id}",
            "filename": file.filename,
        }
        print(f"✅ Enviando respuesta: {response_data}")
        
        return jsonify(response_data), 200
        
    except Exception as e:
        error_msg = f"Error al procesar: {str(e)}"
        print(f"❌ {error_msg}")
        print(f"🔍 Traceback completo: ", exc_info=True)
        return jsonify({"error": error_msg}), 500

@app.route('/progress/<task_id>', methods=['GET'])
def get_progress_polling(task_id):
    """Endpoint para polling (obtener progreso vía GET)"""
    print(f"📊 Solicitando progreso para task: {task_id[:8]}")
    
    if task_id not in progress_store:
        print(f"❌ Task {task_id[:8]} no encontrado")
        return jsonify({"error": "Task not found"}), 404
    
    progress_data = progress_store[task_id].copy()
    print(f"📊 Enviando progreso para task {task_id[:8]}: {progress_data.get('stage', 'Unknown')} - {progress_data.get('progress', 0)}%")
    
    # Si la tarea está completa, programar limpieza después de 60 segundos
    if progress_data.get('completed', False):
        def cleanup():
            progress_store.pop(task_id, None)
            print(f"🧹 Limpieza de task {task_id[:8]}")
        
        timer = threading.Timer(60, cleanup)
        timer.start()
    
    return jsonify(progress_data)

@app.errorhandler(413)
def too_large(e):
    """Manejo de archivos muy grandes"""
    return jsonify({
        "error": "El archivo es demasiado grande. Máximo permitido: 16MB"
    }), 413

@app.errorhandler(500)
def internal_error(e):
    """Manejo de errores internos"""
    print(f"❌ Error interno del servidor: {e}")
    return jsonify({
        "error": "Error interno del servidor"
    }), 500

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Iniciando servidor en puerto {port}")
    app.run(host='0.0.0.0', port=port, debug=False)