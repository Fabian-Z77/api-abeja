import librosa
import numpy as np
import signal
import time

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timeout en extracción de características")

def extract_audio_features(audio_path, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, max_duration=30, timeout_seconds=120):
    """
    Extrae características tradicionales de un archivo de audio
    max_duration: duración máxima en segundos para procesar (evita problemas de memoria)
    timeout_seconds: timeout máximo para evitar que se cuelgue
    """
    
    # Configurar timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        print(f"🎵 Iniciando extracción de características para: {audio_path}")
        start_time = time.time()
        
        # Cargar el audio con duración máxima para evitar problemas de memoria
        print("📂 Cargando archivo de audio...")
        y, sr = librosa.load(audio_path, sr=sr, duration=max_duration)
        print(f"✅ Audio cargado: {len(y)} samples, {len(y)/sr:.2f} segundos")
        
        # Si el audio es muy largo, tomar solo una muestra representativa
        if len(y) > sr * max_duration:
            # Tomar el segmento del medio del audio
            start = len(y) // 2 - (sr * max_duration) // 2
            end = start + sr * max_duration
            y = y[start:end]
            print(f"🎯 Audio recortado a {len(y)/sr:.2f} segundos")

        # Lista para almacenar características en orden específico
        feature_vector = []

        print("📊 Extrayendo características espectrales...")
        # 1. CARACTERÍSTICAS ESPECTRALES
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        feature_vector.extend([
            np.mean(spectral_centroids),
            np.std(spectral_centroids),
            np.median(spectral_centroids)
        ])

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        feature_vector.extend([
            np.mean(spectral_bandwidth),
            np.std(spectral_bandwidth)
        ])

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        feature_vector.extend([
            np.mean(spectral_rolloff),
            np.std(spectral_rolloff)
        ])

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        feature_vector.extend([
            np.mean(zcr),
            np.std(zcr)
        ])

        print("🎼 Extrayendo coeficientes MFCC...")
        # 2. COEFICIENTES MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

        for i in range(n_mfcc):
            feature_vector.extend([
                np.mean(mfccs[i]),
                np.std(mfccs[i])
            ])

        print("⚡ Extrayendo características de energía...")
        # 3. CARACTERÍSTICAS DE ENERGÍA
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        feature_vector.extend([
            np.mean(rms),
            np.std(rms),
            np.max(rms)
        ])

        print("🎵 Analizando pitch...")
        # 4. CARACTERÍSTICAS DE PITCH (simplificado para ser más rápido)
        try:
            # Usar método más rápido para pitch
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length, threshold=0.1)
            pitch_values = []
            
            # Solo procesar cada 10mo frame para acelerar
            for t in range(0, pitches.shape[1], 10):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)

            if len(pitch_values) > 0:
                feature_vector.extend([
                    np.mean(pitch_values),
                    np.std(pitch_values)
                ])
            else:
                feature_vector.extend([0, 0])
        except Exception as pitch_error:
            print(f"⚠️ Warning en pitch analysis: {pitch_error}")
            feature_vector.extend([0, 0])

        print("🔊 Analizando frecuencias específicas...")
        # 5. ANÁLISIS DE FRECUENCIAS ESPECÍFICAS PARA ABEJAS
        # Usar ventana más pequeña para acelerar FFT
        window_size = min(len(y), sr * 5)  # máximo 5 segundos
        y_window = y[:window_size]
        
        fft = np.fft.fft(y_window)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/sr)

        # Energía en bandas específicas
        low_freq_mask = (freqs >= 0) & (freqs <= 200)
        mid_freq_mask = (freqs >= 200) & (freqs <= 800)
        high_freq_mask = (freqs >= 800) & (freqs <= 2000)

        total_energy = np.sum(magnitude)
        energy_low = np.sum(magnitude[low_freq_mask])
        energy_mid = np.sum(magnitude[mid_freq_mask])
        energy_high = np.sum(magnitude[high_freq_mask])

        feature_vector.extend([
            energy_low,
            energy_mid,
            energy_high,
            energy_low / total_energy if total_energy > 0 else 0,
            energy_mid / total_energy if total_energy > 0 else 0,
            energy_high / total_energy if total_energy > 0 else 0
        ])

        # Frecuencia dominante
        dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
        dominant_frequency = abs(freqs[dominant_freq_idx])
        feature_vector.append(dominant_frequency)

        # Convertir a numpy array
        features_array = np.array(feature_vector, dtype=np.float32)
        
        # Verificar que no hay NaN o infinitos
        if np.any(np.isnan(features_array)) or np.any(np.isinf(features_array)):
            print("⚠️ Warning: Se encontraron NaN o infinitos, reemplazando con 0")
            features_array = np.nan_to_num(features_array, nan=0.0, posinf=0.0, neginf=0.0)

        end_time = time.time()
        print(f"✅ Extracción completada en {end_time - start_time:.2f}s")
        print(f"📏 Vector de características: {len(features_array)} dimensiones")
        print(f"🔍 Primeras 5 características: {features_array[:5]}")

        # Desactivar timeout
        signal.alarm(0)
        
        return features_array

    except TimeoutError:
        print(f"❌ Timeout: La extracción de características tomó más de {timeout_seconds} segundos")
        signal.alarm(0)
        return None
    except Exception as e:
        print(f"❌ Error procesando {audio_path}: {str(e)}")
        print(f"🔍 Tipo de error: {type(e).__name__}")
        signal.alarm(0)
        return None

def extract_audio_features_dict(audio_path, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, max_duration=30):
    """
    Versión que devuelve diccionario (para compatibilidad)
    """
    features_array = extract_audio_features(audio_path, sr, n_mfcc, n_fft, hop_length, max_duration)
    
    if features_array is None:
        return None
    
    # Convertir array a diccionario con nombres de características
    feature_names = [
        'spectral_centroid_mean', 'spectral_centroid_std', 'spectral_centroid_median',
        'spectral_bandwidth_mean', 'spectral_bandwidth_std',
        'spectral_rolloff_mean', 'spectral_rolloff_std',
        'zcr_mean', 'zcr_std'
    ]
    
    # MFCC features
    for i in range(n_mfcc):
        feature_names.extend([f'mfcc_{i+1}_mean', f'mfcc_{i+1}_std'])
    
    feature_names.extend([
        'rms_mean', 'rms_std', 'rms_max',
        'pitch_mean', 'pitch_std',
        'energy_low_freq', 'energy_mid_freq', 'energy_high_freq',
        'low_freq_ratio', 'mid_freq_ratio', 'high_freq_ratio',
        'dominant_frequency'
    ])
    
    features_dict = {}
    for i, name in enumerate(feature_names):
        if i < len(features_array):
            features_dict[name] = features_array[i]
    
    return features_dict