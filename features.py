import librosa
import numpy as np

def extract_audio_features(audio_path, sr=22050, n_mfcc=13, n_fft=2048, hop_length=512, max_duration=30):
    """
    Extrae características tradicionales de un archivo de audio
    max_duration: duración máxima en segundos para procesar (evita problemas de memoria)
    """
    try:
        # Cargar el audio con duración máxima para evitar problemas de memoria
        y, sr = librosa.load(audio_path, sr=sr, duration=max_duration)
        
        # Si el audio es muy largo, tomar solo una muestra representativa
        if len(y) > sr * max_duration:
            # Tomar el segmento del medio del audio
            start = len(y) // 2 - (sr * max_duration) // 2
            end = start + sr * max_duration
            y = y[start:end]

        # Diccionario para almacenar características
        features = {}

        # 1. CARACTERÍSTICAS ESPECTRALES
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
        features['spectral_centroid_mean'] = np.mean(spectral_centroids)
        features['spectral_centroid_std'] = np.std(spectral_centroids)
        features['spectral_centroid_median'] = np.median(spectral_centroids)

        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, hop_length=hop_length)[0]
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=hop_length)[0]
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        zcr = librosa.feature.zero_crossing_rate(y, hop_length=hop_length)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # 2. COEFICIENTES MFCC
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

        for i in range(n_mfcc):
            features[f'mfcc_{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc_{i+1}_std'] = np.std(mfccs[i])

        # 3. CARACTERÍSTICAS DE ENERGÍA
        rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        features['rms_max'] = np.max(rms)

        # 4. CARACTERÍSTICAS DE PITCH
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr, hop_length=hop_length)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)

        if len(pitch_values) > 0:
            features['pitch_mean'] = np.mean(pitch_values)
            features['pitch_std'] = np.std(pitch_values)
        else:
            features['pitch_mean'] = 0
            features['pitch_std'] = 0

        # 5. ANÁLISIS DE FRECUENCIAS ESPECÍFICAS PARA ABEJAS
        fft = np.fft.fft(y)
        magnitude = np.abs(fft)
        freqs = np.fft.fftfreq(len(fft), 1/sr)

        # Energía en bandas específicas
        low_freq_mask = (freqs >= 0) & (freqs <= 200)
        mid_freq_mask = (freqs >= 200) & (freqs <= 800)
        high_freq_mask = (freqs >= 800) & (freqs <= 2000)

        total_energy = np.sum(magnitude)
        features['energy_low_freq'] = np.sum(magnitude[low_freq_mask])
        features['energy_mid_freq'] = np.sum(magnitude[mid_freq_mask])
        features['energy_high_freq'] = np.sum(magnitude[high_freq_mask])

        features['low_freq_ratio'] = features['energy_low_freq'] / total_energy if total_energy > 0 else 0
        features['mid_freq_ratio'] = features['energy_mid_freq'] / total_energy if total_energy > 0 else 0
        features['high_freq_ratio'] = features['energy_high_freq'] / total_energy if total_energy > 0 else 0

        # Frecuencia dominante
        dominant_freq_idx = np.argmax(magnitude[:len(magnitude)//2])
        features['dominant_frequency'] = freqs[dominant_freq_idx]

        return features

    except Exception as e:
        print(f"Error procesando {audio_path}: {str(e)}")
        return None
