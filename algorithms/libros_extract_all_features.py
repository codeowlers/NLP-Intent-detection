from scipy.stats import skew, kurtosis
import noisereduce as nr
import librosa
import numpy as np


def extract_all_features(df):
    # Extract features for each audio
    for index, row in df.iterrows():
        y_untrimmed, sr = librosa.load(row["path"], mono=True)
        y_trimmed, i = librosa.effects.trim(y_untrimmed, top_db=30, frame_length=2048, hop_length=512)
        y_noise_reduced = nr.reduce_noise(y=y_trimmed, sr=sr)
        extracted_duration = librosa.get_duration(y=y_noise_reduced, sr=sr)
        chroma_stft = librosa.feature.chroma_stft(y=y_noise_reduced, sr=sr)
        rmse = librosa.feature.rms(y=y_noise_reduced)
        spec_cent = librosa.feature.spectral_centroid(y=y_noise_reduced, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y_noise_reduced, sr=sr)
        rolloff = librosa.feature.spectral_rolloff(y=y_noise_reduced, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y=y_noise_reduced)
        mfcc = librosa.feature.mfcc(y=y_noise_reduced, sr=sr)
        tonnetz = librosa.feature.tonnetz(y=y_noise_reduced, sr=sr)
        spectral_contrast = librosa.feature.spectral_contrast(y=y_noise_reduced, sr=sr)
        spectrogram = librosa.feature.melspectrogram(y=y_noise_reduced, sr=sr)

        df.at[index, "audio_duration"] = extracted_duration
        # Fill in the features for each audio
        df.at[index, "chroma_stft_mean"] = np.mean(chroma_stft)
        df.at[index, "chroma_stft_std"] = np.std(chroma_stft)
        df.at[index, "chroma_stft_min"] = np.min(chroma_stft)
        df.at[index, "chroma_stft_max"] = np.max(chroma_stft)

        df.at[index, "rmse_mean"] = np.mean(rmse)
        df.at[index, "rmse_std"] = np.std(rmse)
        df.at[index, "rmse_min"] = np.min(rmse)
        df.at[index, "rmse_max"] = np.max(rmse)

        df.at[index, "spectral_centroid_mean"] = np.mean(spec_cent)
        df.at[index, "spectral_centroid_std"] = np.std(spec_cent)
        df.at[index, "spectral_centroid_min"] = np.min(spec_cent)
        df.at[index, "spectral_centroid_max"] = np.max(spec_cent)

        df.at[index, "spectral_bandwidth_mean"] = np.mean(spec_bw)
        df.at[index, "spectral_bandwidth_std"] = np.std(spec_bw)
        df.at[index, "spectral_bandwidth_min"] = np.min(spec_bw)
        df.at[index, "spectral_bandwidth_max"] = np.max(spec_bw)

        df.at[index, "rolloff_mean"] = np.mean(rolloff)
        df.at[index, "rolloff__std"] = np.std(rolloff)
        df.at[index, "rolloff_min"] = np.min(rolloff)
        df.at[index, "rolloff_max"] = np.max(rolloff)

        df.at[index, "zero_crossing_rate_mean"] = np.mean(zcr)
        df.at[index, "zero_crossing_rate_std"] = np.std(zcr)
        df.at[index, "zero_crossing_rate_min"] = np.min(zcr)
        df.at[index, "zero_crossing_rate_max"] = np.max(zcr)

        for i in range(len(mfcc)):
            df.at[index, f"mfcc_mean{i + 1}"] = np.mean(mfcc[i])
            df.at[index, f"mfcc_std{i + 1}"] = np.std(mfcc[i])
            df.at[index, f"mfcc_min{i + 1}"] = np.min(mfcc[i])
            df.at[index, f"mfcc_max{i + 1}"] = np.max(mfcc[i])
            df.at[index, f"mfcc_skew{i + 1}"] = skew(mfcc[i])
            df.at[index, f"mfcc_kurtosis{i + 1}"] = kurtosis(mfcc[i])

        for i in range(len(tonnetz)):
            df.at[index, f"tonnetz_mean{i + 1}"] = np.mean(tonnetz[i])
            df.at[index, f"tonnetz_std{i + 1}"] = np.std(tonnetz[i])
            df.at[index, f"tonnetz_min{i + 1}"] = np.min(tonnetz[i])
            df.at[index, f"tonnetz_max{i + 1}"] = np.max(tonnetz[i])
            df.at[index, f"tonnetz_skew{i + 1}"] = skew(tonnetz[i])
            df.at[index, f"tonnetz_kurtosis{i + 1}"] = kurtosis(tonnetz[i])

        for i in range(len(spectral_contrast)):
            df.at[index, f"spectral_contrast_mean{i + 1}"] = np.mean(spectral_contrast[i])
            df.at[index, f"spectral_contrast_std{i + 1}"] = np.std(spectral_contrast[i])
            df.at[index, f"spectral_contrast_min{i + 1}"] = np.min(spectral_contrast[i])
            df.at[index, f"spectral_contrast_max{i + 1}"] = np.max(spectral_contrast[i])
            df.at[index, f"spectral_contrast_skew{i + 1}"] = skew(spectral_contrast[i])
            df.at[index, f"spectral_contrast_kurtosis{i + 1}"] = kurtosis(spectral_contrast[i])

        for i in range(len(chroma_stft)):
            df.at[index, f"chroma_stft_mean{i + 1}"] = np.mean(chroma_stft[i])
            df.at[index, f"chroma_stft_std{i + 1}"] = np.std(chroma_stft[i])
            df.at[index, f"chroma_stft_min{i + 1}"] = np.min(chroma_stft[i])
            df.at[index, f"chroma_stft_max{i + 1}"] = np.max(chroma_stft[i])
            df.at[index, f"chroma_stft_skew{i + 1}"] = skew(chroma_stft[i])
            df.at[index, f"chroma_stft_kurtosis{i + 1}"] = kurtosis(chroma_stft[i])

        for i in range(len(spectrogram)):
            df.at[index, f"spectogram_mean{i + 1}"] = np.mean(spectrogram[i])
            df.at[index, f"spectogram_std{i + 1}"] = np.std(spectrogram[i])
            df.at[index, f"spectogram_min{i + 1}"] = np.min(spectrogram[i])
            df.at[index, f"spectogram_max{i + 1}"] = np.max(spectrogram[i])
            df.at[index, f"spectogram_skew{i + 1}"] = skew(spectrogram[i])
            df.at[index, f"spectogram_kurtosis{i + 1}"] = kurtosis(spectrogram[i])

    return df
