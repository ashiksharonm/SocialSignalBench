import librosa
import numpy as np
import soundfile as sf
import os
import yaml


class AudioPreprocessor:
    def __init__(self, config):
        self.sr = config["preprocessing"]["audio"]["sample_rate"]
        self.n_mfcc = config["preprocessing"]["audio"]["n_mfcc"]
        self.hop_length = config["preprocessing"]["audio"]["hop_length"]
        self.duration = 3.0  # Max duration in seconds to pad/crop
        self.target_length = int(self.duration * self.sr)

    def process_file(self, file_path):
        """
        Load audio, extract MFCCs, and return features.
        Output shape: (n_mfcc, time_steps)
        """
        try:
            # Load
            y, sr = librosa.load(file_path, sr=self.sr)

            # Trim silence
            y, _ = librosa.effects.trim(y, top_db=20)

            # Pad or Crop
            if len(y) > self.target_length:
                y = y[: self.target_length]
            else:
                padding = self.target_length - len(y)
                y = np.pad(y, (0, padding), mode="constant")

            # Extract MFCC
            mfcc = librosa.feature.mfcc(
                y=y, sr=sr, n_mfcc=self.n_mfcc, hop_length=self.hop_length
            )

            # Normalize (Z-score per sample)
            mean = np.mean(mfcc, axis=1, keepdims=True)
            std = np.std(mfcc, axis=1, keepdims=True)
            mfcc_norm = (mfcc - mean) / (std + 1e-6)

            return mfcc_norm.astype(np.float32)

        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return None

    def save_features(self, features, output_path):
        np.save(output_path, features)
