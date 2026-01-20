import pytest
import numpy as np
import os
import yaml
from src.preprocess.audio import AudioPreprocessor
from src.preprocess.face import FacePreprocessor
import cv2

# Mock Config
config = {
    "preprocessing": {
        "audio": {"sample_rate": 16000, "n_mfcc": 13, "hop_length": 512},
        "face": {"image_size": [64, 64], "detect_face": True},
    }
}


def test_audio_preprocessing(tmp_path):
    # Create dummy wav
    wav_path = tmp_path / "test.wav"
    sr = 16000
    y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr)  # 1 sec sine wave
    import soundfile as sf

    sf.write(wav_path, y, sr)

    preprocessor = AudioPreprocessor(config)
    features = preprocessor.process_file(str(wav_path))

    assert features is not None
    assert features.shape[0] == 13  # n_mfcc
    # Expected Time steps: (3.0 * 16000) / 512 = 93.75 -> approx 94 frames
    # Our padding logic fixes length to 3.0 sec.
    assert features.shape[1] > 0


def test_face_preprocessing(tmp_path):
    # Create dummy image
    img_path = tmp_path / "test.png"
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    cv2.rectangle(img, (50, 50), (150, 150), (255, 255, 255), -1)  # Pseudo face
    cv2.imwrite(str(img_path), img)

    preprocessor = FacePreprocessor(config)
    features = preprocessor.process_image(str(img_path))

    assert features is not None
    assert features.shape == (64, 64, 1)
    assert features.max() <= 1.0
    assert features.min() >= 0.0
