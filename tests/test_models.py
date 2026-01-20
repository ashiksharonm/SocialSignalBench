import pytest
import torch
from src.models.baselines import AudioBaseline, FaceBaseline
from src.dataset import SocialSignalDataset
import pandas as pd
import numpy as np


def test_audio_model_shape():
    batch = 4
    mfcc = 40
    time = 94
    classes = 4

    model = AudioBaseline(n_mfcc=mfcc, n_classes=classes)
    x = torch.randn(batch, mfcc, time)
    out = model(x)

    assert out.shape == (batch, classes)


def test_face_model_shape():
    batch = 4
    channels = 1
    size = 128
    classes = 4

    model = FaceBaseline(in_channels=channels, n_classes=classes)
    x = torch.randn(batch, channels, size, size)
    out = model(x)

    assert out.shape == (batch, classes)


def test_dataset_item(tmp_path):
    # Mock processed file
    processed_dir = tmp_path / "data/processed/audio"
    processed_dir.mkdir(parents=True)

    dummy_npy = processed_dir / "test.npy"
    np.save(dummy_npy, np.random.randn(40, 94).astype(np.float32))

    # Mock Metadata
    meta_path = tmp_path / "metadata.csv"
    pd.DataFrame(
        {
            "sample_id": ["test"],
            "modality": ["audio"],
            "split": ["train"],
            "proxy_label": ["engagement"],
            "processed_path": [str(dummy_npy)],
            "raw_label": ["happy"],
        }
    ).to_csv(meta_path, index=False)

    ds = SocialSignalDataset(str(meta_path), modality="audio", split="train")

    item, label = ds[0]
    assert item.shape == (40, 94)
    assert label.dtype == torch.long
