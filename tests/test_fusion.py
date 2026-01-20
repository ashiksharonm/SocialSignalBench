import torch
from src.models.baselines import AudioBaseline, FaceBaseline
from src.models.fusion import MultimodalFusion
from src.dataset import PairedSocialDataset, SocialSignalDataset
import pytest
import numpy as np


def test_fusion_model_forward():
    a_model = AudioBaseline(40, 4)
    f_model = FaceBaseline(1, 4)
    model = MultimodalFusion(a_model, f_model, 4)

    batch = 2
    audio = torch.randn(batch, 40, 94)
    face = torch.randn(batch, 1, 128, 128)

    out = model(audio, face)
    assert out.shape == (batch, 4)


def test_paired_dataset(tmp_path):
    # Create fake meta with matching labels
    meta_path = tmp_path / "metadata.csv"
    processed_dir = tmp_path / "data/processed/audio"
    processed_dir.mkdir(parents=True)

    # Audio Sample (Label=Engagement)
    np.save(processed_dir / "a1.npy", np.zeros((40, 94)))

    # Face Sample (Label=Engagement, but missing file so returns zeros)

    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "sample_id": "a1",
                "modality": "audio",
                "split": "train",
                "proxy_label": "engagement",
                "processed_path": str(processed_dir / "a1.npy"),
                "raw_label": "happy",
            },
            {
                "sample_id": "f1",
                "modality": "image",
                "split": "train",
                "proxy_label": "engagement",
                "processed_path": "dummy.npz",
                "raw_label": "happy",
            },
        ]
    )
    df.to_csv(meta_path, index=False)

    ds = PairedSocialDataset(str(meta_path), "train")
    assert len(ds) == 1

    a, f, l = ds[0]
    assert a.shape == (40, 94)
    assert f.shape == (1, 128, 128)
    assert l.item() == 0  # Engagement
