import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from src.schema import LabelMapper


class SocialSignalDataset(Dataset):
    def __init__(self, metadata_path, modality, split="train", label_mapping=None):
        self.metadata = pd.read_csv(metadata_path)
        self.modality = modality
        self.split = split

        # Filter
        self.metadata = self.metadata[
            (self.metadata["modality"] == modality) & (self.metadata["split"] == split)
        ]

        # Drop samples with 'unknown' or unmapped labels if desired (or keep them?)
        # For now, we drop 'unknown' proxy labels
        self.metadata = self.metadata[
            self.metadata["proxy_label"] != LabelMapper.UNKNOWN
        ]

        self.class_map = {
            LabelMapper.ENGAGEMENT: 0,
            LabelMapper.CONFUSION: 1,
            LabelMapper.HESITATION: 2,
            LabelMapper.NEUTRAL: 3,
            # Note: Neutral might be mapped to Confusion/Hesitation later or kept separate.
            # Strategy: Train 4-way, and during eval metrics, maybe merge.
        }

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        file_path = row["processed_path"]
        label_str = row["proxy_label"]

        # Load Data
        if self.modality == "audio":
            try:
                data = np.load(file_path)  # (n_mfcc, time) -> (40, 94)
                
                # Check NaNs
                if np.isnan(data).any():
                     data = np.nan_to_num(data)
                
                # SpecAugment (Train Only)
                if self.split == 'train':
                    # 1. Freq Masking
                    # Mask 1 band of max width 5
                    F = 5
                    f0 = np.random.randint(0, max(1, data.shape[0] - F))
                    data[f0:f0+F, :] = 0
                    
                    # 2. Time Masking
                    # Mask 1 band of max width 10
                    T = 10
                    t0 = np.random.randint(0, max(1, data.shape[1] - T))
                    data[:, t0:t0+T] = 0
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                data = np.zeros((40, 94), dtype=np.float32)

        elif self.modality == "image":
            try:
                # loaded = np.load(file_path)
                # data = loaded['features'] # (H, W, 1)
                # Transpose to (1, H, W) for PyTorch
                # data = data.transpose(2, 0, 1)
                pass  # Placeholder until we have face data
                data = np.zeros((1, 128, 128), dtype=np.float32)
            except:
                data = np.zeros((1, 128, 128), dtype=np.float32)

        # Label
        label = self.class_map.get(label_str, 3)  # Default to Neutral if issue

        return torch.from_numpy(data), torch.tensor(label, dtype=torch.long)


class PairedSocialDataset(Dataset):
    def __init__(self, metadata_path, split="train"):
        # We need two internal datasets
        self.audio_ds = SocialSignalDataset(metadata_path, "audio", split)
        # Face requires existing data, but we lack CK+. We will filter gracefully.
        # If no face data, we might need to mock or fail.
        # For now, let's assume we want to match labels.
        self.face_ds = SocialSignalDataset(metadata_path, "image", split)

        # Group indices by label
        self.audio_indices = self._group_by_label(self.audio_ds)
        self.face_indices = self._group_by_label(self.face_ds)

        # Max length determined by audio usually (since raven is dominant if ck+ missing)
        # But for pairing, we iterate over one and random sample other?
        # Let's iterate over Audio and pair with random Face of same label.
        self.primary_modality = "audio"

    def _group_by_label(self, ds):
        groups = {}
        for idx in range(len(ds)):
            # We need to access label without loading data.
            # DS.metadata is accessible.
            row = ds.metadata.iloc[idx]
            label = ds.class_map.get(row["proxy_label"], 3)
            if label not in groups:
                groups[label] = []
            groups[label].append(idx)
        return groups

    def __len__(self):
        return len(self.audio_ds)

    def __getitem__(self, idx):
        # Get Audio
        audio_data, label = self.audio_ds[idx]
        label_item = label.item()

        # Get matching Face
        if label_item in self.face_indices and len(self.face_indices[label_item]) > 0:
            face_idx = np.random.choice(self.face_indices[label_item])
            face_data, _ = self.face_ds[face_idx]
        else:
            # Fallback if no matching face class (e.g. dataset imbalance or missing modality)
            # Return zeros
            face_data = torch.zeros((1, 128, 128), dtype=torch.float32)

        return audio_data, face_data, label
