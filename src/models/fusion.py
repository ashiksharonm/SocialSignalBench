import torch
import torch.nn as nn
from src.models.baselines import AudioBaseline, FaceBaseline


class MultimodalFusion(nn.Module):
    def __init__(self, audio_model, face_model, n_classes, audio_dim=512, face_dim=16384):
        super(MultimodalFusion, self).__init__()
        self.audio_encoder = audio_model
        self.face_encoder = face_model

        # Audio Config
        self.audio_feat_dim = audio_dim

        # Face Config
        self.face_feat_dim = face_dim

        # Fusion MLP
        self.fusion_fc = nn.Sequential(
            nn.Linear(self.audio_feat_dim + self.face_feat_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, n_classes),
        )

    def forward(self, audio, face):
        # Audio Path
        # We need to call specific layers of audio_model since `forward` does full pass.
        # Or we can modify AudioBaseline to have `extract_features` method.
        # Let's rely on `extract_features` existing. I will update baselines.py next.

        a_emb = self.audio_encoder.extract_features(audio)
        f_emb = self.face_encoder.extract_features(face)
        
        # print(f"DEBUG: Audio Shape: {a_emb.shape}, Face Shape: {f_emb.shape}")

        # Concatenate
        combined = torch.cat((a_emb, f_emb), dim=1)
        # print(f"DEBUG: Combined Shape: {combined.shape}")

        out = self.fusion_fc(combined)
        return out
