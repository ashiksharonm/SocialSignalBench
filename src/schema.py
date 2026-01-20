from dataclasses import dataclass
from typing import Optional, List
import os


@dataclass
class SocialSignalSample:
    sample_id: str
    dataset: str  # 'ravdess' or 'ckplus'
    modality: str  # 'audio' or 'image'
    file_path: str
    raw_label: str
    proxy_label: str  # 'engagement', 'confusion', 'hesitation', 'neutral', 'unknown'
    split: str  # 'train', 'val', 'test'
    confidence: float = 1.0  # For gold slice mining later


class LabelMapper:
    # Target Proxy Labels
    ENGAGEMENT = "engagement"
    CONFUSION = "confusion"
    HESITATION = "hesitation"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"

    # Raw -> Proxy Mapping
    MAPPING = {
        # RAVDESS (01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised)
        "happy": ENGAGEMENT,
        "surprised": CONFUSION,
        "fearful": HESITATION,
        "neutral": NEUTRAL,
        "calm": NEUTRAL,  # Calm is similar to neutral
        # CK+ (0=neutral, 1=anger, 2=contempt, 3=disgust, 4=fear, 5=happy, 6=sadness, 7=surprise)
        "happy_ck": ENGAGEMENT,
        "surprise_ck": CONFUSION,
        "fear_ck": HESITATION,
        "neutral_ck": NEUTRAL,
    }

    @classmethod
    def map_label(cls, raw_label: str) -> str:
        return cls.MAPPING.get(raw_label.lower(), cls.UNKNOWN)


DATASET_SPLITS = [0.8, 0.1, 0.1]  # Train, Val, Test
RANDOM_SEED = 42
