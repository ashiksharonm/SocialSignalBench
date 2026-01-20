import os
import argparse
import pandas as pd
from pathlib import Path
from src.schema import SocialSignalSample

# TESS structure assumption (after manual unzip or download):
# TESS/
#   OAF_angry/
#      OAF_back_angry.wav
#   YAF_fear/
#      ...

EMOTION_MAP = {
    'happiness': 'engagement',
    'pleasant_surprise': 'confusion', # Using surprise as confusion proxy
    'fear': 'hesitation',             # Using fear as hesitation proxy
    'neutral': 'neutral',
    # Others to exclude or map?
    'anger': 'unknown',
    'disgust': 'unknown',
    'sadness': 'unknown'
}

def ingest_tess(raw_dir):
    root = Path(raw_dir)
    if not root.exists():
        print(f"Directory {root} does not exist. Please download TESS and extract it there.")
        print("URL: https://borealisdata.ca/dataset.xhtml?persistentId=doi:10.5683/SP2/E8H2MF")
        return

    samples = []
    
    # Walk directories recursively
    # We look for folders ending in specific emotion suffixes usually
    # Pattern: Actor_Emotion (OAF_Fear)
    
    # Let's search for all wav files and parse their parent folder
    for audio_file in root.rglob("*.wav"):
        folder = audio_file.parent
        folder_name = folder.name
        
        # Folder checking
        parts = folder_name.split('_')
        # Typical: OAF_Fear or OAF_back_fear? 
        # Actually TESS folders are "OAF_Fear".
        # Filenames are "OAF_back_fear.wav" or similar.
        
        # Let's try to parse from folder name first
        if len(parts) >= 2:
            actor_prefix = parts[0]
            if actor_prefix not in ['OAF', 'YAF']:
                # Maybe filename has it?
                # Filename: OAF_back_angry.wav
                fparts = audio_file.stem.split('_')
                if fparts[0] in ['OAF', 'YAF']:
                     actor_prefix = fparts[0]
                     # Emotion is last part usually? "angry"
                     emotion_raw = fparts[-1].lower()
                else:
                     continue
            else:
                 # Folder name based
                 # Check last part: "Fear" -> fear
                 emotion_raw = parts[-1].lower()
        else:
            continue
            
        if emotion_raw == 'ps': emotion_raw = 'pleasant_surprise'
        if emotion_raw == 'happy': emotion_raw = 'happiness'
        if emotion_raw == 'surprised': emotion_raw = 'pleasant_surprise' 
        
        proxy_label = EMOTION_MAP.get(emotion_raw, 'unknown')
        if proxy_label == 'unknown':
            continue
            
        samples.append({
            "sample_id": audio_file.stem,
            "dataset": "tess",
            "modality": "audio",
            "file_path": str(audio_file.absolute()),
            "raw_label": emotion_raw,
            "proxy_label": proxy_label,
            "actor": actor_prefix,
            "split": "train" 
        })

    # Split TESS: 80/10/10 random split since no defined splits
    df = pd.DataFrame(samples)
    if df.empty:
        print("No samples found. Check directory structure.")
        return
        
    print(f"Found {len(df)} TESS samples.")
    
    # Shuffle and split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.8)
    val_end = int(n * 0.9)
    
    df.loc[:train_end, 'split'] = 'train'
    df.loc[train_end:val_end, 'split'] = 'val'
    df.loc[val_end:, 'split'] = 'test'
    
    out_path = root / "metadata.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved metadata to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/raw/tess", help="Directory containing TESS folders")
    args = parser.parse_args()
    ingest_tess(args.dir)
