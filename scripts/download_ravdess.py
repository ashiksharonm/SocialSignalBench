import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse
from src.schema import SocialSignalSample, LabelMapper, RANDOM_SEED
from sklearn.model_selection import train_test_split

# RAVDESS Filename Identifiers
# Modality (01 = full-AV, 02 = video-only, 03 = audio-only).
# Vocal channel (01 = speech, 02 = song).
# Emotion (01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised).
# Intensity (01 = normal, 02 = strong). NOTE: There is no strong intensity for the 'neutral' emotion.
# Statement (01 = "Kids are talking by the door", 02 = "Dogs are sitting by the door").
# Repetition (01 = 1st repetition, 02 = 2nd repetition).
# Actor (01 to 24. Odd numbered actors are male, even numbered actors are female).

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


def download_file(url, dest_path):
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024
    with open(dest_path, "wb") as file, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            bar.update(len(data))
            file.write(data)


def process_ravdess(raw_dir, output_file):
    raw_dir = Path(raw_dir)
    records = []

    # Walk through all files
    audio_files = list(raw_dir.glob("**/*.wav"))
    if not audio_files:
        print(f"No .wav files found in {raw_dir}. Did download fail?")
        return

    print(f"Found {len(audio_files)} audio files. Processing metadata...")

    for file_path in audio_files:
        filename = file_path.name
        parts = filename.split("-")
        if len(parts) != 7:
            continue

        modality = parts[0]
        emotion_code = parts[2]
        actor = parts[6].split(".")[0]

        # We only want Audio-only (03) or checking validity
        # Actually RAVDESS Speech zip contains Audio_Speech_Actors_01-24.zip

        raw_label = EMOTION_MAP.get(emotion_code)
        if not raw_label:
            continue

        proxy_label = LabelMapper.map_label(raw_label)

        # Create Sample ID
        sample_id = f"ravdess_{filename.replace('.wav', '')}"

        records.append(
            {
                "sample_id": sample_id,
                "dataset": "ravdess",
                "modality": "audio",
                "file_path": str(
                    file_path.relative_to(raw_dir.parent.parent)
                ),  # data/raw/ravdess/...
                "raw_label": raw_label,
                "proxy_label": proxy_label,
                "actor": actor,
            }
        )

    df = pd.DataFrame(records)

    # Split
    actors = df["actor"].unique()
    train_actors, test_actors = train_test_split(
        actors, test_size=0.2, random_state=RANDOM_SEED
    )
    val_actors, test_actors = train_test_split(
        test_actors, test_size=0.5, random_state=RANDOM_SEED
    )

    def get_split(actor):
        if actor in train_actors:
            return "train"
        if actor in val_actors:
            return "val"
        return "test"

    df["split"] = df["actor"].apply(get_split)

    # Save
    print(f"Saving metadata to {output_file}")
    df.to_csv(output_file, index=False)
    print("Stats:")
    print(df["proxy_label"].value_counts())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="data/raw/ravdess")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # RAVDESS Audio Speech URL (Zenodo)
    # Using the direct link to the speech audio zip
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip?download=1"
    zip_path = output_dir / "ravdess_audio.zip"

    if not zip_path.exists():
        print("Downloading RAVDESS...")
        download_file(url, zip_path)
    else:
        print("RAVDESS zip already exists.")

    # Unzip
    print("Extracting...")
    # Check if already extracted (simple check)
    if not (output_dir / "Actor_01").exists():
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)

    # Process Metadata
    process_ravdess(output_dir, output_dir / "metadata.csv")


if __name__ == "__main__":
    main()
