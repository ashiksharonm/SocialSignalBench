import os
import pandas as pd
from pathlib import Path
import argparse
from src.schema import SocialSignalSample, LabelMapper, RANDOM_SEED
from sklearn.model_selection import train_test_split

# CK+ Structure usually:
# cohn-kanade-images/Subject/Sequence/filename.png
# Emotion/Subject/Sequence/filename.txt (contains singular label 0-7)

# Mapping from CK+ numeric to string
CK_EMOTION_MAP = {
    0: "neutral_ck",
    1: "anger",
    2: "contempt",
    3: "disgust",
    4: "fear_ck",
    5: "happy_ck",
    6: "sadness",
    7: "surprise_ck",
}


def process_ckplus(raw_dir, output_file):
    raw_dir = Path(raw_dir)

    image_dir = raw_dir / "cohn-kanade-images"
    emotion_dir = raw_dir / "Emotion"

    if not image_dir.exists() or not emotion_dir.exists():
        print(f"Error: CK+ structure not found in {raw_dir}")
        print("Expected 'cohn-kanade-images' and 'Emotion' subdirectories.")
        print("Please download CK+ manually and unzip into data/raw/ckplus/")
        return

    records = []

    # Iterate over Emotion labels because not all sequences have labels
    # Structure: Emotion/Subject/Sequence/xxxx_emotion.txt

    print("Walking through Emotion labels...")
    for subject_dir in emotion_dir.iterdir():
        if not subject_dir.is_dir():
            continue

        subject_id = subject_dir.name

        for seq_dir in subject_dir.iterdir():
            if not seq_dir.is_dir():
                continue

            seq_id = seq_dir.name

            # Find label file
            label_files = list(seq_dir.glob("*.txt"))
            if not label_files:
                continue

            # Read label
            try:
                with open(label_files[0], "r") as f:
                    val = float(f.read().strip())
                    label_idx = int(val)
            except:
                continue

            raw_label = CK_EMOTION_MAP.get(label_idx)
            if not raw_label:
                continue

            proxy_label = LabelMapper.map_label(raw_label)

            # Find corresponding image sequence
            # Usually the last frame is the apex of emotion
            img_seq_path = image_dir / subject_id / seq_id
            if not img_seq_path.exists():
                continue

            images = sorted(list(img_seq_path.glob("*.png")))
            if not images:
                continue

            # Take the last 3 frames as "Peak" emotion samples
            # This is a common heuristic for CK+
            selected_images = images[-3:]

            for img_path in selected_images:
                sample_id = f"ckplus_{img_path.stem}"

                records.append(
                    {
                        "sample_id": sample_id,
                        "dataset": "ckplus",
                        "modality": "image",
                        "file_path": str(img_path.relative_to(raw_dir.parent.parent)),
                        "raw_label": raw_label,
                        "proxy_label": proxy_label,
                        "actor": subject_id,
                    }
                )

    if not records:
        print("No labeled samples found. Check directory structure.")
        return

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

    print(f"Saving metadata to {output_file}")
    df.to_csv(output_file, index=False)
    print("Stats:")
    print(df["proxy_label"].value_counts())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckplus_dir", default="data/raw/ckplus")
    args = parser.parse_args()

    process_ckplus(args.ckplus_dir, Path(args.ckplus_dir) / "metadata.csv")


if __name__ == "__main__":
    main()
