import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path


def mine_gold_slice(model, dataset, output_path, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    gold_samples = []

    print("Mining Gold Slice...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            # Handle both Single and Paired cases
            if len(batch) == 2:
                inputs, labels = batch
                inputs = inputs.to(device)
                outputs = model(inputs)
            else:
                audio, face, labels = batch
                audio, face = audio.to(device), face.to(device)
                outputs = model(audio, face)

            labels = labels.to(device)
            probs = torch.softmax(outputs, dim=1)
            max_prob, pred = torch.max(probs, dim=1)

            confidence = max_prob.item()
            true_label = labels.item()
            pred_label = pred.item()

            # Metadata lookup (hacky via dataset.metadata if available)
            # Since dataset might be filtered, we need original row.
            # Assuming dataset is SocialSignalDataset
            if hasattr(dataset, "metadata"):
                meta_row = dataset.metadata.iloc[i]
                sample_id = meta_row["sample_id"]
            elif hasattr(dataset, "audio_ds"):  # Paired
                meta_row = dataset.audio_ds.metadata.iloc[i]
                sample_id = meta_row["sample_id"]

            # Criteria
            is_gold = False
            reason = []

            # 1. Low Confidence
            if confidence < config["gold_slice"]["confidence_threshold"]:
                is_gold = True
                reason.append("low_confidence")

            # 2. Confusion/Hesitation mixup
            # Map: 0=Eng, 1=Conf, 2=Hes, 3=Neu
            if (true_label == 1 and pred_label == 2) or (
                true_label == 2 and pred_label == 1
            ):
                is_gold = True
                reason.append("confusion_hesitation_swap")

            if is_gold:
                gold_samples.append(
                    {
                        "sample_id": sample_id,
                        "reason": "|".join(reason),
                        "confidence": confidence,
                        "true_label": true_label,
                        "pred_label": pred_label,
                    }
                )

    if gold_samples:
        df = pd.DataFrame(gold_samples)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} gold slice samples to {output_path}")
    else:
        print("No gold slice samples found.")
