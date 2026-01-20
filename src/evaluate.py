import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from pathlib import Path


def plot_confusion_matrix(y_true, y_pred, classes, output_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def run_evaluation(model, dataset, output_dir, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    loader = DataLoader(
        dataset, batch_size=config["model"]["batch_size"], shuffle=False
    )

    all_preds = []
    all_labels = []
    all_confidences = []

    # Store errors
    error_list = []

    print("Running Evaluation...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader)):
            # Handle tuple unpacking
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
            conf, preds = torch.max(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(conf.cpu().numpy())

            # Identify errors in this batch
            # We need absolute indices to map back to metadata?
            # DataLoader shuffles=False, so sequential.
            # But batch logic matches i * batch_size + j

            batch_size_actual = labels.size(0)
            base_idx = (
                i * config["model"]["batch_size"]
            )  # Assuming fixed, but actual might vary at end

            # Correction: enumerate(loader) yields batches.
            # better to keep a running counter or just use simple append.
            # For "errors.csv", we need sample IDs.
            # The dataset object has metadata if it's our CustomDataset.

            # Let's reconstruct or rely on sequential access if dataset supports it.
            # If `dataset` is PairedSocialDataset, it has audio_ds which has metadata.

            preds_np = preds.cpu().numpy()
            labels_np = labels.cpu().numpy()
            conf_np = conf.cpu().numpy()

            for j in range(batch_size_actual):
                if preds_np[j] != labels_np[j]:
                    idx_in_dataset = base_idx + j
                    # Safely try to get sample ID
                    sample_id = "unknown"
                    if hasattr(dataset, "metadata"):
                        sample_id = dataset.metadata.iloc[idx_in_dataset]["sample_id"]
                    elif hasattr(dataset, "audio_ds"):
                        sample_id = dataset.audio_ds.metadata.iloc[idx_in_dataset][
                            "sample_id"
                        ]

                    error_list.append(
                        {
                            "sample_id": sample_id,
                            "true_label": int(labels_np[j]),
                            "pred_label": int(preds_np[j]),
                            "confidence": float(conf_np[j]),
                        }
                    )

            base_idx += batch_size_actual  # Update base_idx accurately

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average="macro")
    f1_weighted = f1_score(all_labels, all_preds, average="weighted")

    metrics = {"accuracy": acc, "f1_macro": f1_macro, "f1_weighted": f1_weighted}

    print(f"Evaluation Results:\nAccuracy: {acc:.4f}\nF1 Macro: {f1_macro:.4f}")
    print(classification_report(all_labels, all_preds))

    # Save Artifacts
    os.makedirs(output_dir, exist_ok=True)

    # 1. Metrics JSON
    with open(os.path.join(output_dir, "metrics_eval.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # 2. Confusion Matrix
    classes = ["Engagement", "Confusion", "Hesitation", "Neutral"]  # 0, 1, 2, 3
    plot_confusion_matrix(
        all_labels, all_preds, classes, os.path.join(output_dir, "confusion_matrix.png")
    )

    # 3. Errors CSV
    if error_list:
        pd.DataFrame(error_list).to_csv(
            os.path.join(output_dir, "errors.csv"), index=False
        )
        print(f"Saved {len(error_list)} errors to {output_dir}/errors.csv")

    return metrics
