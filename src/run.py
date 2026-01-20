import argparse
import yaml
import os
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.preprocess.audio import AudioPreprocessor
from src.preprocess.face import FacePreprocessor


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def run_preprocess(config):
    print("Starting Preprocssing (RAVDESS + TESS)...")

    raw_dir = Path(config["paths"]["raw_data"])
    processed_dir = Path(config["paths"]["processed_data"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Metadata
    dfs = []
    
    # 1. RAVDESS
    rav_meta = raw_dir / "ravdess" / "metadata.csv"
    if rav_meta.exists():
        dfs.append(pd.read_csv(rav_meta))
        
    # 2. TESS
    tess_meta = raw_dir / "tess" / "metadata.csv"
    if tess_meta.exists():
        dfs.append(pd.read_csv(tess_meta))
        
    if not dfs:
        print("No metadata found in data/raw/ravdess or data/raw/tess.")
        return

    df = pd.concat(dfs, ignore_index=True)
    print(f"Total samples to process: {len(df)}")

    # Initialize Preprocessors
    audio_proc = AudioPreprocessor(config)

    new_records = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        sample_id = row["sample_id"]
        modality = row["modality"]
        file_path = row["file_path"]

        # Path resolution logic
        # TESS ingest stored absolute paths, RAVDESS ingest might have too.
        # But safeguards are good.
        full_path = Path(file_path)

        if not full_path.exists():
            # Try relative to project root or data/raw
            if (raw_dir / file_path).exists():
                 full_path = raw_dir / file_path
            elif Path(f"data/{file_path}").exists():
                 full_path = Path(f"data/{file_path}")
            else:
                 print(f"Skipping missing: {file_path}")
                 continue

        if modality == "audio":
            # Check if already done? (Optional, but good for speed)
            save_dir = processed_dir / "audio"
            save_dir.mkdir(exist_ok=True)
            save_path = save_dir / f"{sample_id}.npy"
            
            if save_path.exists():
                # Skip if exists? Or overwrite? 
                pass # Proceed to process for now to be safe, or check config?
            
            try:
                features = audio_proc.process_file(str(full_path))
                if features is not None:
                    audio_proc.save_features(features, save_path)
                    
                    # Update record
                    row["processed_path"] = str(save_path)
                    new_records.append(row)
            except Exception as e:
                print(f"Error processing {full_path}: {e}")

    # Save new metadata
    out_df = pd.DataFrame(new_records)
    out_path = processed_dir / "metadata_processed.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Preprocessing complete. Saved {len(out_df)} records to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config yaml")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "preprocess",
            "train",
            "evaluate",
            "full_pipeline",
            "train_fusion",
            "mine_gold_slice",
        ],
    )

    args = parser.parse_args()
    config = load_config(args.config)

    if args.mode == "preprocess":
        run_preprocess(config)
    elif args.mode == "train":
        from src.dataset import SocialSignalDataset
        from src.models.baselines import AudioBaseline
        from src.train import run_training

        # Load Datasets
        meta_path = Path(config["paths"]["processed_data"]) / "metadata_processed.csv"

        if config["model"]["type"] == "baseline_audio":
            train_ds = SocialSignalDataset(meta_path, "audio", "train")
            val_ds = SocialSignalDataset(meta_path, "audio", "val")

            # Init Model
            model = AudioBaseline(
                n_mfcc=config["preprocessing"]["audio"]["n_mfcc"], n_classes=4
            )

            run_training(config, model, train_ds, val_ds)
        else:
            print(f"Model type {config['model']['type']} not implemented yet.")

    elif args.mode == "train_fusion":
        from src.dataset import PairedSocialDataset
        from src.models.baselines import AudioBaseline, FaceBaseline
        from src.models.fusion import MultimodalFusion
        from src.train import run_training

        meta_path = Path(config["paths"]["processed_data"]) / "metadata_processed.csv"
        train_ds = PairedSocialDataset(meta_path, "train")
        val_ds = PairedSocialDataset(meta_path, "val")

        # Init Sub-Models
        # Determine Sub-Models based on Config
        
        use_resnet = config['model'].get('use_resnet', False)
        use_eff = config['model'].get('use_efficientnet', False)
        
        if use_eff:
             from src.models.transfer import AudioEfficientNet
             # Face EfficientNet? We only implemented AudioEfficientNet so far.
             # Let's reuse ResNet for Face or implement FaceEfficientNet?
             # For now, let's use AudioEfficientNet for audio, and FaceResNet for face (since dataset is dummy mostly).
             # But we should implement FaceEfficientNet for completeness or re-use AudioEfficientNet if generic?
             # AudioEfficientNet takes 1 channel expand to 3. Face is same.
             # So we can use AudioEfficientNet class for Face too actually.
             print("Initializing Fusion with EfficientNet-B0...")
             audio_model = AudioEfficientNet(n_classes=4)
             face_model = AudioEfficientNet(n_classes=4) # Reusing class logic (1->3 chan)
             
             # EfficientNet-B0 output dim = 1280
             model = MultimodalFusion(audio_model, face_model, n_classes=4, audio_dim=1280, face_dim=1280)
             
        elif use_resnet:
             from src.models.transfer import AudioResNet, FaceResNet 
             print("Initializing Fusion with ResNet-18 Encoders...")
             audio_model = AudioResNet(n_classes=4)
             face_model = FaceResNet(n_classes=4)
             model = MultimodalFusion(audio_model, face_model, n_classes=4, audio_dim=512, face_dim=512)
        else:
             from src.models.baselines import AudioBaseline, FaceBaseline
             audio_model = AudioBaseline(n_mfcc=config['preprocessing']['audio']['n_mfcc'], n_classes=4)
             face_model = FaceBaseline(in_channels=1, n_classes=4)
             model = MultimodalFusion(audio_model, face_model, n_classes=4)
        
        run_training(config, model, train_ds, val_ds)

    elif args.mode == "mine_gold_slice":
        from src.gold_slice import mine_gold_slice
        from src.dataset import SocialSignalDataset
        from src.models.baselines import AudioBaseline

        meta_path = Path(config["paths"]["processed_data"]) / "metadata_processed.csv"
        # Mining on Audio Baseline for now as we have data
        ds = SocialSignalDataset(meta_path, "audio", "test")
        model = AudioBaseline(n_mfcc=40, n_classes=4)

        # Load weights
        # model.load_state_dict(...)

        out_path = Path(config["paths"]["gold_slice"]) / "metadata.csv"
        mine_gold_slice(model, ds, out_path, config)

    elif args.mode == "evaluate":
        from src.dataset import PairedSocialDataset
        from src.models.baselines import AudioBaseline, FaceBaseline
        from src.models.fusion import MultimodalFusion
        from src.evaluate import run_evaluation

        # Load Data
        meta_path = Path(config["paths"]["processed_data"]) / "metadata_processed.csv"
        test_ds = PairedSocialDataset(meta_path, "test")

        # Init Sub-Models (Same as train_fusion)
        use_resnet = config['model'].get('use_resnet', False)
        use_eff = config['model'].get('use_efficientnet', False)
        
        if use_eff:
             from src.models.transfer import AudioEfficientNet
             print("Initializing Fusion with EfficientNet-B0...")
             audio_model = AudioEfficientNet(n_classes=4)
             face_model = AudioEfficientNet(n_classes=4)
             model = MultimodalFusion(audio_model, face_model, n_classes=4, audio_dim=1280, face_dim=1280)
        elif use_resnet:
             from src.models.transfer import AudioResNet, FaceResNet 
             print("Initializing Fusion with ResNet-18 Encoders...")
             audio_model = AudioResNet(n_classes=4)
             face_model = FaceResNet(n_classes=4)
             model = MultimodalFusion(audio_model, face_model, n_classes=4, audio_dim=512, face_dim=512)
        else:
             from src.models.baselines import AudioBaseline, FaceBaseline
             audio_model = AudioBaseline(n_mfcc=config['preprocessing']['audio']['n_mfcc'], n_classes=4)
             face_model = FaceBaseline(in_channels=1, n_classes=4)
             model = MultimodalFusion(audio_model, face_model, n_classes=4)

        # Load Weights (Latest run or specific)
        run_dir = Path(config["paths"]["outputs"]) / "runs" / "latest_run"
        model_path = run_dir / "model_best.pt"

        if model_path.exists():
            print(f"Loading model from {model_path}")
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            print("No trained model found at default location. Evaluating random weights.")

        output_dir = run_dir / "eval_results"
        run_evaluation(model, test_ds, output_dir, config)

    else:
        print(f"Mode {args.mode} not implemented yet.")


if __name__ == "__main__":
    main()
