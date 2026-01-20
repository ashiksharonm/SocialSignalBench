import kagglehub
import shutil
import os
from pathlib import Path

# Download latest version
print("Downloading TESS dataset...")
path = kagglehub.dataset_download("ejlok1/toronto-emotional-speech-set-tess")

print("Path to dataset files:", path)

# Move to data/raw/tess
target_dir = Path("data/raw/tess")
target_dir.mkdir(parents=True, exist_ok=True)

source = Path(path)
# Check if source has subfolders or files directly
# Kagglehub usually downloads to a cache dir.
# We want to copy everything from source to data/raw/tess

print(f"Moving files from {source} to {target_dir}...")
# Use dirs_exist_ok=True for Python 3.8+
shutil.copytree(source, target_dir, dirs_exist_ok=True)
print("Download and move complete.")
