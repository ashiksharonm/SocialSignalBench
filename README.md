# SocialSignalBench: Multimodal Engagement Signal Benchmark

**SocialSignalBench** is an experimental benchmark and evaluation toolkit designed to measure model quality on multimodal behavioral-signal proxies: **Engagement**, **Confusion**, and **Hesitation**. It utilizes both **Audio** (Speech) and **Visual** (Facial Expression) signals.

## 🎯 Goal
Build a robust, reproducible framework to evaluate how well models can detect these social signals using free, public datasets.

## 📊 Behavioral Proxies
| Target Label | Proxy Source | Justification |
| :--- | :--- | :--- |
| **Engagement** | Happy | Positive affect correlates with active engagement. |
| **Confusion** | Surprise + Neutral | Surprise often signals unexpectedness; combined with neutral it mimics "puzzlement". |
| **Hesitation** | Fear + Neutral | Fear (mild anxiety) + Neutral mimics uncertainty or hesitation. |

## 📂 Datasets
- **Audio**: [RAVDESS](https://zenodo.org/record/1188976) (Speech) - Requires manual download.
- **Video/Face**: [CK+]([https://www.consortium.ri.cmu.edu/ckplus/](https://www.ri.cmu.edu/project/cohn-kanade-au-coded-facial-expression-database/)) (Cohn-Kanade) - Requires manual download.

- **Audio (alternative)**: TESS (`scripts/ingest_tess.py`), which has only 2 speakers.

## ⚠️ Limitations and Validity of Results
No benchmark results are committed to this repo, and none should be quoted from it (earlier claims such as "93.8%" had no committed results behind them).

- **Proxy labels, not ground truth.** Engagement/Confusion/Hesitation are relabelled basic emotions (see table above); they are not annotated behavioral states, so scores measure emotion recognition mapped through an assumed proxy.
- **TESS split leaks speakers.** `ingest_tess.py` makes a random 80/10/10 split over utterances, and TESS contains only 2 speakers, so the same speakers appear in train and test. Reported audio accuracy under this split overstates generalization to new speakers.
- **Fusion results are invalid.** `PairedSocialDataset` pairs each audio clip with a randomly chosen face image that has the same label. The two modalities are therefore never from the same person or moment, and the label is effectively leaked across modalities. Any fusion metric from this pairing should be treated as invalid until real co-recorded audio-visual data (or at least speaker-disjoint evaluation) is used.
- Training uses label smoothing of 0.1 in `src/train.py`.

## 🏗 Architecture
- **Preprocessing**: 
  - Audio: MFCC extraction (Librosa).
  - Face: Haar Cascade detection + Resize (OpenCV).
- **Models**:
  - `Baseline Audio`: 1D CNN on MFCCs.
  - `Baseline Face`: 2D CNN on Face Images.
  - `Fusion`: Late Fusion (Concatenation of embeddings) trained on synthetic label-matched pairs (see Limitations; results are not valid).
- **Gold Slice**: A mined subset of "Hard" cases (low confidence, specific confusion pairs) for robust benchmarking.

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- Docker (optional)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/SocialSignalBench.git
   cd SocialSignalBench
   ```
2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
### Data Ingestion
1. **Audio**: Run the auto-downloader.
   ```bash
   python scripts/download_ravdess.py
   ```
2. **Face** (Optional): Download CK+ manually to `data/raw/ckplus` and run:
   ```bash
   python scripts/ingest_ckplus.py
   ```

### Running the Pipeline

**1. Preprocessing**
Extract features from raw data.
```bash
python -m src.run --config configs/base_config.yaml --mode preprocess
```

**2. Train Baselines**
Train the Audio Baseline model.
```bash
python -m src.run --config configs/base_config.yaml --mode train
```

**3. Train Fusion Model**
Train the Multimodal Fusion model (requires preprocess to be done).
```bash
python -m src.run --config configs/fusion.yaml --mode train_fusion
```

**4. Mine Gold Slice**
Identify hard examples from the trained model to create a Gold Slice benchmark.
```bash
python -m src.run --config configs/base_config.yaml --mode mine_gold_slice
```

**5. Evaluation**
Run comprehensive evaluation and generate reports.
```bash
python -m src.run --config configs/fusion.yaml --mode evaluate
```

## 📈 Results & Artifacts
Outputs are saved to `outputs/runs/<run_id>/`:
- `metrics.json`: Accuracy, F1 scores.
- `confusion_matrix.png`: Visualizes classification errors.
- `errors.csv`: Detailed log of misclassified samples.
- `model_best.pt`: Saved model weights.

### Compare Runs
Use the utility script to compare performance between two runs:
```bash
python scripts/compare_runs.py outputs/runs/run_A outputs/runs/run_B
```

## 🛠 CI/CD
A GitHub Actions workflow (`.github/workflows/ci.yml`) runs linting (flake8) and unit tests (pytest) on every push.

## License
MIT
