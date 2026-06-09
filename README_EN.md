# BirdCLEF+ 2026 — Silver Medal Solution 🥈

**Kaggle Public LB: 0.948 | Silver Medal**

中文 README: [README.md](README.md)

---

### Competition Background

[BirdCLEF+ 2026](https://www.kaggle.com/competitions/birdclef-2026) is a Kaggle competition focused on **passive acoustic monitoring (PAM)**. The goal is to build ML models that automatically identify wildlife species calls from continuous audio recorded in the Pantanal wetlands of South America — one of the most biodiverse regions on Earth. The organizers provided data from 1,000 acoustic recorders deployed across the Brazilian Pantanal.

**Key constraints:**
- Input: 5-second audio windows, multi-species soundscapes
- Metric: Macro-averaged ROC-AUC (classes with no positive labels are skipped)
- Submission: Notebook-only, **CPU-only, 90-minute time limit** → models must be converted to ONNX or use lightweight architectures

### Core Challenges

- **Long-tail distribution**: 234 species, many with very few training examples
- **Noisy soundscapes**: Overlapping calls, wind, rain, and insect noise
- **No GPU at inference**: All heavy models must be distilled or ONNX-exported before submission
- **Sequence context**: A single 5-second window often lacks enough signal — temporal context across windows matters

### Solution Overview

Our solution is a **dual-pipeline architecture** that combines two complementary models and applies several post-processing refinements.

The **most impactful single change** was retraining the SED model and replacing the baseline `.pt` weights — this alone drove the majority of the score improvement.

```
Audio (5s windows)
        │
        ├── Pipeline 1: ProtoSSMv5  (sequence-aware, Perch embeddings)
        │       └── Score matrix  (N_windows × 234)
        │
        └── Pipeline 2: Distilled SED  (frame-wise CNN, EfficientNet-B0)
                └── Score matrix  (N_windows × 234)
                        │
                Rank-based blend: 0.60 × ProtoSSM + 0.40 × SED
                        │
                Post-processing (thresholding, smoothing, mirroring)
                        │
                submission.csv
```

### Pipeline 1 — ProtoSSMv5 (Sequence Model)

This pipeline leverages **Google Perch v2**, a frozen bird-vocalization embedding model producing 1536-dim vectors per 5-second window.

- **Perch embeddings** extracted for all training soundscapes (708 windows from 59 fully-labeled files)
- **PCA compression**: 1536-dim → 64-dim (retaining 81.47% variance)
- **MLP probes**: 63 per-class MLP classifiers trained on PCA embeddings
- **ResidualSSM**: A lightweight State Space Model (d_model=128, d_state=16, 2 layers) learns sequential correction on top of MLP probe outputs
  - Best correction weight found via OOF sweep: `0.40` (OOF macro-AUC = 0.99402)
- **Species mapping**: 203/234 species mapped to Perch logits; 31 unmapped species handled via genus-level proxy mapping
- Training time: ~12.7s (ProtoSSM) + 1.6s (ResidualSSM)

### Pipeline 2 — Distilled SED (CNN Model)

An EfficientNet-B0 based Sound Event Detection model trained with **teacher-student distillation** from Perch v2, optimized for fast CPU inference via ONNX.

**Architecture highlights:**
- **Backbone**: `tf_efficientnet_b0.ns_jft_in1k` on Mel Spectrograms (256 mels, n_fft=2048, SR=32kHz)
- **GeM Frequency Pooling**: Learnable generalized mean pooling (p=3.0 init) over the frequency axis — focuses on bird-relevant frequency bands
- **Attention Bottleneck**: 512-dim dense → 1D Conv → frame-wise attention weights → clip-level prediction
- **Perch Distillation Head**: GAP + linear projection mapping EfficientNet features into Perch's 1536-dim embedding space
- **10-fold ensemble**: `sed_fold0.onnx` through `sed_fold9.onnx` — all folds averaged at inference

**Data augmentation:**
- Signal-level: random gain jitter (±6 dB), noise injection (SNR 10–30 dB)
- SpecAugment: frequency and time masking on GPU
- Focal-Focal MixUp: Beta-distribution blending of two focal recordings
- Focal-Soundscape MixUp: injecting focal calls into labeled soundscape backgrounds
- Rare species dynamic upsampling: every class guaranteed ≥20 samples per fold

### Optimization Strategies

**1. Model ensemble with dynamic weighting**
Combine ProtoSSM and SED predictions with per-class weights (`xSED`). Mapped species use weight 0.60/0.40; unmapped species use 0.35 for the Proto component.

**2. Temporal smoothing**
Predictions across adjacent time windows are smoothed with `gaussian_filter1d`, suppressing isolated false positives and reinforcing continuous vocalization events.

**3. Metadata and prior injection**
Site and time-of-day features parsed from filenames (`BC2026_Train/Test_{id}_{site}_{date}_{time}.ogg`) — used to inject species distribution priors and calibrate scores for location-specific species.

**4. Adaptive thresholding**
Per-species dynamic thresholds replace a single global threshold. 44 rare species receive adaptive thresholds calibrated on OOF predictions (mean: 0.464, range: [0.20, 0.50]).

**5. Sonotype mirroring**
Taxonomically related species share prediction signal — applied to 10 species pairs where co-occurrence or misidentification is common.

**6. Taxonomic smoothing**
Post-processing smooths predictions at the genus/family level, improving robustness for species with limited training data.

### Background Knowledge Required

- **Audio deep learning**: Mel-Spectrogram and MFCC fundamentals; tuning spectrogram parameters (n_fft, n_mels, hop_length) for GPU efficiency
- **Passive acoustic monitoring (PAM)**: Analyzing temporal sequences of soundscape data; multi-channel audio processing
- **Ensemble learning**: K-fold cross-validation and multi-fold ensemble submission
- **Kaggle offline competition workflow**: Packaging all dependencies (Python wheels) into Datasets so the notebook runs in a fully air-gapped environment within the 90-minute CPU limit

### Files

| File | Description |
|------|-------------|
| `birdclef-2026.ipynb` | Main inference / submission notebook |
| `bc2026-distilled-sed.ipynb` | SED model training notebook (retrain from scratch) |
| `utils_all.py` | Shared utilities called by the inference notebook |

### How to Run

1. Upload `birdclef-2026.ipynb` and `utils_all.py` to a Kaggle notebook.
2. Attach the required datasets:
   - SED ONNX fold models (`sed_fold0.onnx` … `sed_fold9.onnx`)
   - Perch v2 ONNX model (`perch_v2_no_dft.onnx`)
   - Perch embedding cache (`full_perch_meta.parquet`, `full_perch_arrays.npz`)
3. Set accelerator to **CPU** (required by competition rules).
4. Run all cells — the notebook generates `submission.csv` within 90 minutes.

To retrain the SED model, use `bc2026-distilled-sed.ipynb` (GPU recommended).

### References

- Baseline: [BirdCLEF-2026-v9 by raunakdey07](https://www.kaggle.com/code/raunakdey07/birdclef-2026-v9)
- SED training baseline: [bc2026-distilled-sed by tuckerarrants](https://www.kaggle.com/code/tuckerarrants/bc2026-distilled-sed)
- Google Perch v2: [bird-vocalization-classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier)

---

## License

This project is released under the [MIT License](LICENSE).
