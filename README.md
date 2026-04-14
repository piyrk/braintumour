# Brain Tumour Detection & Augmentation with GANs

A comprehensive solution for the **Global Brain Tumour GAN Challenge** — detecting and segmenting brain tumours in MRI scans, with synthetic data augmentation via Conditional GANs.

---

## 📋 Overview

This project provides a complete pipeline for:
- **Baseline Detection**: CNN-based tumour detection (EfficientNet-b0)
- **Segmentation**: U-Net architecture for precise tumour boundary delineation
- **Synthetic Augmentation**: Conditional GAN to generate synthetic MRI scans with tumours
- **Evaluation**: Comprehensive metrics (Dice, IoU, F1, confusion matrices)
- **Visualization**: Streamlit app for interactive result exploration

---

## 🏗️ Project Structure

```
├── src/                          # Core package
│   └── gan_brain_tumour_challenge/  # Main module
├── artifacts/                    # Training outputs
│   ├── classifier/               # Classification models & checkpoints
│   ├── detection/                # Detection models
│   ├── gan/                       # GAN checkpoints
│   └── segmentation/             # Segmentation models
├── data/                         # Dataset folders
│   ├── raw/                      # Original MRI files (NIFTI format)
│   └── processed/                # Preprocessed PNG images
├── docs/                         # Phase documentation
├── train_*.py                    # Training scripts
├── preprocess_data.py            # Data preprocessing pipeline
├── streamlit_app.py              # Interactive visualization
├── check.py                      # GPU validation
└── requirements.txt              # Dependencies
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data (NIFTI → PNG)
Convert BraTS-GLI case folders and extract modalities:
```bash
python prepare_brats_gli.py --source data --target data/processed --modalities t1 t2 t1ce flair
python preprocess_data.py --raw-dir data/raw --processed-dir data/processed
```

### 3. Run Data Exploration
```bash
streamlit run streamlit_app.py
```

---

## 🖥️ Windows DirectML GPU Setup (No CUDA Required)

This project uses **TensorFlow 2.10 with DirectML** for GPU acceleration on Windows without CUDA.

### Setup Steps

#### 1. Create Virtual Environment
```powershell
python -m venv .venv_gpu
.venv_gpu\Scripts\Activate.ps1
```

#### 2. Install TensorFlow with DirectML
```powershell
pip install --upgrade pip
pip install tensorflow==2.10.0 tensorflow-directml==0.4.0.dev230202 numpy==1.26.4
```

#### 3. Validate GPU Setup
```powershell
python check.py
```
✅ If successful, you'll see GPU device detected.  
❌ If it fails, the DirectML plugin isn't active — re-run step 2.

#### 4. Run Training Scripts
All training scripts automatically detect GPU and use it:
```powershell
python train_detection.py     # Train detection model
python train_segmentation.py  # Train segmentation model
python train_classifier.py    # Train classifier
python train_gan.py           # Train GAN
```

**Resume from Checkpoint**: If interrupted, scripts automatically resume from the latest checkpoint in `artifacts/checkpoints/`.

---

## 📊 Training Pipeline

Follow the phases in order for best results:

| Phase | Task | Script |
|-------|------|--------|
| 1 | Setup & validation | `check.py` |
| 2 | Data preparation | `prepare_brats_gli.py`, `preprocess_data.py` |
| 3 | Baseline models | `train_detection.py`, `train_classifier.py` |
| 4–6 | GAN training & augmentation | `train_gan.py` |
| 7 | Deployment & export | `build_submission_package.py` |
| 8 | Validation & submission | `validate_submission.py` |

### Compare Baseline vs. Augmented
After generating synthetic data with the GAN:
```bash
python compare_runs.py
```
Compares metrics from `artifacts/` JSON files to show augmentation impact.

---

## 📁 Input Data Format

**Expected structure** for `data/raw/`:
```
data/raw/
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000_t1.nii.gz
│   ├── BraTS-GLI-00000-000_t2.nii.gz
│   ├── BraTS-GLI-00000-000_t1ce.nii.gz
│   ├── BraTS-GLI-00000-000_flair.nii.gz
│   └── BraTS-GLI-00000-000_seg.nii.gz
├── BraTS-GLI-00001-000/
...
```

After preprocessing, `data/processed/` will contain PNG slices organized by case and modality.

---

## 🔧 Key Scripts

| Script | Purpose |
|--------|---------|
| `check.py` | Verify GPU/DirectML setup |
| `prepare_brats_gli.py` | Extract MRI modalities from NIFTI |
| `preprocess_data.py` | Normalize and slice MRI volumes to PNG |
| `train_detection.py` | Train CNN detector (EfficientNet-b0) |
| `train_segmentation.py` | Train segmentation model (U-Net) |
| `train_classifier.py` | Train binary classifier on processed data |
| `train_gan.py` | Train Conditional GAN for synthetic MRI generation |
| `dataset_report.py` | Generate dataset statistics |
| `compare_runs.py` | Compare baseline vs. augmented metrics |
| `build_submission_package.py` | Create submission archives |
| `validate_submission.py` | Validate submission package contents |
| `streamlit_app.py` | Interactive visualization dashboard |

---

## 📈 Outputs & Artifacts

Training outputs are saved to `artifacts/`:
- **Checkpoints**: Model weights in `.keras` format
- **Plots**: Training curves, confusion matrices
- **Metrics**: JSON files with evaluation results
- **Reports**: Dataset analysis and performance summaries

---

## 🐛 Troubleshooting

### No GPU Detected
1. Ensure you're using Python 3.10 in `.venv_gpu`
2. Verify DirectML plugin installation: `pip list | grep directml`
3. Re-install: `pip uninstall tensorflow-directml && pip install tensorflow-directml==0.4.0.dev230202`

### NIFTI Conversion Fails
- Check that `.nii.gz` files exist in `data/raw/BraTS-GLI-*/`
- Verify nibabel is installed: `pip install nibabel`

### Out of Memory (VRAM)
- Reduce batch size in training scripts
- Use a smaller model variant or reduce input resolution

---

## 📝 Documentation

Detailed phase guides are in `docs/`:
- **[Phase 1: Setup](docs/phase_1_setup.md)** — Environment & requirements
- **[Phase 2: Data Prep](docs/phase_2_data_preparation.md)** — Data pipeline
- **[Phase 3: Baselines](docs/phase_3_baselines.md)** — Baseline models
- **[Phases 4–6](docs/phase_4_5_6_plan.md)** — GAN & augmentation strategy
- **[Phase 7: Deployment](docs/phase_7_deployment.md)** — Model export & serving
- **[Phase 8: Validation](docs/phase_8_final_check.md)** — Final checks

---

## 📦 Dependencies

See [requirements.txt](requirements.txt) for all packages. Key dependencies:
- **TensorFlow** 2.10 (with DirectML for Windows GPU)
- **OpenCV** (image processing)
- **NumPy** (numerical operations)
- **Streamlit** (interactive dashboard)
- **Nibabel** (NIFTI file I/O)

---

## 🤝 Contributing

Contributions welcome! Please follow the existing code structure and document changes in the respective phase docs.

---

## ⚖️ License

This project is part of the Global Brain Tumour GAN Challenge. See challenge terms for usage rights.

---

## 📧 Support

For issues or questions:
1. Check the [troubleshooting section](#-troubleshooting)
2. Review phase-specific documentation in `docs/`
3. Validate GPU setup with `python check.py`

