# SAR-Colorization

A deep learning pipeline for colorizing Synthetic Aperture Radar (SAR) images using co-registered optical data from Sentinel-2. This project supports data acquisition, preprocessing, training, inference, and monitoring in a modular and scalable way.

---

## 📁 Project Structure

```bash
sar-colorization/
├── .venv/                   # Python virtual environment (already set up with uv)
├── data/
│   ├── raw/                 # Raw downloaded Sentinel-1 and Sentinel-2 images
│   └── processed/           # Preprocessed, aligned, and tiled image data
├── notebooks/
│   └── 01_data_exploration.ipynb  # Jupyter notebook for initial EDA and visualization
├── outputs/
│   ├── checkpoints/         # Model checkpoints saved during training
│   └── results/             # Output images and logs from inference
├── scripts/
│   ├── 01_fetch_data.py     # Download SAR and optical images via Sentinel Hub API
│   └── 02_preprocess_data.py # Co-registration, tiling, cloud masking, etc.
├── src/
│   ├── __init__.py
│   ├── config.py            # Configuration (paths, API keys, constants)
│   ├── data_loader.py       # Custom PyTorch Dataset and DataLoader
│   ├── losses.py            # Loss functions (e.g., perceptual loss, SSIM)
│   ├── model.py             # UNet/ResNet Generator & PatchGAN Discriminator
│   └── train.py             # PyTorch Lightning training loop
├── monitoring/
│   ├── docker-compose.yml   # Monitoring stack setup (Prometheus + Grafana)
│   └── prometheus/
│       └── prometheus.yml   # Prometheus scraping config
├── inference.py             # Inference script to run trained model on new SAR data
├── pyproject.toml           # Python project metadata and dependencies
└── README.md                # You're reading this file
