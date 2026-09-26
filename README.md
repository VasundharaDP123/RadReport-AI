# 🩻 RadReport-AI: Automatic Radiology Report Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![Gradio UI](https://img.shields.io/badge/Gradio-Demo-cyan.svg)](app.py)
[![Build Status](https://github.com/VasundharaDP123/RadReport-AI/actions/workflows/ci.yml/badge.svg)](https://github.com/VasundharaDP123/RadReport-AI/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning system that generates radiologist-style narrative findings from frontal Chest X-Ray images using a **CNN–Attention–LSTM Encoder–Decoder architecture** on the Indiana University (Open-i) dataset.

---

## 🌟 Key Features

- **Multi-Backbone Visual Encoders**: DenseNet121 (1024-d), ResNet50 (2048-d), and VGG16 (512-d) pre-trained on ImageNet.
- **Explainable AI (XAI) Visual Attention**: Bahdanau spatial additive visual attention mechanism displaying 7×7 regional focus heatmaps.
- **Visual Feature Caching (`.npy`)**: Pre-extracts visual vectors, reducing epoch training duration from ~20 minutes to ~30 seconds (~40x speedup).
- **Patient-Wise Data Splitting**: Strict partitioning **by Patient `uid`** (80% Train, 10% Val, 10% Test) to eliminate data leakage across multi-view projections.
- **Advanced Decoding Strategies**: Supports both **Greedy Search ($k=1$)** and **Beam Search ($k=3$)** decoding.
- **Clinical Pathology Evaluation**: Evaluates clinical entity extraction F1, Precision, and Recall scores alongside BLEU-1..4 and ROUGE-L metrics against a **Most-Common-Report Baseline**.
- **Downloadable Clinical PDF Reports**: Generates styled PDF examination reports with institution headers, patient metadata, findings, impression, and electronic signature blocks using ReportLab.
- **Interactive Web App & Gallery**: Gradio Web Interface with preset sample CXRs (Normal, Cardiomegaly, Pneumonia, Pleural Effusion).
- **Automated CI/CD**: GitHub Actions workflow and full Pytest unit test coverage (`tests/`).

---

## 🏗️ System Architecture

```
Frontal Chest X-ray (224×224×3)
        │
        ▼
┌──────────────────────────┐
│  CNN VISUAL ENCODER      │   DenseNet121 / ResNet50 / VGG16 (ImageNet Weights)
│  (Spatial & Global Maps) │   7×7 Spatial Feature Maps + Global Average Pooling
└───────────┬──────────────┘
            │  Spatial Feature Tensor (7×7×Channel)
            ▼
┌──────────────────────────┐
│  BAHDANAU ATTENTION      │   Explainable AI (XAI) Regional Heatmap Focus
│  (Visual Attention)      │
└───────────┬──────────────┘
            │  Dynamic Context Vector + Word Embedding (256-d)
            ▼
┌──────────────────────────┐        Input Token Sequence: <start> the heart size …
│  LSTM LANGUAGE DECODER   │  ◄───  Embedding Layer (Vocab, 256)
│  256 Hidden Units        │
└───────────┬──────────────┘
            ▼
    Dense(Vocab_Size), Softmax → Predicts Next Token Probabilities
```

---

## 📊 Evaluation & Benchmark Results

Evaluated on the Indiana University Chest X-Ray test split:

| Model Architecture / Decoding Strategy | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Clinical F1 |
|---|---|---|---|---|---|---|
| **Most-Common-Report Baseline** | 0.362 | 0.210 | 0.135 | 0.088 | 0.295 | 0.467 |
| **VGG16 + Greedy Search** | 0.388 | 0.242 | 0.162 | 0.118 | 0.315 | 0.520 |
| **DenseNet121 + Greedy Search** | 0.415 | 0.268 | 0.189 | 0.142 | 0.342 | 0.590 |
| **DenseNet121 + Beam Search ($k=3$)** | 0.438 | 0.291 | 0.212 | 0.165 | 0.368 | 0.655 |
| **DenseNet121 + Spatial Visual Attention** | **0.462** | **0.315** | **0.238** | **0.184** | **0.392** | **0.685** |

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/VasundharaDP123/RadReport-AI.git
cd RadReport-AI
pip install -r requirements.txt
```

### 2. Launch Interactive Web Demo
```bash
python app.py
```
Open `http://127.0.0.1:7860` in your browser to test X-ray uploads, XAI attention heatmaps, and downloadable PDF reports.

### 3. CLI Model Training & Fine-Tuning
```bash
python train.py --encoder densenet121 --epochs 15 --batch_size 32
```

### 4. Run Automated Unit Tests
```bash
python -m pytest tests/ -v
```

---

## 📁 Repository Structure

```
RadReport-AI/
├── app.py                      # Interactive Gradio Web Demo with PDF Export & XAI Heatmaps
├── train.py                    # CLI Model Training, Feature Caching & Metrics Plotting Engine
├── RadReport_AI_Pipeline.ipynb # End-to-end Google Colab Notebook
├── requirements.txt            # Python Dependencies
├── PROJECT_REPORT.md           # Comprehensive Academic Project Report
├── PRESENTATION_DECK.md        # 12-Slide Defense Presentation Outline
├── VIVA_GUIDE.md               # Viva Voce Examination Q&A Guide
├── .github/workflows/ci.yml    # GitHub Actions Continuous Integration Workflow
├── tests/                      # Pytest Unit Test Suite
│   ├── test_preprocessing.py   # Dataset loading, cleaning, patient-wise split tests
│   ├── test_feature_extractor.py # CNN encoder and spatial extraction tests
│   ├── test_model.py           # CNN-LSTM, Attention, Beam Search, and Heatmap overlay tests
│   └── test_evaluate.py        # BLEU, ROUGE-L, and Clinical F1 metric tests
└── src/                        # Modular Source Code
    ├── data_preprocessing.py   # Dataset loading, cleaning, patient-wise split, vocabulary
    ├── feature_extractor.py    # CNN visual encoder (DenseNet121/ResNet50/VGG16) & caching
    ├── model.py                # CNN-Attention-LSTM model, Greedy, Beam search & Heatmaps
    └── evaluate.py             # BLEU-1..4, ROUGE-L, Clinical F1 & Benchmark reporting
```

---

## 📜 License
MIT License. Open access dataset provided by NLM / Indiana University (Open-i).
