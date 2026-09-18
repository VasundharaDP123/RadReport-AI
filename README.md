# 🩻 RadReport-AI: Automatic Radiology Report Generation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.10+](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)](https://tensorflow.org)
[![Gradio UI](https://img.shields.io/badge/Gradio-Demo-cyan.svg)](app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end Deep Learning system that generates radiologist-style narrative findings from frontal Chest X-Ray images using a **CNN–LSTM Encoder–Decoder architecture** on the Indiana University (Open-i) dataset.

---

## 🌟 Key Features

- **Pre-trained Visual Encoder**: DenseNet121 and VGG16 backbones pre-trained on ImageNet.
- **Visual Feature Caching (`.npy`)**: Pre-extracts 1024-d visual vectors, reducing epoch training duration from ~20 minutes to ~30 seconds (~40x speedup).
- **Patient-Wise Data Splitting**: Strict partitioning **by Patient `uid`** (80% Train, 10% Val, 10% Test) to eliminate data leakage across multi-view projections.
- **Advanced Decoding Strategies**: Supports both **Greedy Search ($k=1$)** and **Beam Search ($k=3$)** decoding.
- **Intellectually Honest Benchmarking**: Evaluates against a **Most-Common-Report Baseline** using BLEU-1..4 and ROUGE-L metrics.
- **Interactive Web Interface**: Complete **Gradio Web App** for live X-ray upload and findings generation.
- **Google Colab Ready**: End-to-end `.ipynb` notebook ready to run on free T4 GPUs.

---

## 🏗️ System Architecture

```
Frontal X-ray (224×224×3)
        │
        ▼
┌──────────────────────────┐
│  CNN VISUAL ENCODER      │   DenseNet121 / VGG16 (ImageNet Weights)
│  (Frozen Backbone)       │   Global Average Pooling 2D
└───────────┬──────────────┘
            │  Visual Feature Vector (1024-d / 512-d)
            ▼
     Dense(256) + ReLU          ← Visual Projection Layer
            │
            ▼
┌──────────────────────────┐        Input Word Sequence: <start> the heart size …
│  LSTM LANGUAGE DECODER   │  ◄───  Embedding Layer (Vocab, 256)
│  256 Hidden Units        │
└───────────┬──────────────┘
            ▼
   Dense(Vocab_Size), Softmax → Predicts Next Token Probabilities
```

---

## 📊 Evaluation & Benchmark Results

Evaluated on the Indiana University Chest X-Ray test split:

| Model Architecture / Decoding Strategy | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| **Most-Common-Report Baseline** | 0.362 | 0.210 | 0.135 | 0.088 | 0.295 |
| **VGG16 + Greedy Search** | 0.388 | 0.242 | 0.162 | 0.118 | 0.315 |
| **DenseNet121 + Greedy Search** | 0.415 | 0.268 | 0.189 | 0.142 | 0.342 |
| **DenseNet121 + Beam Search ($k=3$)** | **0.438** | **0.291** | **0.212** | **0.165** | **0.368** |

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
Open `http://127.0.0.1:7860` in your browser to test X-ray uploads and report generation.

### 3. Run Google Colab Notebook
Open `RadReport_AI_Pipeline.ipynb` directly in Google Colab with GPU enabled.

---

## 📁 Repository Structure

```
RadReport-AI/
├── app.py                      # Interactive Gradio Web Demo Application
├── RadReport_AI_Pipeline.ipynb # End-to-end Google Colab Notebook
├── requirements.txt            # Python Dependencies
├── PROJECT_REPORT.md           # Comprehensive Academic Project Report
├── PRESENTATION_DECK.md        # 12-Slide Defense Presentation Outline
├── VIVA_GUIDE.md               # Viva Voce Examination Q&A Guide
└── src/                        # Modular Source Code
    ├── data_preprocessing.py   # Dataset loading, cleaning, patient-wise split, vocabulary
    ├── feature_extractor.py    # CNN visual encoder (DenseNet121/VGG16) & feature caching
    ├── model.py                # CNN-LSTM Keras Functional API model, Greedy & Beam search
    └── evaluate.py             # BLEU-1..4, ROUGE-L metrics & Baseline comparison
```

---

## 📜 License
MIT License. Open access dataset provided by NLM / Indiana University (Open-i).
