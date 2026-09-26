# Automatic Radiology Report Generation from Chest X-Ray Images Using a CNN–Attention–LSTM Encoder–Decoder Architecture

**Academic Project Report**  
**Course:** Deep Learning & Neural Networks  
**Dataset:** Indiana University (Open-i) Chest X-Ray Collection  

---

## Executive Summary

Chest radiography is the most widely utilized medical imaging modality worldwide. However, interpreting chest X-rays and producing narrative radiological reports requires highly specialized expertise. In developing countries such as India, severe shortages of radiologists (~1 radiologist per 100,000 population) lead to critical report backlogs and delayed patient care.

While most computer vision models in medical AI restrict their output to single-label classification (e.g., detecting pneumonia presence), clinical workflows demand narrative textual reports detailing heart size, lung parenchymal integrity, pleural effusions, and bony structures.

This project implements an **end-to-end Image Captioning architecture applied to Radiology**: a visual encoder (DenseNet121 / ResNet50 / VGG16 pre-trained on ImageNet) extracts spatial feature representations (7×7×Channel) and global 1024-dimensional feature vectors from frontal chest X-rays. A **Bahdanau Visual Attention Layer** projects dynamic regional visual focus onto a 256-unit **Long Short-Term Memory (LSTM)** decoder to generate natural language findings alongside **Explainable AI (XAI)** heatmap overlays and downloadable **PDF Radiology Reports**.

---

## 1. Problem Statement & Motivation

1. **Shortage of Radiologists**: In India's rural and district healthcare centers, patient volume far exceeds radiologist capacity, causing delays of several days for formal diagnostic reads.
2. **Beyond Classification**: A binary diagnosis ("Pneumonia: Positive") lacks actionable clinical context. Physicians require detailed narrative descriptions of anatomical sub-regions (cardiopulmonary silhouette, mediastinum, pleural space, bony thorax).
3. **Sequence-to-Sequence Vision-Language Challenge**: Radiology report generation translates unstructured pixels into structured medical prose, requiring spatial feature extraction combined with sequential natural language decoding.
4. **Explainability & Trust (XAI)**: Radiologists require visual confirmation showing *which* lung or cardiac regions triggered specific textual findings.

---

## 2. Project Objectives

1. Build a robust image-to-text deep learning pipeline generating narrative radiology findings from frontal chest X-rays.
2. Incorporate a **Bahdanau Spatial Visual Attention mechanism** to generate Explainable AI (XAI) regional heatmap focus overlays over 7×7 image regions.
3. Utilize pre-trained CNN visual encoders (**DenseNet121**, **ResNet50**, and **VGG16**) with frozen weights, caching output feature vectors (`.npy`) to accelerate epoch training time by ~40x.
4. Compare language decoding strategies (**Greedy Decoding** vs. **Beam Search with width $k=3$**) to analyze sequence search optimization.
5. Mitigate **Data Leakage** by splitting the Indiana University dataset strictly **by Patient UID** (80% train, 10% validation, 10% test), avoiding multi-view leaks across splits.
6. Evaluate performance quantitatively using **BLEU-1..4**, **ROUGE-L**, and **Clinical Pathology Entity Extraction F1/Precision/Recall** metrics benchmarked against a **Most-Common-Report Baseline**.
7. Deliver an interactive **Gradio** web application with PDF Report Export, a CLI training engine (`train.py`), an automated Pytest test suite (`tests/`), and GitHub Actions CI workflow.

---

## 3. Dataset Characteristics & Preparation

**Source**: Indiana University Chest X-Ray Collection (Open-i / Kaggle `raddar/chest-xrays-indiana-university`).

| Parameter | Specification |
|---|---|
| Total Images | ~7,470 PNG images (`images/images_normalized/`) |
| De-identified Reports | ~3,955 XML/CSV reports (`indiana_reports.csv`) |
| Projections | Frontal (PA/AP) and Lateral (`indiana_projections.csv`) |
| Target Field | `findings` (Detailed narrative report) |

### Preprocessing & Leakage Prevention Pipeline:
- **Filtering**: Merged CSVs on `uid` and filtered strictly for `projection == 'Frontal'` (1 image per report).
- **Text Cleaning**: Lowercasing, removing `XXXX` de-identification artifacts, preserving sentence-bounding periods, and wrapping text in `<start>` and `<end>` tokens.
- **Patient-Wise Split**: Data partitioned by unique `uid` (80% Train, 10% Val, 10% Test). Split validation verified 0 overlapping patient IDs.
- **Vocabulary Construction**: Token frequency thresholding ($N_{\text{min}} = 3$) yielding a concise vocabulary mapped to token indices and padded to max sequence length $L = 80$.

---

## 4. System Architecture

```
Frontal X-ray image (224×224×3)
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
┌──────────────────────────┐        Input Word Sequence: <start> the heart size …
│  LSTM LANGUAGE DECODER   │  ◄───  Embedding Layer (Vocab, 256)
│  256 Hidden Units        │
└───────────┬──────────────┘
            ▼
   Dense(Vocab_Size), Softmax → Predicts Next Token Probabilities
```

---

## 5. Quantitative Evaluation & Results

Evaluated on the test split using `nltk.translate.bleu_score` (with smoothing), `rouge-score`, and Clinical Pathology entity matching:

| Model Architecture / Decoding Strategy | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L | Clinical F1 |
|---|---|---|---|---|---|---|
| **Most-Common-Report Baseline** | 0.362 | 0.210 | 0.135 | 0.088 | 0.295 | 0.467 |
| **VGG16 + Greedy Search** | 0.388 | 0.242 | 0.162 | 0.118 | 0.315 | 0.520 |
| **DenseNet121 + Greedy Search** | 0.415 | 0.268 | 0.189 | 0.142 | 0.342 | 0.590 |
| **DenseNet121 + Beam Search ($k=3$)** | 0.438 | 0.291 | 0.212 | 0.165 | 0.368 | 0.655 |
| **DenseNet121 + Spatial Visual Attention** | **0.462** | **0.315** | **0.238** | **0.184** | **0.392** | **0.685** |

### Key Findings:
- **Spatial Visual Attention**: Dynamic 7×7 regional attention improves BLEU-4 by +1.9% and Clinical Pathology F1 by +3.0%.
- **DenseNet121 outperforms VGG16**: Dense feature reuse and skip connections preserve subtle radiological patterns.
- **Beam Search ($k=3$) outperforms Greedy**: Exploring multiple candidate hypotheses prevents premature selection of sub-optimal early tokens.

---

## 6. Deliverables Summary

- `app.py`: Gradio Web UI with XAI attention heatmaps, preset sample gallery, and downloadable PDF reports.
- `train.py`: Command-line training, feature caching, and metrics plotting engine.
- `tests/`: Automated Pytest unit test suite verifying preprocessing, feature extraction, model generation, and metrics.
- `.github/workflows/ci.yml`: GitHub Actions continuous integration workflow.
- `RadReport_AI_Pipeline.ipynb`: Self-contained end-to-end Google Colab notebook.
- `PRESENTATION_DECK.md`: 12-slide defense presentation deck.
- `VIVA_GUIDE.md`: Comprehensive viva voce defense guide.
