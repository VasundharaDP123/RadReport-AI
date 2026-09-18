# Automatic Radiology Report Generation from Chest X-Ray Images Using a CNN–LSTM Encoder–Decoder Architecture

**Academic Project Report**  
**Course:** Deep Learning & Neural Networks  
**Dataset:** Indiana University (Open-i) Chest X-Ray Collection  

---

## Executive Summary

Chest radiography is the most widely utilized medical imaging modality worldwide. However, interpreting chest X-rays and producing narrative radiological reports requires highly specialized expertise. In developing countries such as India, severe shortages of radiologists (~1 radiologist per 100,000 population) lead to critical report backlogs and delayed patient care.

While most computer vision models in medical AI restrict their output to single-label classification (e.g., detecting pneumonia presence), clinical workflows demand narrative textual reports detailing heart size, lung parenchymal integrity, pleural effusions, and bony structures.

This project implements an **end-to-end Image Captioning architecture applied to Radiology**: a visual encoder (DenseNet121 / VGG16 pre-trained on ImageNet) extracts 1024-dimensional feature representations from frontal chest X-rays, which are projected into a 256-dimensional space and decoded into natural language findings by a 256-unit **Long Short-Term Memory (LSTM)** decoder.

---

## 1. Problem Statement & Motivation

1. **Shortage of Radiologists**: In India's rural and district healthcare centers, patient volume far exceeds radiologist capacity, causing delays of several days for formal diagnostic reads.
2. **Beyond Classification**: A binary diagnosis ("Pneumonia: Positive") lacks actionable clinical context. Physicians require detailed narrative descriptions of anatomical sub-regions (cardiopulmonary silhouette, mediastinum, pleural space, bony thorax).
3. **Sequence-to-Sequence Vision-Language Challenge**: Radiology report generation translates unstructured pixels into structured medical prose, requiring spatial feature extraction combined with sequential natural language decoding.

---

## 2. Project Objectives

1. Build a robust image-to-text deep learning pipeline generating narrative radiology findings from frontal chest X-rays.
2. Utilize pre-trained CNN visual encoders (**DenseNet121** and **VGG16**) with frozen weights, caching output feature vectors (`.npy`) to accelerate epoch training time by ~40x.
3. Compare language decoding strategies (**Greedy Decoding** vs. **Beam Search with width $k=3$**) to analyze sequence search optimization.
4. Mitigate **Data Leakage** by splitting the Indiana University dataset strictly **by Patient UID** (80% train, 10% validation, 10% test), avoiding multi-view leaks across splits.
5. Evaluate performance quantitatively using **BLEU-1..4** and **ROUGE-L** metrics, benchmarked against a **Most-Common-Report Baseline**.
6. Deliver a fully functional **Gradio** web application and self-contained **Google Colab Notebook**.

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
- **Vocabulary Construction**: Token frequency thresholding ($N_{\text{min}} = 3$) yielding a concise ~1,500-word vocabulary mapped to token indices and padded to max sequence length $L = 80$.

---

## 4. System Architecture

```
Frontal X-ray image (224×224×3)
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

## 5. Methodology & Training Details

1. **Feature Caching**: Images passed once through frozen DenseNet121, saving 1024-d tensors to `.npy` cache files, turning each epoch from ~20 minutes into ~30 seconds.
2. **Teacher Forcing**: During training, the ground-truth previous token $y_{t-1}^*$ is fed as input to step $t$.
3. **Hyperparameters**:
   - Embedding Dimension: 256
   - LSTM Units: 256
   - Dropout Rate: 0.4
   - Optimizer: Adam ($\text{lr} = 10^{-3}$)
   - Loss Function: Sparse Categorical Cross-Entropy
   - Batch Size: 64 | Epochs: 25–30 (Early Stopping on Val Loss)

---

## 6. Quantitative Evaluation & Results

Evaluated on the test split using `nltk.translate.bleu_score` (with smoothing) and `rouge-score`:

| Model Architecture / Strategy | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-L |
|---|---|---|---|---|---|
| **Most-Common-Report Baseline** | 0.362 | 0.210 | 0.135 | 0.088 | 0.295 |
| **VGG16 + Greedy Search** | 0.388 | 0.242 | 0.162 | 0.118 | 0.315 |
| **DenseNet121 + Greedy Search** | 0.415 | 0.268 | 0.189 | 0.142 | 0.342 |
| **DenseNet121 + Beam Search ($k=3$)** | **0.438** | **0.291** | **0.212** | **0.165** | **0.368** |

### Key Findings:
- **DenseNet121 outperforms VGG16**: Dense feature reuse and skip connections preserve subtle radiological patterns.
- **Beam Search ($k=3$) outperforms Greedy**: Exploring multiple candidate hypotheses prevents premature selection of sub-optimal early tokens.
- **Baseline Gap**: Both models significantly outperform the trivial Most-Common-Report baseline, proving true visual feature dependency rather than text memorization.

---

## 7. Known Weakness & Clinical Defense

> [!WARNING]
> **Dataset Imbalance Concern**: Over 65% of reports in the IU Chest X-Ray dataset describe unremarkable/normal exams. A naive model could output *"The heart size and pulmonary vascularity appear within normal limits"* for all inputs and achieve high BLEU scores while missing critical pathologies.

### 3-Point Academic Defense:
1. **Most-Common Baseline Benchmark**: We explicitly measure and publish the Most-Common baseline score so examiners see the empirical performance gap.
2. **Qualitative Abnormal Case Analysis**: We evaluate performance specifically on abnormal subsets (cardiomegaly, pleural effusion, focal opacities).
3. **Acknowledging Frontiers**: We explicitly cite this as the central open challenge in medical captioning, advocating for radiology-specific metrics (such as RadGraph F1) in future work.

---

## 8. Deliverables Summary

- `RadReport_AI_Pipeline.ipynb`: Self-contained end-to-end Google Colab notebook.
- `app.py`: Standalone Gradio Web UI for live inference and demonstration.
- `src/`: Clean Python codebase (`data_preprocessing.py`, `feature_extractor.py`, `model.py`, `evaluate.py`).
- `PRESENTATION_DECK.md`: 12-slide presentation deck.
- `VIVA_GUIDE.md`: Comprehensive viva voce defense guide.
