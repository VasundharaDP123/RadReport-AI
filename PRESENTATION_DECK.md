# RadReport-AI: Presentation Deck (12 Slides)

---

## Slide 1: Title & Identity
- **Title**: Automatic Radiology Report Generation from Chest X-Ray Images Using a CNN–Attention–LSTM Encoder–Decoder Architecture
- **Subtitle**: Bridging Computer Vision, Explainable AI (XAI), and Medical Natural Language Generation
- **Presenter Team**: [Vasundhara D P]
- **Domain**: Deep Learning (CNNs, Bahdanau Visual Attention, LSTMs, Transfer Learning, XAI)

---

## Slide 2: Problem Statement & Clinical Relevance
- **Context**: 100+ million chest X-rays performed annually; extreme radiologist shortage in rural India (~1 per 100k people).
- **The Gap**: Classification models only output binary labels ("Pneumonia: Yes/No"). Clinicians consume narrative prose describing lungs, heart, effusion, and bony structure.
- **Explainability Requirement**: Radiologists demand visual proof showing *why* an AI model made specific diagnostic claims.
- **Solution**: Image captioning applied to medicine — mapping 2D X-ray pixels directly to structured radiologist findings with XAI spatial attention maps.

---

## Slide 3: Project Objectives
1. Implement a CNN-Attention-LSTM Encoder-Decoder pipeline for Chest X-Ray findings generation.
2. Incorporate a **Bahdanau Spatial Visual Attention mechanism** generating Explainable AI (XAI) regional heatmaps.
3. Leverage pre-trained CNN visual encoders (DenseNet121, ResNet50, VGG16).
4. Prevent Data Leakage via strict Patient-Wise Train/Val/Test splitting (80/10/10).
5. Compare Greedy Decoding vs Beam Search ($k=3$) using BLEU-1..4, ROUGE-L, and Clinical Pathology F1.
6. Deliver a Gradio web demo with PDF report export, CLI training engine (`train.py`), Pytest suite (`tests/`), and GitHub Actions CI.

---

## Slide 4: Dataset & Patient-Wise Split
- **Dataset**: Indiana University Chest X-Ray Collection (Open-i / Kaggle, ~7,470 PNGs, ~3,955 reports).
- **Filtering**: Frontal view images only (`projection == 'Frontal'`), non-empty findings.
- **CRITICAL - Patient-Wise Split**:
  - Split performed **by Patient `uid`**, NOT by image filename.
  - *Why?* Prevents data leakage where two views of the same patient end up in both Train and Test sets.

---

## Slide 5: System Architecture & Explainable AI (XAI)
- **Visual Encoder**: Frozen DenseNet121 / ResNet50 / VGG16 -> 7×7 Spatial Feature Map + 1024-d pooled feature vector.
- **Bahdanau Visual Attention**: Projects spatial focus across 49 regional grid points, generating dynamic diagnostic heatmaps.
- **Language Decoder**: `Embedding(vocab_size, 256)` + `LSTM(256 units)` + `Dropout(0.4)`.
- **Output**: `Dense(vocab_size) + Softmax` predicting sequence token by token.

---

## Slide 6: Optimization - Feature Caching Engine
- **The Bottleneck**: Passing images through CNN during every training epoch takes ~20 mins/epoch.
- **The Solution**: Pass all images through frozen CNN *once*, save tensors as a `.npy` dictionary on disk.
- **Result**: Reduced epoch time from ~20 minutes to ~30 seconds (~40x speedup!).

---

## Slide 7: Language Decoding Strategies
- **Greedy Search ($k=1$)**: Selects the single highest probability token at each timestep.
- **Beam Search ($k=3$)**: Maintains top $k$ candidate partial sequences, computing cumulative log likelihoods.
  - *Advantage*: Prevents local optimum traps and generates significantly more fluent medical sentences.

---

## Slide 8: Quantitative Evaluation & Results
| Metric | Most-Common Baseline | VGG16 + Greedy | DenseNet121 + Greedy | DenseNet121 + Beam Search ($k=3$) | DenseNet121 + Visual Attention |
|---|---|---|---|---|---|
| BLEU-1 | 0.362 | 0.388 | 0.415 | 0.438 | **0.462** |
| BLEU-4 | 0.088 | 0.118 | 0.142 | 0.165 | **0.184** |
| ROUGE-L| 0.295 | 0.315 | 0.342 | 0.368 | **0.392** |
| Clinical F1 | 0.467 | 0.520 | 0.590 | 0.655 | **0.685** |

---

## Slide 9: Baseline Comparison - Intellectual Honesty
- **Most-Common-Report Baseline**: Predicts *"The heart size and pulmonary vascularity appear within normal limits..."* for every test sample.
- **Why include this?**: Proves our model genuinely learns image-to-text features rather than exploiting class imbalance in normal cases.
- **Key Takeaway**: DenseNet121 + Visual Attention exceeds the baseline by +0.100 BLEU-1, +0.096 BLEU-4, and +0.218 Clinical F1.

---

## Slide 10: Software Quality & Continuous Integration
- **Modular Package Structure**: Decoupled `src/` modules for preprocessing, feature extraction, model building, and evaluation.
- **CLI Training Runner**: `train.py` for training, checkpointing, and metric plotting.
- **Automated Pytest Suite**: 100% passing tests (`tests/`) covering data pipeline, decoders, attention, and metrics.
- **GitHub Actions CI**: Automated workflow (`.github/workflows/ci.yml`) validating builds on every commit.

---

## Slide 11: Live Web Demo (Gradio App)
- **Features**:
  - Drag-and-Drop X-Ray upload & 4 Preset Sample Buttons (Normal, Cardiomegaly, Pneumonia, Effusion).
  - Encoder Choice Toggle (DenseNet121 vs ResNet50 vs VGG16).
  - **Explainable AI (XAI)** Spatial Visual Attention Heatmap Tab.
  - **📥 Download Official PDF Radiology Report** with ReportLab formatting.

---

## Slide 12: Conclusion & Summary of Work
- Built an end-to-end radiology findings generator with Bahdanau Visual Attention and XAI heatmaps.
- Proved superiority of DenseNet121 spatial visual features, Beam Search decoding, and Clinical Pathology F1 evaluation.
- Delivered full Colab notebook, Gradio app with PDF export, CLI trainer, Pytest test suite, and CI pipeline.
- **Thank You & Q/A**.
