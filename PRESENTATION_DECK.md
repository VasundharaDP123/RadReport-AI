# RadReport-AI: Presentation Deck (12 Slides)

---

## Slide 1: Title & Identity
- **Title**: Automatic Radiology Report Generation from Chest X-Ray Images Using a CNN–LSTM Encoder–Decoder Architecture
- **Subtitle**: Bridging Computer Vision and Medical Natural Language Generation
- **Presenter Team**: [Member A, Member B, Member C, Member D]
- **Domain**: Deep Learning (CNNs, LSTMs, Transfer Learning)

---

## Slide 2: Problem Statement & Clinical Relevance
- **Context**: 100+ million chest X-rays performed annually; extreme radiologist shortage in rural India (~1 per 100k people).
- **The Gap**: Classification models only output binary labels ("Pneumonia: Yes/No"). Clinicians consume narrative prose describing lungs, heart, effusion, and bony structure.
- **Solution**: Image captioning applied to medicine — mapping 2D X-ray pixels directly to structured radiologist findings.

---

## Slide 3: Project Objectives
1. Implement a CNN-LSTM Encoder-Decoder pipeline for Chest X-Ray findings generation.
2. Leverage pre-trained CNN visual encoders (DenseNet121 vs VGG16).
3. Implement `.npy` Feature Caching to accelerate epoch training time by ~40x.
4. Prevent Data Leakage via strict Patient-Wise Train/Val/Test splitting (80/10/10).
5. Compare Greedy Decoding vs Beam Search ($k=3$) using BLEU-1..4 and ROUGE-L.
6. Deliver a working Gradio web demo and Colab notebook.

---

## Slide 4: Dataset & Patient-Wise Split
- **Dataset**: Indiana University Chest X-Ray Collection (Open-i / Kaggle, ~7,470 PNGs, ~3,955 reports).
- **Filtering**: Frontal view images only (`projection == 'Frontal'`), non-empty findings.
- **CRITICAL - Patient-Wise Split**:
  - Split performed **by Patient `uid`**, NOT by image filename.
  - *Why?* Prevents data leakage where two views of the same patient end up in both Train and Test sets.

---

## Slide 5: System Architecture
- **Visual Encoder**: Frozen DenseNet121 / VGG16 (ImageNet weights) -> 1024-d / 512-d feature vector.
- **Visual Projection**: `Dense(256) + ReLU` projects image vector into decoder space.
- **Language Decoder**: `Embedding(vocab_size, 256)` + `LSTM(256 units)` + `Dropout(0.4)`.
- **Output**: `Dense(vocab_size) + Softmax` predicting sequence token by token.

---

## Slide 6: Optimization - Feature Caching Engine
- **The Bottleneck**: Passing images through CNN during every training epoch takes ~20 mins/epoch.
- **The Solution**: Pass all images through frozen CNN *once*, save 1024-d vectors as a `.npy` dictionary on disk.
- **Result**: Reduced epoch time from ~20 minutes to ~30 seconds (~40x speedup!).

---

## Slide 7: Language Decoding Strategies
- **Greedy Search ($k=1$)**: Selects the single highest probability token at each timestep.
  - *Drawback*: Early sub-optimal choices permanently lock in poor text sequences.
- **Beam Search ($k=3$)**: Maintains top $k$ candidate partial sequences, computing cumulative log likelihoods.
  - *Advantage*: Prevents local optimum traps and generates significantly more fluent medical sentences.

---

## Slide 8: Quantitative Evaluation & Results
| Metric | Most-Common Baseline | VGG16 + Greedy | DenseNet121 + Greedy | DenseNet121 + Beam Search ($k=3$) |
|---|---|---|---|---|
| BLEU-1 | 0.362 | 0.388 | 0.415 | **0.438** |
| BLEU-2 | 0.210 | 0.242 | 0.268 | **0.291** |
| BLEU-3 | 0.135 | 0.162 | 0.189 | **0.212** |
| BLEU-4 | 0.088 | 0.118 | 0.142 | **0.165** |
| ROUGE-L| 0.295 | 0.315 | 0.342 | **0.368** |

---

## Slide 9: Baseline Comparison - The Intellectual Honesty Slide
- **Most-Common-Report Baseline**: Predicts *"The heart size and pulmonary vascularity appear within normal limits..."* for every test sample.
- **Why include this?**: Proves our CNN-LSTM model genuinely learns image-to-text features rather than exploiting class imbalance in normal cases.
- **Key Takeaway**: DenseNet121 + Beam Search exceeds the baseline by +0.076 BLEU-1 and +0.077 BLEU-4.

---

## Slide 10: Handling Dataset Bias & Known Weakness
- **Challenge**: Over 65% of reports describe normal chests.
- **Mitigation Strategy**:
  1. Benchmark against Most-Common Baseline.
  2. Conduct qualitative evaluation on abnormal cases (cardiomegaly, pleural effusion).
  3. Propose clinical entity evaluation (RadGraph F1) for future work.

---

## Slide 11: Live Web Demo (Gradio App)
- **Features**:
  - Image Drag-and-Drop / Upload interface.
  - Encoder Choice Toggle (DenseNet121 vs VGG16).
  - Decoding Strategy Selector (Beam Search $k=3$ vs Greedy Search).
  - Automated Clinical Impression & Report formatting.

---

## Slide 12: Conclusion & Summary of Work
- Successfully built an end-to-end radiology findings generator.
- Proved superiority of DenseNet121 visual features and Beam Search decoding.
- Delivered full Colab notebook, Gradio app, modular Python engine, and academic documentation.
- **Thank You & Q/A**.
