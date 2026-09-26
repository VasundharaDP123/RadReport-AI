# RadReport-AI: Viva Voce Examination & Defense Guide

This guide prepares team members to answer faculty cross-questions during the viva voce examination.

---

## 1. Architecture & Deep Learning Fundamentals

### Q1: Why use an LSTM decoder instead of a traditional Vanilla RNN?
> **Answer:** Vanilla RNNs suffer severely from the **vanishing gradient problem** when processing long sequences. Radiology findings average 30–80 tokens in length. LSTMs utilize memory cells governed by three gates (Input Gate, Forget Gate, Output Gate) that preserve long-range token dependencies without gradient degradation.

### Q2: How does the Bahdanau Visual Attention mechanism work in this project?
> **Answer:** Bahdanau additive attention projects dynamic regional visual focus over 7×7 spatial feature maps extracted by the CNN encoder. At each decoding step $t$, the attention layer computes alignment scores between the LSTM hidden state $h_{t-1}$ and all 49 spatial grid locations, producing a normalized softmax weight distribution. These spatial weights form a visual context vector and generate an **Explainable AI (XAI)** diagnostic heatmap overlay on the Chest X-Ray.

### Q3: Why freeze the CNN encoder during feature extraction?
> **Answer:** The Indiana University dataset contains ~7,470 images, which is far too small to fine-tune millions of CNN parameters from scratch without severe overfitting. Pre-trained ImageNet features (DenseNet121 / ResNet50 / VGG16) extract universal low-level edges, textures, and high-level anatomical shapes effectively. Freezing the CNN also allowed us to **cache visual features** to `.npy` files, accelerating training per epoch by ~40x.

### Q4: What is Teacher Forcing and why is it necessary?
> **Answer:** Teacher Forcing is a training technique for sequence generation where the ground-truth target token $y_{t-1}^*$ is fed as input to the LSTM at step $t$, rather than the model's own predicted output $\hat{y}_{t-1}$. In early training epochs, model predictions are noisy; without Teacher Forcing, early prediction errors compound continuously, causing training instability and failure to converge.

### Q5: Why is DenseNet121 superior to VGG16 for this medical task?
> **Answer:** DenseNet121 features **dense connections** between layers, where each layer receives direct feature inputs from all preceding layers. This encourages feature reuse, mitigates vanishing gradients, and produces rich 1024-dimensional feature maps that preserve subtle radiological patterns (e.g., faint bibasilar infiltrates or small effusions) better than VGG16's 512-dimensional vector.

---

## 2. Dataset & Preprocessing

### Q6: What is Data Leakage in this project, and how did you prevent it?
> **Answer:** In the IU Chest X-Ray dataset, many patients have multiple image views (e.g., frontal and lateral). If we split data randomly by *image filename*, one view of Patient X could end up in the training set while another view of Patient X lands in the test set. The model would memorize Patient X's report, artificially inflating BLEU scores. We prevented this by splitting strictly **by Patient `uid`** (80% Train, 10% Val, 10% Test).

### Q7: Why filter for Frontal projection only?
> **Answer:** Restricting our pipeline to frontal projections (PA/AP) establishes a clean 1-to-1 mapping between one visual image and one report. Including lateral views without multi-view attention mechanisms would introduce ambiguity when mapping two images to a single text output.

---

## 3. Decoding Strategies & Search Algorithms

### Q8: How does Beam Search differ from Greedy Decoding?
> **Answer:**
> - **Greedy Decoding ($k=1$)**: At each timestep, selects the single token with the maximum softmax probability $\text{argmax}_w P(w|w_{<t}, \text{image})$. It is computationally fast but can fall into local optima if an early word choice is sub-optimal.
> - **Beam Search ($k=3$)**: Expands and maintains the top $k$ partial sequence hypotheses at each step, tracking cumulative log-probabilities. It selects the candidate sequence with the highest total log likelihood once `<end>` is generated, yielding significantly more coherent sentences.

---

## 4. Metrics, Evaluation & Intellectual Honesty

### Q9: What does BLEU actually measure, and why is BLEU-4 relatively low (~0.18)?
> **Answer:** BLEU (Bilingual Evaluation Understudy) measures $n$-gram precision between generated text and reference text with a brevity penalty. BLEU-4 measures 4-gram phrase matching. In radiology, BLEU-4 is around 0.15–0.18 because medical language allows multiple valid phrasing variations for the same clinical finding (e.g., *"no active disease"* vs. *"unremarkable chest radiograph"*). BLEU penalizes valid paraphrases.

### Q10: What is Clinical Pathology Entity Extraction F1?
> **Answer:** NLP BLEU scores measure exact word matching but can miss clinical accuracy. We implemented Clinical Pathology Entity Matching across 8 key radiological conditions (Cardiomegaly, Effusion, Atelectasis, Pneumonia, Opacity, Pneumothorax, Edema, Normal). Clinical F1 evaluates whether the AI correctly identifies the underlying medical pathologies present in the ground-truth report.

### Q11: What is the Most-Common-Report Baseline, and why did you include it?
> **Answer:** In medical datasets, ~65% of reports are normal chest X-rays. A naive baseline model that always prints *"The heart size and pulmonary vascularity appear within normal limits..."* achieves ~0.36 BLEU-1 without ever looking at the image. Including this baseline proves that our CNN-Attention-LSTM model genuinely extracts visual features and exceeds the baseline (+0.462 vs 0.362 BLEU-1).

---

## 5. Software Engineering & MLOps

### Q12: How are software quality and continuous integration ensured?
> **Answer:** We implemented a modular Python package architecture in `src/`, a CLI training runner (`train.py`), an automated **Pytest unit test suite** (`tests/`) covering preprocessing, feature extraction, model building, attention heatmaps, and metrics, and a **GitHub Actions CI workflow** (`.github/workflows/ci.yml`) that automatically runs linting and test suites on every commit.

### Q13: How does the PDF Report Export feature work?
> **Answer:** Using the ReportLab library, `app.py` dynamically formats generated findings into an official PDF medical report complete with hospital branding headers, patient demographics, examination timestamps, pathology status badges, and electronic signature blocks.
