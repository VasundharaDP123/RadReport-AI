import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.data_preprocessing import RadiologyDataPreprocessor
from src.feature_extractor import ImageFeatureExtractor
from src.model import RadiologyReportGenerator
from src.evaluate import RadiologyEvaluator

def run_training_pipeline(args):
    """
    Executes complete end-to-end model training, feature caching, evaluation,
    and checkpoint saving pipeline for RadReport-AI.
    """
    print("=" * 70)
    print(f"[Start] RadReport-AI Model Training | Backbone: {args.encoder.upper()}")
    print("=" * 70)

    # 1. Load and merge dataset
    preprocessor = RadiologyDataPreprocessor(min_word_count=args.min_freq, max_seq_len=args.max_len)
    df = preprocessor.load_and_merge(args.reports_csv, args.projections_csv)

    # 2. Patient-wise Split
    train_df, val_df, test_df = preprocessor.patient_wise_split(df)
    word2idx, idx2word = preprocessor.build_vocabulary(train_df['cleaned_findings'])

    # 3. Extract and cache visual features
    extractor = ImageFeatureExtractor(architecture=args.encoder)
    cache_path = os.path.join(args.output_dir, f"{args.encoder.lower()}_features_cache.npy")
    feature_dict = extractor.cache_features(df, image_dir=args.image_dir, cache_output_path=cache_path)

    # 4. Prepare batch arrays for training (Teacher Forcing)
    print("Preparing training matrices...")
    X_train_img, X_train_seq, Y_train = [], [], []

    pad_id = word2idx[preprocessor.PAD_TOKEN]
    start_id = word2idx[preprocessor.START_TOKEN]
    end_id = word2idx[preprocessor.END_TOKEN]

    for _, row in train_df.iterrows():
        fn = row['filename']
        text = row['cleaned_findings']
        feat = feature_dict.get(fn, np.zeros(extractor.feature_dim))
        
        tokens = text.split()
        token_ids = [word2idx.get(t, word2idx[preprocessor.UNK_TOKEN]) for t in tokens]
        
        # Build full sequence input (<start> ... text ...) and target sequence (... text ... <end>)
        in_seq = token_ids[:-1] if len(token_ids) > 1 else [start_id]
        out_seq = token_ids[1:] if len(token_ids) > 1 else [end_id]
        
        # Pad to max_len
        if len(in_seq) < args.max_len:
            in_seq = in_seq + [pad_id] * (args.max_len - len(in_seq))
        else:
            in_seq = in_seq[:args.max_len]
            
        if len(out_seq) < args.max_len:
            out_seq = out_seq + [pad_id] * (args.max_len - len(out_seq))
        else:
            out_seq = out_seq[:args.max_len]

        X_train_img.append(feat)
        X_train_seq.append(in_seq)
        Y_train.append(out_seq)

    X_train_img = np.array(X_train_img)
    X_train_seq = np.array(X_train_seq)
    Y_train = np.array(Y_train)

    print(f"Training Data Tensor Shapes -> Images: {X_train_img.shape}, Sequences: {X_train_seq.shape}, Targets: {Y_train.shape}")

    # 5. Build Model
    generator = RadiologyReportGenerator(
        vocab_size=preprocessor.vocab_size,
        max_seq_len=args.max_len,
        feature_dim=extractor.feature_dim,
        embed_dim=256,
        lstm_units=256
    )
    generator.build_model()

    os.makedirs(args.output_dir, exist_ok=True)
    weights_path = os.path.join(args.output_dir, f"radreport_{args.encoder.lower()}_weights.h5")

    # 6. Fit Model
    print(f"Training for {args.epochs} epochs with batch size {args.batch_size}...")
    history = generator.model.fit(
        [X_train_img, X_train_seq],
        Y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.1,
        verbose=1
    )

    # 7. Save Model Weights
    generator.model.save_weights(weights_path)
    print(f"[OK] Successfully saved trained model weights to: {weights_path}")

    # 8. Plot Training Loss and Accuracy
    plot_path = os.path.join(args.output_dir, f"{args.encoder.lower()}_training_curves.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='Train Loss', color='#0284c7', linewidth=2)
    if 'val_loss' in history.history:
        ax1.plot(history.history['val_loss'], label='Val Loss', color='#e11d48', linestyle='--', linewidth=2)
    ax1.set_title(f"Training Loss ({args.encoder.upper()})")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(history.history['accuracy'], label='Train Accuracy', color='#16a34a', linewidth=2)
    if 'val_accuracy' in history.history:
        ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='#ca8a04', linestyle='--', linewidth=2)
    ax2.set_title(f"Training Accuracy ({args.encoder.upper()})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved training metric curves plot to: {plot_path}")

    # 9. Evaluate Test Split
    print("\nRunning Evaluation on Test Set...")
    evaluator = RadiologyEvaluator()
    test_references = test_df['cleaned_findings'].tolist()
    
    test_hypotheses = []
    for _, row in test_df.iterrows():
        fn = row['filename']
        feat = feature_dict.get(fn, np.zeros(extractor.feature_dim))
        hyp = generator.generate_report_beam_search(feat, word2idx, idx2word, beam_width=3)
        test_hypotheses.append(hyp)

    scores = evaluator.evaluate_corpus(test_references, test_hypotheses)
    baseline_scores = evaluator.compute_most_common_baseline(train_df['cleaned_findings'].tolist(), test_references)

    benchmark = {
        f"Baseline (Most Common Report)": baseline_scores,
        f"{args.encoder.upper()} + Beam Search (k=3)": scores
    }
    
    print("\n" + "=" * 70)
    print(evaluator.generate_benchmark_table(benchmark))
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RadReport-AI Model Training and Evaluation CLI")
    parser.add_argument("--reports_csv", type=str, default="indiana_reports.csv", help="Path to Indiana reports CSV")
    parser.add_argument("--projections_csv", type=str, default="indiana_projections.csv", help="Path to Indiana projections CSV")
    parser.add_argument("--image_dir", type=str, default="images/", help="Directory containing Chest X-ray images")
    parser.add_argument("--encoder", type=str, default="densenet121", choices=["densenet121", "vgg16", "resnet50"], help="CNN visual backbone")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--max_len", type=int, default=80, help="Maximum caption sequence length")
    parser.add_argument("--min_freq", type=int, default=3, help="Minimum word frequency threshold for vocabulary")
    parser.add_argument("--output_dir", type=str, default="models", help="Directory to save weights and plots")

    args = parser.parse_args()
    run_training_pipeline(args)
