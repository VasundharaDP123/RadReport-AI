import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class RadiologyDataPreprocessor:
    """
    Handles loading, merging, cleaning, patient-wise splitting,
    and vocabulary building for the IU Chest X-Ray dataset.
    """
    def __init__(self, min_word_count=3, max_seq_len=80):
        self.min_word_count = min_word_count
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        self.PAD_TOKEN = "<pad>"
        self.UNK_TOKEN = "<unk>"
        self.START_TOKEN = "<start>"
        self.END_TOKEN = "<end>"

    def clean_text(self, text):
        """
        Cleans findings text:
        - Converts to lowercase
        - Strips 'XXXX' de-identification placeholders
        - Retains sentence periods while removing unnecessary punctuation
        - Adds start and end tags
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove XXXX placeholders common in Open-i de-identified reports
        text = re.sub(r'x{2,}', '', text)
        text = re.sub(r'\bxxxx\b', '', text)
        
        # Remove special characters except periods and spaces
        text = re.sub(r'[^a-z0-9\.\s]', ' ', text)
        
        # Fix multiple spaces and periods
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'\s+\.', '.', text)
        
        # Ensure space after period for proper tokenization
        text = re.sub(r'\.', ' . ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        if not text:
            return ""
            
        return f"{self.START_TOKEN} {text} {self.END_TOKEN}"

    def load_and_merge(self, reports_path, projections_path):
        """
        Loads indiana_reports.csv and indiana_projections.csv, merges on 'uid',
        filters for Frontal projection, and drops empty findings.
        """
        if not os.path.exists(reports_path) or not os.path.exists(projections_path):
            print(f"[Warning] Dataset files not found at {reports_path}. Generating synthetic demo dataset.")
            return self._generate_synthetic_dataset()

        reports = pd.read_csv(reports_path)
        projections = pd.read_csv(projections_path)

        # Merge on uid
        df = pd.merge(reports, projections, on='uid')

        # Filter Frontal projection only
        if 'projection' in df.columns:
            df = df[df['projection'].str.lower() == 'frontal'].copy()

        # Clean findings
        df['cleaned_findings'] = df['findings'].apply(self.clean_text)

        # Drop empty findings
        df = df[df['cleaned_findings'] != ""].copy()
        
        print(f"Loaded and merged dataset: {len(df)} frontal X-ray records.")
        return df

    def _generate_synthetic_dataset(self, num_samples=300):
        """
        Generates synthetic Radiology reports for pipeline verification and demo mode.
        """
        np.random.seed(42)
        sample_reports = [
            "the heart size and pulmonary vascularity appear within normal limits . no focal consolidation or pleural effusion .",
            "lungs are clear without focal infiltrate . cardiomegaly is noted with subtle vascular congestion .",
            "no acute cardiopulmonary abnormality . clear lungs and normal cardiac silhouette .",
            "mild bibasilar atelectasis . cardiac silhouette is mildly enlarged . no pneumothorax .",
            "right lower lobe opacification consistent with focal pneumonia . small right pleural effusion ."
        ]
        
        data = []
        for i in range(num_samples):
            uid = f"patient_{i // 2 + 1}"
            filename = f"image_{i+1}.png"
            text = sample_reports[i % len(sample_reports)]
            cleaned = f"{self.START_TOKEN} {text} {self.END_TOKEN}"
            data.append({
                'uid': uid,
                'filename': filename,
                'projection': 'Frontal',
                'findings': text,
                'cleaned_findings': cleaned
            })
        return pd.DataFrame(data)

    def patient_wise_split(self, df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
        """
        Splits dataset BY PATIENT UID to eliminate data leakage across views.
        """
        unique_uids = df['uid'].unique()
        train_uids, temp_uids = train_test_split(unique_uids, test_size=(val_ratio + test_ratio), random_state=42)
        
        relative_val_size = val_ratio / (val_ratio + test_ratio)
        val_uids, test_uids = train_test_split(temp_uids, test_size=(1.0 - relative_val_size), random_state=42)

        train_df = df[df['uid'].isin(train_uids)].copy()
        val_df = df[df['uid'].isin(val_uids)].copy()
        test_df = df[df['uid'].isin(test_uids)].copy()

        print(f"Patient-wise Split -> Train: {len(train_df)} ({len(train_uids)} patients), "
              f"Val: {len(val_df)} ({len(val_uids)} patients), "
              f"Test: {len(test_df)} ({len(test_uids)} patients)")

        return train_df, val_df, test_df

    def build_vocabulary(self, train_texts):
        """
        Builds word2idx and idx2word mappings based on min_word_count threshold.
        """
        word_counts = {}
        for text in train_texts:
            tokens = text.split()
            for token in tokens:
                word_counts[token] = word_counts.get(token, 0) + 1

        # Special tokens
        vocab = [self.PAD_TOKEN, self.UNK_TOKEN, self.START_TOKEN, self.END_TOKEN]
        
        # Add words meeting threshold
        for word, count in sorted(word_counts.items(), key=lambda x: -x[1]):
            if count >= self.min_word_count and word not in vocab:
                vocab.append(word)

        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab_size = len(vocab)

        print(f"Built Vocabulary: {self.vocab_size} words (min_freq={self.min_word_count}).")
        return self.word2idx, self.idx2word

    def text_to_sequence(self, text):
        """
        Converts a text string to a padded token index sequence.
        """
        tokens = text.split()
        seq = [self.word2idx.get(token, self.word2idx[self.UNK_TOKEN]) for token in tokens]
        
        # Truncate or Pad
        if len(seq) > self.max_seq_len:
            seq = seq[:self.max_seq_len]
        else:
            seq = seq + [self.word2idx[self.PAD_TOKEN]] * (self.max_seq_len - len(seq))
            
        return np.array(seq)

    def sequence_to_text(self, sequence):
        """
        Converts token index sequence back to readable text string.
        """
        words = []
        for idx in sequence:
            word = self.idx2word.get(idx, self.UNK_TOKEN)
            if word == self.END_TOKEN:
                break
            if word not in [self.PAD_TOKEN, self.START_TOKEN]:
                words.append(word)
        return " ".join(words)


if __name__ == "__main__":
    preprocessor = RadiologyDataPreprocessor()
    df = preprocessor.load_and_merge("indiana_reports.csv", "indiana_projections.csv")
    train_df, val_df, test_df = preprocessor.patient_wise_split(df)
    word2idx, idx2word = preprocessor.build_vocabulary(train_df['cleaned_findings'])
    sample_seq = preprocessor.text_to_sequence(train_df['cleaned_findings'].iloc[0])
    sample_recon = preprocessor.sequence_to_text(sample_seq)
    print("Sample Token Sequence Shape:", sample_seq.shape)
    print("Reconstructed Text:", sample_recon)
