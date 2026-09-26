import pytest
import numpy as np
import pandas as pd
from src.data_preprocessing import RadiologyDataPreprocessor

def test_clean_text():
    preprocessor = RadiologyDataPreprocessor()
    raw = "The heart size is normal XXXX . No pleural effusion xxxx ."
    cleaned = preprocessor.clean_text(raw)
    
    assert "<start>" in cleaned
    assert "<end>" in cleaned
    assert "xxxx" not in cleaned.lower()
    assert "heart" in cleaned

def test_patient_wise_split():
    preprocessor = RadiologyDataPreprocessor()
    df = preprocessor._generate_synthetic_dataset(num_samples=100)
    train_df, val_df, test_df = preprocessor.patient_wise_split(df, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    
    # Verify no patient UID overlap across splits (Zero Data Leakage)
    train_uids = set(train_df['uid'])
    val_uids = set(val_df['uid'])
    test_uids = set(test_df['uid'])
    
    assert len(train_uids.intersection(val_uids)) == 0
    assert len(train_uids.intersection(test_uids)) == 0
    assert len(val_uids.intersection(test_uids)) == 0

def test_vocabulary_and_sequences():
    preprocessor = RadiologyDataPreprocessor(min_word_count=1)
    texts = [
        "<start> normal heart size <end>",
        "<start> focal opacity pneumonia <end>"
    ]
    w2i, i2w = preprocessor.build_vocabulary(texts)
    
    assert preprocessor.vocab_size > 4
    assert "<start>" in w2i
    assert "<end>" in w2i
    
    seq = preprocessor.text_to_sequence("<start> normal heart <end>")
    assert isinstance(seq, np.ndarray)
    assert len(seq) == preprocessor.max_seq_len
    
    recon = preprocessor.sequence_to_text(seq)
    assert "normal" in recon or "heart" in recon
