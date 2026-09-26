import pytest
import numpy as np
from PIL import Image
from src.model import RadiologyReportGenerator, BahdanauAttention

def test_model_build_and_generation():
    vocab_size = 50
    w2i = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3, "heart": 4, "normal": 5}
    i2w = {idx: word for word, idx in w2i.items()}
    
    generator = RadiologyReportGenerator(vocab_size=vocab_size, feature_dim=1024, max_seq_len=20)
    model = generator.build_model()
    
    assert model is not None
    assert model.output_shape == (None, 20, vocab_size)
    
    dummy_feat = np.random.randn(1024).astype(np.float32)
    greedy_text = generator.generate_report_greedy(dummy_feat, w2i, i2w)
    assert isinstance(greedy_text, str)
    
    beam_text = generator.generate_report_beam_search(dummy_feat, w2i, i2w, beam_width=2)
    assert isinstance(beam_text, str)

def test_attention_and_heatmap_overlay():
    vocab_size = 50
    w2i = {"<pad>": 0, "<unk>": 1, "<start>": 2, "<end>": 3, "heart": 4, "normal": 5}
    i2w = {idx: word for word, idx in w2i.items()}
    
    generator = RadiologyReportGenerator(vocab_size=vocab_size, feature_dim=1024, max_seq_len=20)
    generator.build_model()
    
    spatial_feat = np.random.randn(7, 7, 1024).astype(np.float32)
    text, attn_maps = generator.generate_report_with_attention(spatial_feat, w2i, i2w)
    
    assert isinstance(text, str)
    assert isinstance(attn_maps, list)
    
    if len(attn_maps) > 0:
        dummy_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
        overlay = generator.overlay_attention_heatmap(dummy_img, attn_maps[0])
        assert isinstance(overlay, Image.Image)
        assert overlay.size == (224, 224)
