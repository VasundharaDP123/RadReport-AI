import pytest
import numpy as np
from PIL import Image
from src.feature_extractor import ImageFeatureExtractor

def test_feature_extraction_shapes():
    extractor = ImageFeatureExtractor(architecture='densenet121')
    dummy_img = Image.new('RGB', (224, 224), color=(100, 100, 100))
    img_arr = np.array(dummy_img)
    
    feat_1d = extractor.extract_single_image(img_arr)
    assert isinstance(feat_1d, np.ndarray)
    assert feat_1d.shape == (1024,)
    
    spatial_3d = extractor.extract_spatial_image(img_arr)
    assert isinstance(spatial_3d, np.ndarray)
    assert spatial_3d.shape == (7, 7, 1024)

def test_missing_image_fallback():
    extractor = ImageFeatureExtractor(architecture='vgg16')
    feat = extractor.extract_single_image("non_existent_xray.png")
    
    assert isinstance(feat, np.ndarray)
    assert feat.shape == (512,)
