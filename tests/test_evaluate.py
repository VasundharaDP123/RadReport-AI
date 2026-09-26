import pytest
from src.evaluate import RadiologyEvaluator

def test_nlp_metrics():
    evaluator = RadiologyEvaluator()
    ref = "the heart size is normal . clear lungs ."
    hyp = "the heart size is normal . clear lungs ."
    
    b1, b2, b3, b4 = evaluator.calculate_bleu(ref, hyp)
    r_l = evaluator.calculate_rouge_l(ref, hyp)
    
    assert b1 > 0.9
    assert b4 > 0.9
    assert r_l > 0.9

def test_clinical_pathology_metrics():
    evaluator = RadiologyEvaluator()
    refs = ["cardiomegaly is noted with focal opacity in right lower lobe ."]
    hyps = ["cardiomegaly present . focal opacity in lung base ."]
    
    scores = evaluator.evaluate_clinical_pathology(refs, hyps)
    assert "Clinical-Precision" in scores
    assert "Clinical-Recall" in scores
    assert "Clinical-F1" in scores
    assert scores["Clinical-Precision"] > 0.0

def test_benchmark_table_generator():
    evaluator = RadiologyEvaluator()
    data = {
        "Baseline": {"BLEU-1": 0.36, "BLEU-4": 0.08, "ROUGE-L": 0.29, "Clinical-F1": 0.40},
        "DenseNet121 + Beam": {"BLEU-1": 0.43, "BLEU-4": 0.16, "ROUGE-L": 0.36, "Clinical-F1": 0.68}
    }
    tbl = evaluator.generate_benchmark_table(data)
    assert "| **Baseline** |" in tbl
    assert "| **DenseNet121 + Beam** |" in tbl
