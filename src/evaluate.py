import numpy as np
import pandas as pd
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

try:
    from rouge_score import rouge_scorer
    HAS_ROUGE_SCORE = True
except ImportError:
    HAS_ROUGE_SCORE = False

class RadiologyEvaluator:
    """
    Evaluates quantitative performance of generated radiology reports
    against ground truth reference reports using BLEU-1..4 and ROUGE-L metrics.
    Includes comparison against the Most-Common-Report Baseline.
    """
    def __init__(self):
        self.smoother = SmoothingFunction().method1
        if HAS_ROUGE_SCORE:
            self.rouge_evaluator = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        else:
            self.rouge_evaluator = None

    def _lcs_length(self, seq1, seq2):
        """Pure Python fallback for Longest Common Subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def calculate_bleu(self, reference, hypothesis):
        """
        Calculates BLEU-1, BLEU-2, BLEU-3, BLEU-4 for a single reference-hypothesis pair.
        """
        ref_tokens = [reference.strip().split()]
        hyp_tokens = hypothesis.strip().split()

        if not hyp_tokens:
            return 0.0, 0.0, 0.0, 0.0

        b1 = sentence_bleu(ref_tokens, hyp_tokens, weights=(1.0, 0, 0, 0), smoothing_function=self.smoother)
        b2 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=self.smoother)
        b3 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=self.smoother)
        b4 = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=self.smoother)

        return b1, b2, b3, b4

    def calculate_rouge_l(self, reference, hypothesis):
        """
        Calculates ROUGE-L F1 score.
        """
        if HAS_ROUGE_SCORE and self.rouge_evaluator:
            scores = self.rouge_evaluator.score(reference, hypothesis)
            return scores['rougeL'].fmeasure
        
        # Fallback LCS ROUGE-L calculation
        ref_tokens = reference.strip().split()
        hyp_tokens = hypothesis.strip().split()
        if not ref_tokens or not hyp_tokens:
            return 0.0
            
        lcs = self._lcs_length(ref_tokens, hyp_tokens)
        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)
        if precision + recall == 0:
            return 0.0
        return (2 * precision * recall) / (precision + recall)

    def evaluate_corpus(self, references, hypotheses):
        """
        Evaluates an entire dataset corpus of reference and hypothesis texts.
        Returns average BLEU-1..4 and ROUGE-L scores.
        """
        b1_list, b2_list, b3_list, b4_list = [], [], [], []
        rouge_list = []

        for ref, hyp in zip(references, hypotheses):
            b1, b2, b3, b4 = self.calculate_bleu(ref, hyp)
            r_l = self.calculate_rouge_l(ref, hyp)

            b1_list.append(b1)
            b2_list.append(b2)
            b3_list.append(b3)
            b4_list.append(b4)
            rouge_list.append(r_l)

        return {
            'BLEU-1': float(np.mean(b1_list)),
            'BLEU-2': float(np.mean(b2_list)),
            'BLEU-3': float(np.mean(b3_list)),
            'BLEU-4': float(np.mean(b4_list)),
            'ROUGE-L': float(np.mean(rouge_list))
        }

    def compute_most_common_baseline(self, train_references, test_references):
        """
        Computes performance of the Most-Common-Report Baseline.
        Finds the single most frequent report in training set and predicts it for all test samples.
        Distinguishes a serious academic project from trivial overfitting.
        """
        if not train_references:
            most_common = "the heart size and pulmonary vascularity appear within normal limits . no focal consolidation or pleural effusion ."
        else:
            # Find most common report
            counts = pd.Series(train_references).value_counts()
            most_common = counts.index[0]

        print(f"Most-Common-Report Baseline Text: '{most_common}'")
        baseline_hypotheses = [most_common] * len(test_references)
        
        return self.evaluate_corpus(test_references, baseline_hypotheses)


if __name__ == "__main__":
    evaluator = RadiologyEvaluator()
    refs = ["the heart is normal in size . no focal opacity ."]
    hyps = ["the heart size is normal . no consolidation ."]
    scores = evaluator.evaluate_corpus(refs, hyps)
    print("Sample Evaluation Scores:", scores)
