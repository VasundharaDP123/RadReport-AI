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

    def evaluate_clinical_pathology(self, references, hypotheses):
        """
        Calculates Clinical Pathology Entity Extraction F1, Precision, and Recall scores.
        Evaluates agreement on 8 core radiological conditions:
        ['cardiomegaly', 'effusion', 'atelectasis', 'pneumonia', 'opacity', 'pneumothorax', 'edema', 'normal']
        """
        pathologies = ['cardiomegaly', 'effusion', 'atelectasis', 'pneumonia', 'opacity', 'pneumothorax', 'edema', 'normal']
        
        precisions, recalls, f1s = [], [], []

        for ref, hyp in zip(references, hypotheses):
            ref_lower = ref.lower()
            hyp_lower = hyp.lower()

            ref_set = set(p for p in pathologies if p in ref_lower)
            hyp_set = set(p for p in pathologies if p in hyp_lower)

            if 'no acute' in ref_lower or 'clear' in ref_lower or 'normal' in ref_lower:
                ref_set.add('normal')
            if 'no acute' in hyp_lower or 'clear' in hyp_lower or 'normal' in hyp_lower:
                hyp_set.add('normal')

            tp = len(ref_set.intersection(hyp_set))
            fp = len(hyp_set - ref_set)
            fn = len(ref_set - hyp_set)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 1.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

            precisions.append(prec)
            recalls.append(rec)
            f1s.append(f1)

        return {
            'Clinical-Precision': float(np.mean(precisions)),
            'Clinical-Recall': float(np.mean(recalls)),
            'Clinical-F1': float(np.mean(f1s))
        }

    def evaluate_corpus(self, references, hypotheses):
        """
        Evaluates an entire dataset corpus of reference and hypothesis texts.
        Returns average BLEU-1..4, ROUGE-L, and Clinical F1 scores.
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

        nlp_scores = {
            'BLEU-1': float(np.mean(b1_list)),
            'BLEU-2': float(np.mean(b2_list)),
            'BLEU-3': float(np.mean(b3_list)),
            'BLEU-4': float(np.mean(b4_list)),
            'ROUGE-L': float(np.mean(rouge_list))
        }

        clinical_scores = self.evaluate_clinical_pathology(references, hypotheses)
        nlp_scores.update(clinical_scores)
        return nlp_scores

    def compute_most_common_baseline(self, train_references, test_references):
        """
        Computes performance of the Most-Common-Report Baseline.
        Finds the single most frequent report in training set and predicts it for all test samples.
        Distinguishes a serious academic project from trivial overfitting.
        """
        if not train_references:
            most_common = "the heart size and pulmonary vascularity appear within normal limits . no focal consolidation or pleural effusion ."
        else:
            counts = pd.Series(train_references).value_counts()
            most_common = counts.index[0]

        print(f"Most-Common-Report Baseline Text: '{most_common}'")
        baseline_hypotheses = [most_common] * len(test_references)
        
        return self.evaluate_corpus(test_references, baseline_hypotheses)

    @staticmethod
    def generate_benchmark_table(results_dict):
        """
        Generates a formatted GitHub Markdown table comparing multiple experiment runs.
        """
        headers = ["Model / Configuration", "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4", "ROUGE-L", "Clinical F1"]
        lines = [
            "| " + " | ".join(headers) + " |",
            "|" + "|".join(["---"] * len(headers)) + "|"
        ]
        
        for name, metrics in results_dict.items():
            row = [
                f"**{name}**",
                f"{metrics.get('BLEU-1', 0):.3f}",
                f"{metrics.get('BLEU-2', 0):.3f}",
                f"{metrics.get('BLEU-3', 0):.3f}",
                f"{metrics.get('BLEU-4', 0):.3f}",
                f"{metrics.get('ROUGE-L', 0):.3f}",
                f"{metrics.get('Clinical-F1', 0):.3f}"
            ]
            lines.append("| " + " | ".join(row) + " |")

        return "\n".join(lines)


if __name__ == "__main__":
    evaluator = RadiologyEvaluator()
    refs = ["the heart is normal in size . no focal opacity ."]
    hyps = ["the heart size is normal . no consolidation ."]
    scores = evaluator.evaluate_corpus(refs, hyps)
    print("Sample Evaluation Scores:", scores)

