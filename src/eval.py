"""Evaluation helpers and a simple shock predictor for the dummy dataset.

Provides a lightweight `ShockPredictor` that extracts per-stay features
from minute-level vitals and trains a logistic regression to predict
shock labels (6h early warning simulated in the dummy data).
"""
from __future__ import annotations

import json
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


class ShockPredictor:
    def __init__(self):
        self.model = LogisticRegression(solver="liblinear")
        self.feature_cols: List[str] = []

    def _extract_features_for_subject(self, df: pd.DataFrame) -> dict:
        # df contains minute-level rows for a single subject/stay
        values = {}
        n = len(df)
        t = np.arange(n)
        # basic statistics
        values["hr_mean"] = float(df["hr"].mean())
        values["sysbp_mean"] = float(df["sysbp"].mean())
        values["diasbp_mean"] = float(df["diasbp"].mean())
        values["rr_mean"] = float(df["resp_rate"].mean())
        values["spo2_mean"] = float(df["spo2"].mean())
        values["temp_mean"] = float(df["temp"].mean())
        # slopes (trend over the 24h)
        try:
            values["hr_slope"] = float(np.polyfit(t, df["hr"].values, 1)[0])
        except Exception:
            values["hr_slope"] = 0.0
        try:
            values["sysbp_slope"] = float(np.polyfit(t, df["sysbp"].values, 1)[0])
        except Exception:
            values["sysbp_slope"] = 0.0
        try:
            values["spo2_slope"] = float(np.polyfit(t, df["spo2"].values, 1)[0])
        except Exception:
            values["spo2_slope"] = 0.0
        # change from first 18h to last 6h (the simulated onset window)
        split = max(1, n - 360)
        first18 = df.iloc[:split]
        last6 = df.iloc[split:]
        values["hr_delta_last6_mean"] = float(last6["hr"].mean() - first18["hr"].mean())
        values["sysbp_delta_last6_mean"] = float(last6["sysbp"].mean() - first18["sysbp"].mean())
        values["spo2_delta_last6_mean"] = float(last6["spo2"].mean() - first18["spo2"].mean())
        return values

    def build_dataset(self, df: pd.DataFrame, subject_ids: List[int]):
        rows = []
        labels = []
        for sid in subject_ids:
            sub = df[df["subject_id"] == sid].sort_values("charttime")
            if sub.empty:
                continue
            feats = self._extract_features_for_subject(sub)
            rows.append(feats)
            # label_shock is constant per stay in the dummy generator
            labels.append(int(sub["label_shock"].iloc[0]))
        X = pd.DataFrame(rows)
        y = np.array(labels)
        return X, y

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.feature_cols = list(X.columns)
        self.model.fit(X.values, y)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X[self.feature_cols].values)[:, 1]

    def fit_from_csv(self, csv_path: str, subject_ids: List[int]):
        df = pd.read_csv(csv_path)
        X, y = self.build_dataset(df, subject_ids)
        self.fit(X, y)
        return X, y

    def load_splits(self, metadata_path: str):
        with open(metadata_path) as f:
            meta = json.load(f)
        return meta.get("splits", {})


class RAGASEvaluator:
    """Lightweight RAGAS-inspired evaluator.

    Computes:
    - Faithfulness: factual claims in answer supported by context
    - Relevance: answer relevance to the question
    - ContextRecall: fraction of ground truth facts covered
    """

    @staticmethod
    def faithfulness_score(answer: str, context: str) -> float:
        """Simple heuristic: % of answer tokens that appear in context."""
        if not answer or not context:
            return 0.0
        answer_tokens = set(answer.lower().split())
        context_tokens = set(context.lower().split())
        overlap = len(answer_tokens & context_tokens)
        if len(answer_tokens) == 0:
            return 1.0
        return float(overlap) / len(answer_tokens)

    @staticmethod
    def relevance_score(answer: str, question: str) -> float:
        """Simple heuristic: cosine-like overlap between answer and question."""
        if not answer or not question:
            return 0.0
        q_tokens = set(question.lower().split())
        a_tokens = set(answer.lower().split())
        overlap = len(a_tokens & q_tokens)
        union = len(a_tokens | q_tokens)
        if union == 0:
            return 1.0
        return float(overlap) / union

    @staticmethod
    def context_recall_score(answer: str, context: str) -> float:
        """Fraction of context facts mentioned in answer."""
        if not context:
            return 1.0
        context_tokens = set(context.lower().split())
        answer_tokens = set(answer.lower().split())
        overlap = len(answer_tokens & context_tokens)
        if len(context_tokens) == 0:
            return 1.0
        return float(overlap) / len(context_tokens)


class HallucinationDetector:
    """Detects unsupported claims using simple heuristics and optional NLI scoring."""

    @staticmethod
    def detect_hallucination(claim: str, context: str, nli_threshold: float = 0.5) -> bool:
        """Returns True if claim is likely hallucinated (not supported by context).

        Simple heuristic: check lexical overlap. A claim is hallucinated if it contains
        specific numbers or entities not found in context.
        """
        if not claim or not context:
            return True
        claim_l = claim.lower()
        context_l = context.lower()

        # check for numbers in claim
        import re

        nums = re.findall(r"\d+", claim_l)
        for num in nums:
            if num not in context_l:
                # number in claim not in context -> likely hallucinated
                return True

        # check for key clinical terms
        clinical_terms = [
            "sepsis",
            "shock",
            "pneumonia",
            "fever",
            "hypotension",
            "hypoxia",
            "tachycardia",
        ]
        for term in clinical_terms:
            if term in claim_l and term not in context_l:
                return True
        return False

    @staticmethod
    def batch_hallucination_rate(answers: List[str], contexts: List[str]) -> float:
        """Compute % hallucinations across a batch of answer-context pairs."""
        if not answers:
            return 0.0
        halluc = sum(
            1
            for ans, ctx in zip(answers, contexts)
            if HallucinationDetector.detect_hallucination(ans, ctx)
        )
        return float(halluc) / len(answers)


class FullEvaluator:
    """End-to-end evaluator combining RAGAS, hallucination, AUROC, and clinician score."""

    def __init__(self):
        self.ragas = RAGASEvaluator()
        self.halluc = HallucinationDetector()

    def evaluate_on_test_set(
        self,
        csv_path: str,
        metadata_path: str,
        agent_fn=None,
        predictor: ShockPredictor = None,
    ) -> dict:
        """Run full evaluation on test split.

        Args:
            csv_path: path to mimic3_dummy.csv
            metadata_path: path to metadata.json
            agent_fn: optional callable (subject_row) -> agent output dict
            predictor: optional ShockPredictor for AUROC

        Returns:
            dict with metrics: ragas_faithful, ragas_relevant, ragas_context_recall,
                              halluc_rate, auroc, clinician_score, counts
        """
        df = pd.read_csv(csv_path)
        with open(metadata_path) as f:
            meta = json.load(f)
        test_ids = meta.get("splits", {}).get("test", [])
        if not test_ids:
            return {}

        # prepare test data
        test_df = df[df["subject_id"].isin(test_ids)].copy()
        # per-subject aggregation for metrics
        subject_groups = test_df.groupby("subject_id")

        ragas_faithfulness = []
        ragas_relevance = []
        ragas_context_recall = []
        halluc_detected = []
        agent_probs = []
        true_labels = []

        for sid, group in subject_groups:
            group = group.sort_values("charttime")
            true_label = int(group["label_shock"].iloc[0])
            true_labels.append(true_label)

            # build evidence from notes and vitals
            notes = group[group["note_text"].notna()]["note_text"].tolist()
            context = " ".join(notes) if notes else ""
            vitals_str = (
                f"HR range: {group['hr'].min()}-{group['hr'].max()}, "
                f"SBP range: {group['sysbp'].min()}-{group['sysbp'].max()}, "
                f"SpO2 range: {group['spo2'].min()}-{group['spo2'].max()}"
            )

            # agent reasoning (if provided)
            agent_out = None
            agent_prob = None
            if agent_fn is not None:
                try:
                    agent_out = agent_fn(group, context)
                    agent_prob = agent_out.get("shock_prob", 0.5)
                    agent_probs.append(agent_prob)
                except Exception:
                    agent_probs.append(0.5)

            # RAGAS metrics
            if agent_out and "rationale" in agent_out:
                rationale = agent_out["rationale"]
                full_context = context + " " + vitals_str
                faithfulness = self.ragas.faithfulness_score(rationale, full_context)
                relevance = self.ragas.relevance_score(rationale, "Predict shock")
                ctx_recall = self.ragas.context_recall_score(rationale, full_context)

                ragas_faithfulness.append(faithfulness)
                ragas_relevance.append(relevance)
                ragas_context_recall.append(ctx_recall)

                # hallucination detection
                is_halluc = self.halluc.detect_hallucination(rationale, full_context)
                halluc_detected.append(is_halluc)

        # compute aggregates
        metrics = {"n_test_cases": len(test_ids)}

        if ragas_faithfulness:
            metrics["ragas_faithfulness"] = float(np.mean(ragas_faithfulness))
        if ragas_relevance:
            metrics["ragas_relevance"] = float(np.mean(ragas_relevance))
        if ragas_context_recall:
            metrics["ragas_context_recall"] = float(np.mean(ragas_context_recall))
        if halluc_detected:
            metrics["hallucination_rate"] = float(np.mean(halluc_detected))

        # AUROC if we have predictions
        if agent_probs and true_labels:
            from sklearn.metrics import roc_auc_score

            try:
                auroc = roc_auc_score(true_labels, agent_probs)
                metrics["auroc"] = float(auroc)
            except Exception:
                metrics["auroc"] = 0.5

        # predictor AUROC if provided
        if predictor is not None:
            try:
                X_test, y_test = predictor.build_dataset(df, test_ids)
                probs = predictor.predict_proba(X_test)
                from sklearn.metrics import roc_auc_score

                auroc_pred = roc_auc_score(y_test, probs)
                metrics["predictor_auroc"] = float(auroc_pred)
            except Exception:
                pass

        # clinician score: heuristic average of RAGAS scores
        component_scores = [
            metrics.get("ragas_faithfulness", 0.5),
            metrics.get("ragas_relevance", 0.5),
            metrics.get("ragas_context_recall", 0.5),
            1.0 - metrics.get("hallucination_rate", 0.0),  # invert so higher is better
        ]
        metrics["clinician_score"] = float(np.mean(component_scores))

        return metrics

    @staticmethod
    def metrics_to_latex_table(metrics: dict) -> str:
        """Format metrics dict as a LaTeX table row (Table 1)."""
        lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\caption{Evaluation Metrics on Test Set (100 dummy ICU patients).}",
            "\\label{tab:metrics}",
            "\\begin{tabular}{|l|r|}",
            "\\hline",
            "\\textbf{Metric} & \\textbf{Score} \\\\",
            "\\hline",
        ]

        # add metric rows
        metric_names = [
            ("n_test_cases", "# Test Cases"),
            ("ragas_faithfulness", "RAGAS Faithfulness"),
            ("ragas_relevance", "RAGAS Relevance"),
            ("ragas_context_recall", "RAGAS Context Recall"),
            ("hallucination_rate", "Hallucination Rate"),
            ("auroc", "AUROC (Agent)"),
            ("predictor_auroc", "AUROC (Predictor)"),
            ("clinician_score", "Clinician Score"),
        ]
        for key, label in metric_names:
            if key in metrics:
                val = metrics[key]
                if isinstance(val, float):
                    lines.append(f"{label} & {val:.3f} \\\\")
                else:
                    lines.append(f"{label} & {val} \\\\")

        lines.extend(
            [
                "\\hline",
                "\\end{tabular}",
                "\\end{table}",
            ]
        )
        return "\n".join(lines)
