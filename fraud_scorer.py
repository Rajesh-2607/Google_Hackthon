"""
Fraud Risk Scoring Engine
=========================
Loads the trained XGBoost model and provides:
  - Per-transaction fraud risk score (0–100%)
  - Risk-level categorisation (Low / Medium / High)
  - SHAP-based feature-level explanations for each prediction

Usage:
    from fraud_scorer import FraudScorer
    scorer = FraudScorer()
    result = scorer.score(transaction_array)   # single or batch
"""

import numpy as np
import pandas as pd
import json, os
from pathlib import Path
from typing import Dict, List, Optional, Union

from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from gemini_explainer import (
    GeminiExplainer, fallback_explanation, GEMINI_AVAILABLE,
)


# ── Risk-level thresholds (configurable) ────────────────────────────────────

DEFAULT_THRESHOLDS = {
    "high":   0.70,   # ≥ 70 % → Block
    "medium": 0.30,   # ≥ 30 % → Manual Review
    # < 30 %           → Allow
}

RISK_ACTIONS = {
    "HIGH":   "BLOCK — immediate investigation required",
    "MEDIUM": "REVIEW — flag for manual analyst review",
    "LOW":    "ALLOW — transaction appears legitimate",
}


class FraudScorer:
    """Production-ready fraud risk scoring engine."""

    def __init__(
        self,
        model_path: str = "results/xgboost_fraud_model.json",
        results_path: str = "results/results.json",
        feature_names_path: str = "processed_data/feature_names.csv",
        thresholds: Optional[Dict[str, float]] = None,
    ):
        self.model = XGBClassifier()
        self.model.load_model(model_path)

        # Load optimal threshold from training results
        try:
            with open(results_path) as f:
                res = json.load(f)
            self.optimal_threshold = res.get("xgb_tuned", {}).get("threshold", 0.5)
        except Exception:
            self.optimal_threshold = 0.5

        # Feature names
        try:
            self.feature_names = pd.read_csv(feature_names_path)["feature_name"].tolist()
        except Exception:
            self.feature_names = None

        # Risk-level thresholds
        self.thresholds = thresholds or DEFAULT_THRESHOLDS

        # SHAP explainer (lazy init)
        self._explainer = None

    # ── helpers ──────────────────────────────────────────────────────────

    def _ensure_2d(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return X

    def _risk_level(self, prob: float) -> str:
        if prob >= self.thresholds["high"]:
            return "HIGH"
        elif prob >= self.thresholds["medium"]:
            return "MEDIUM"
        return "LOW"

    @property
    def explainer(self):
        if self._explainer is None:
            if not SHAP_AVAILABLE:
                raise ImportError("shap is required for explanations. pip install shap")
            self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    # ── core scoring ─────────────────────────────────────────────────────

    def score(self, X) -> List[Dict]:
        """
        Score one or more transactions.

        Parameters
        ----------
        X : array-like, shape (n_features,) or (n_samples, n_features)

        Returns
        -------
        List of dicts with keys:
            risk_score, risk_pct, risk_level, action, binary_prediction
        """
        X = self._ensure_2d(X)
        probs = self.model.predict_proba(X)[:, 1]

        results = []
        for i, prob in enumerate(probs):
            level = self._risk_level(prob)
            results.append({
                "transaction_index": i,
                "risk_score":        round(float(prob), 6),
                "risk_pct":          f"{prob * 100:.2f}%",
                "risk_level":        level,
                "action":            RISK_ACTIONS[level],
                "binary_prediction": int(prob >= self.optimal_threshold),
            })
        return results

    # ── SHAP explanations ────────────────────────────────────────────────

    def explain(self, X, top_k: int = 5) -> List[Dict]:
        """
        Return per-transaction SHAP explanations.

        Parameters
        ----------
        X : array-like
        top_k : int – number of top contributing features to return

        Returns
        -------
        List of dicts, one per transaction, each containing:
            risk_score, risk_level, top_features (list of
            {feature, shap_value, direction})
        """
        X = self._ensure_2d(X)
        probs = self.model.predict_proba(X)[:, 1]
        shap_vals = self.explainer.shap_values(X)

        names = self.feature_names or [f"Feature_{i}" for i in range(X.shape[1])]

        explanations = []
        for idx in range(len(X)):
            sv = shap_vals[idx]
            top_idx = np.argsort(np.abs(sv))[::-1][:top_k]

            features = []
            for fi in top_idx:
                features.append({
                    "feature":    names[fi],
                    "shap_value": round(float(sv[fi]), 6),
                    "direction":  "increases fraud risk" if sv[fi] > 0 else "decreases fraud risk",
                    "feature_value": round(float(X[idx, fi]), 6),
                })

            level = self._risk_level(probs[idx])
            explanations.append({
                "transaction_index": idx,
                "risk_score":        round(float(probs[idx]), 6),
                "risk_pct":          f"{probs[idx] * 100:.2f}%",
                "risk_level":        level,
                "action":            RISK_ACTIONS[level],
                "top_features":      features,
                "narrative":         self._narrative(probs[idx], level, features),
            })
        return explanations

    @staticmethod
    def _narrative(prob, level, features) -> str:
        """Generate a human-readable explanation string."""
        lines = [f"Fraud risk is {level} ({prob*100:.1f}%) because:"]
        for f in features:
            arrow = "↑" if f["shap_value"] > 0 else "↓"
            lines.append(f"  {arrow} {f['feature']} = {f['feature_value']:.4f} "
                         f"({f['direction']})")
        return "\n".join(lines)

    # ── Gemini business explanations ──────────────────────────────────────

    def business_explain(
        self, X, top_k: int = 5, gemini_api_key: Optional[str] = None,
    ) -> List[Dict]:
        """
        Full pipeline: Score → SHAP → Gemini business narrative.

        If a Gemini API key is available, generates LLM-powered explanations.
        Otherwise falls back to rule-based business language.

        Returns the same dicts as explain(), with an extra key:
            'business_explanation' — plain-English narrative from Gemini/fallback
        """
        explanations = self.explain(X, top_k=top_k)

        # Try Gemini first
        use_gemini = False
        if gemini_api_key or os.environ.get("GEMINI_API_KEY"):
            try:
                gem = GeminiExplainer(api_key=gemini_api_key)
                use_gemini = True
            except Exception:
                use_gemini = False

        for ex in explanations:
            if use_gemini:
                try:
                    ex["business_explanation"] = gem.explain(ex)
                except Exception as e:
                    ex["business_explanation"] = fallback_explanation(ex)
            else:
                ex["business_explanation"] = fallback_explanation(ex)

        return explanations

    # ── batch scoring with DataFrame output ──────────────────────────────

    def score_dataframe(self, X) -> pd.DataFrame:
        """Return a tidy DataFrame of risk scores for all transactions."""
        results = self.score(X)
        return pd.DataFrame(results)

    def explain_dataframe(self, X, top_k: int = 5) -> pd.DataFrame:
        """Return explanations as a flat DataFrame (one row per feature contribution)."""
        explanations = self.explain(X, top_k=top_k)
        rows = []
        for ex in explanations:
            for feat in ex["top_features"]:
                rows.append({
                    "transaction_index": ex["transaction_index"],
                    "risk_score":        ex["risk_score"],
                    "risk_level":        ex["risk_level"],
                    **feat,
                })
        return pd.DataFrame(rows)


# ── CLI quick-test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Fraud Risk Scoring Engine — Quick Test")
    print("=" * 60)

    scorer = FraudScorer()

    # Load test data
    X_test = np.load("processed_data/X_test.npy")
    y_test = np.load("processed_data/y_test.npy")

    # Score first 10 transactions
    sample = X_test[:10]
    results = scorer.score(sample)

    print("\n── Risk Scores ──")
    for r in results:
        tag = "🔴" if r["risk_level"] == "HIGH" else ("🟡" if r["risk_level"] == "MEDIUM" else "🟢")
        print(f"  {tag} Txn {r['transaction_index']:>3}  "
              f"Risk: {r['risk_pct']:>7}  Level: {r['risk_level']:<6}  "
              f"Action: {r['action']}")

    # Explain first 3
    if SHAP_AVAILABLE:
        print("\n── SHAP Explanations ──")
        explanations = scorer.explain(sample[:3], top_k=5)
        for ex in explanations:
            print(f"\n{ex['narrative']}")

        # Business explanations (Gemini or fallback)
        print("\n── Business Explanations (Gemini / Fallback) ──")
        biz_explanations = scorer.business_explain(sample[:3], top_k=3)
        for ex in biz_explanations:
            print(f"\n  Txn {ex['transaction_index']} [{ex['risk_level']}]:")
            print(f"  {ex['business_explanation']}")

    # Summary stats on full test set
    print("\n── Full Test Set Summary ──")
    df = scorer.score_dataframe(X_test)
    print(f"  Total transactions: {len(df):,}")
    print(f"  Risk distribution:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        cnt = (df["risk_level"] == level).sum()
        print(f"    {level:>6}: {cnt:>6,}  ({cnt/len(df)*100:.1f}%)")

    # Accuracy at optimal threshold
    correct = (df["binary_prediction"].values == y_test).sum()
    print(f"\n  Accuracy at threshold {scorer.optimal_threshold}: "
          f"{correct/len(y_test)*100:.2f}%")
