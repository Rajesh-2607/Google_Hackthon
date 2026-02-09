"""
Model & Data Drift Monitor
===========================
Detects distribution shifts between training and live/test data that
signal model degradation.

Techniques:
  - Population Stability Index (PSI) per feature
  - Kolmogorov–Smirnov test per feature
  - Prediction distribution drift
  - Performance decay tracking

Usage:
    from drift_monitor import DriftMonitor
    monitor = DriftMonitor()
    report  = monitor.run(X_new, y_new_optional)
"""

import numpy as np
import pandas as pd
import json, os
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── PSI helper ───────────────────────────────────────────────────────────────

def _psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index between two 1-D distributions.
    PSI < 0.10  → no significant change
    PSI 0.10–0.25 → moderate shift (monitor)
    PSI > 0.25 → significant shift (retrain)
    """
    eps = 1e-6
    breakpoints = np.linspace(
        min(reference.min(), current.min()) - eps,
        max(reference.max(), current.max()) + eps,
        bins + 1,
    )
    ref_pct = np.histogram(reference, bins=breakpoints)[0] / len(reference) + eps
    cur_pct = np.histogram(current, bins=breakpoints)[0] / len(current) + eps
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


# ── Drift severity labels ───────────────────────────────────────────────────

def _psi_label(psi_val: float) -> str:
    if psi_val < 0.10:
        return "STABLE"
    elif psi_val < 0.25:
        return "MODERATE_DRIFT"
    return "SIGNIFICANT_DRIFT"


def _ks_label(p_value: float, alpha: float = 0.01) -> str:
    return "DRIFT_DETECTED" if p_value < alpha else "STABLE"


# =============================================================================

class DriftMonitor:
    """
    Monitors feature drift (PSI + KS-test) and prediction drift.
    """

    def __init__(
        self,
        reference_path: str = "processed_data/X_train.npy",
        feature_names_path: str = "processed_data/feature_names.csv",
        results_dir: str = "results",
        psi_bins: int = 10,
        ks_alpha: float = 0.01,
    ):
        self.reference = np.load(reference_path)
        self.results_dir = results_dir
        self.psi_bins = psi_bins
        self.ks_alpha = ks_alpha
        os.makedirs(results_dir, exist_ok=True)

        try:
            self.feature_names = pd.read_csv(feature_names_path)["feature_name"].tolist()
        except Exception:
            self.feature_names = [f"Feature_{i}" for i in range(self.reference.shape[1])]

    # ── Feature drift ────────────────────────────────────────────────────

    def feature_drift(self, X_current: np.ndarray) -> pd.DataFrame:
        """
        Compute PSI and KS-test for every feature.

        Returns DataFrame with columns:
            feature, psi, psi_status, ks_stat, ks_pvalue, ks_status
        """
        rows = []
        for i in range(self.reference.shape[1]):
            ref_col = self.reference[:, i]
            cur_col = X_current[:, i]

            psi_val = _psi(ref_col, cur_col, bins=self.psi_bins)
            ks_stat, ks_p = stats.ks_2samp(ref_col, cur_col)

            rows.append({
                "feature":    self.feature_names[i],
                "psi":        round(psi_val, 6),
                "psi_status": _psi_label(psi_val),
                "ks_stat":    round(float(ks_stat), 6),
                "ks_pvalue":  round(float(ks_p), 8),
                "ks_status":  _ks_label(ks_p, self.ks_alpha),
            })
        return pd.DataFrame(rows)

    # ── Prediction drift ─────────────────────────────────────────────────

    def prediction_drift(
        self,
        ref_probs: np.ndarray,
        cur_probs: np.ndarray,
    ) -> Dict:
        """Compare prediction probability distributions."""
        psi_val = _psi(ref_probs, cur_probs, bins=self.psi_bins)
        ks_stat, ks_p = stats.ks_2samp(ref_probs, cur_probs)

        return {
            "prediction_psi":       round(psi_val, 6),
            "prediction_psi_status": _psi_label(psi_val),
            "prediction_ks_stat":    round(float(ks_stat), 6),
            "prediction_ks_pvalue":  round(float(ks_p), 8),
            "prediction_ks_status":  _ks_label(ks_p, self.ks_alpha),
            "ref_mean_prob":         round(float(ref_probs.mean()), 6),
            "cur_mean_prob":         round(float(cur_probs.mean()), 6),
            "ref_fraud_rate":        round(float((ref_probs >= 0.5).mean()), 6),
            "cur_fraud_rate":        round(float((cur_probs >= 0.5).mean()), 6),
        }

    # ── Performance decay ────────────────────────────────────────────────

    @staticmethod
    def performance_decay(
        baseline_metrics: Dict,
        current_metrics: Dict,
        decay_threshold: float = 0.05,
    ) -> Dict:
        """
        Compare current metrics against baseline and flag decay.
        """
        decay_report = {}
        for metric in ["roc_auc", "pr_auc", "f1", "recall", "precision"]:
            base_val = baseline_metrics.get(metric, 0)
            curr_val = current_metrics.get(metric, 0)
            drop = base_val - curr_val
            decay_report[metric] = {
                "baseline":  round(base_val, 6),
                "current":   round(curr_val, 6),
                "drop":      round(drop, 6),
                "status":    "DEGRADED" if drop > decay_threshold else "HEALTHY",
            }
        return decay_report

    # ── Full report ──────────────────────────────────────────────────────

    def run(
        self,
        X_current: np.ndarray,
        cur_probs: Optional[np.ndarray] = None,
        ref_probs: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Run the full drift analysis and return a structured report.
        """
        # 1. Feature drift
        feat_df = self.feature_drift(X_current)
        n_psi_drift = (feat_df["psi_status"] != "STABLE").sum()
        n_ks_drift  = (feat_df["ks_status"] != "STABLE").sum()

        report = {
            "feature_drift": {
                "summary": {
                    "total_features":     len(feat_df),
                    "psi_drifted":        int(n_psi_drift),
                    "ks_drifted":         int(n_ks_drift),
                    "overall_status":     "DRIFT_DETECTED" if n_psi_drift > 0 else "STABLE",
                },
                "details": feat_df.to_dict(orient="records"),
            },
        }

        # 2. Prediction drift (if probabilities supplied)
        if cur_probs is not None and ref_probs is not None:
            report["prediction_drift"] = self.prediction_drift(ref_probs, cur_probs)

        return report

    # ── Visualisations ───────────────────────────────────────────────────

    def plot_drift_report(
        self,
        feat_df: pd.DataFrame,
        save_path: Optional[str] = None,
    ):
        """Bar chart of PSI values per feature, colour-coded by severity."""
        colours = []
        for status in feat_df["psi_status"]:
            if status == "SIGNIFICANT_DRIFT":
                colours.append("#e74c3c")
            elif status == "MODERATE_DRIFT":
                colours.append("#f39c12")
            else:
                colours.append("#2ecc71")

        fig, ax = plt.subplots(figsize=(12, max(6, len(feat_df) * 0.35)))
        y_pos = range(len(feat_df))
        ax.barh(y_pos, feat_df["psi"], color=colours)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feat_df["feature"])
        ax.axvline(0.10, color="orange", ls="--", lw=1, label="Moderate (0.10)")
        ax.axvline(0.25, color="red",    ls="--", lw=1, label="Significant (0.25)")
        ax.set_xlabel("PSI")
        ax.set_title("Feature Drift — Population Stability Index")
        ax.legend()
        plt.tight_layout()

        path = save_path or os.path.join(self.results_dir, "drift_psi_chart.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path

    def plot_prediction_distribution(
        self,
        ref_probs: np.ndarray,
        cur_probs: np.ndarray,
        save_path: Optional[str] = None,
    ):
        """Overlay histograms of reference vs current prediction distributions."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(ref_probs, bins=50, alpha=0.5, density=True, label="Reference (train)", color="#3498db")
        ax.hist(cur_probs, bins=50, alpha=0.5, density=True, label="Current",           color="#e74c3c")
        ax.set_xlabel("Predicted Fraud Probability")
        ax.set_ylabel("Density")
        ax.set_title("Prediction Distribution Drift")
        ax.legend()
        plt.tight_layout()

        path = save_path or os.path.join(self.results_dir, "prediction_drift.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return path


# ── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from xgboost import XGBClassifier

    print("=" * 60)
    print("  Drift Monitor — Train vs Test Analysis")
    print("=" * 60)

    monitor = DriftMonitor()

    X_test = np.load("processed_data/X_test.npy")
    X_val  = np.load("processed_data/X_val.npy")

    # Feature drift: train vs test
    feat_df = monitor.feature_drift(X_test)
    print("\n── Feature Drift (PSI) ──")
    print(feat_df[["feature", "psi", "psi_status"]].to_string(index=False))

    drifted = feat_df[feat_df["psi_status"] != "STABLE"]
    if len(drifted):
        print(f"\n  ⚠ {len(drifted)} feature(s) show drift:")
        for _, row in drifted.iterrows():
            print(f"    {row['feature']}: PSI={row['psi']:.4f} ({row['psi_status']})")
    else:
        print("\n  ✓ All features are stable — no drift detected.")

    # Prediction drift
    model = XGBClassifier()
    model.load_model("results/xgboost_fraud_model.json")
    ref_probs = model.predict_proba(X_val)[:, 1]
    cur_probs = model.predict_proba(X_test)[:, 1]

    pred_drift = monitor.prediction_drift(ref_probs, cur_probs)
    print("\n── Prediction Drift ──")
    for k, v in pred_drift.items():
        print(f"  {k}: {v}")

    # Save plots
    p1 = monitor.plot_drift_report(feat_df)
    p2 = monitor.plot_prediction_distribution(ref_probs, cur_probs)
    print(f"\n  ✓ Drift PSI chart      → {p1}")
    print(f"  ✓ Prediction drift plot → {p2}")

    # Save full report
    report = monitor.run(X_test, cur_probs=cur_probs, ref_probs=ref_probs)
    with open(os.path.join("results", "drift_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"  ✓ Drift report → results/drift_report.json")
