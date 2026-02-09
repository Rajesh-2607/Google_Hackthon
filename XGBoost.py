"""
XGBoost Credit Card Fraud Detection — Improved Pipeline
========================================================
Improvements over v1:
  1. RandomizedSearchCV hyperparameter tuning (lightweight, no optuna needed)
  2. Threshold tuning on validation set (maximise F1 or recall)
  3. Comprehensive evaluation: confusion matrix, classification report, PR-AUC
  4. ROC + Precision-Recall curve plots
  5. Feature importance bar chart
  6. SHAP explainability summary plot
  7. Model + results saved to results/ folder
"""

import pandas as pd
import numpy as np
import os, json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, make_scorer
)
from scipy.stats import uniform, randint
from preprocessed import load_processed_data
from xgboost import XGBClassifier

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("Info: shap not installed — skipping explainability plots.")
    print("      Install with: pip install shap")

# ── Helpers ──────────────────────────────────────────────────────────────────

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate(y_true, y_prob, threshold=0.5, label="Model"):
    """Compute & print full metrics for a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    m = {
        "threshold":  round(threshold, 4),
        "precision":  round(precision_score(y_true, y_pred, zero_division=0), 6),
        "recall":     round(recall_score(y_true, y_pred, zero_division=0), 6),
        "f1":         round(f1_score(y_true, y_pred, zero_division=0), 6),
        "roc_auc":    round(roc_auc_score(y_true, y_prob), 6),
        "pr_auc":     round(average_precision_score(y_true, y_prob), 6),
    }
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'─'*55}")
    print(f"  {label}  (threshold = {threshold:.3f})")
    print(f"{'─'*55}")
    for k, v in m.items():
        print(f"  {k:<12}: {v}")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(classification_report(y_true, y_pred, target_names=["Legit", "Fraud"]))
    return m


def best_threshold(y_true, y_prob, goal="f1"):
    """Sweep thresholds 0.01–0.99 (fine 0.005 steps) on val set."""
    best_s, best_t = 0, 0.5
    for t in np.arange(0.01, 0.995, 0.005):
        yp = (y_prob >= t).astype(int)
        if goal == "f1":
            s = f1_score(y_true, yp, zero_division=0)
        else:  # high-recall: maximise recall while keeping precision ≥ 0.50
            p = precision_score(y_true, yp, zero_division=0)
            s = recall_score(y_true, yp, zero_division=0) if p >= 0.50 else 0
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s


def save_curves(y_true, y_prob, label="XGBoost"):
    """Save ROC + PR curves as PNG."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    axes[0].plot(fpr, tpr, lw=2,
                 label=f"{label} (AUC={roc_auc_score(y_true, y_prob):.4f})")
    axes[0].plot([0, 1], [0, 1], "k--", lw=1)
    axes[0].set(title="ROC Curve", xlabel="FPR", ylabel="TPR")
    axes[0].legend()

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    axes[1].plot(rec, prec, lw=2,
                 label=f"{label} (AP={average_precision_score(y_true, y_prob):.4f})")
    axes[1].set(title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision")
    axes[1].legend()

    plt.tight_layout()
    p = os.path.join(RESULTS_DIR, "roc_pr_curves.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"  ✓ Curves saved → {p}")


# =============================================================================
# LOAD DATA
# =============================================================================

X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()


# =============================================================================
# 1. BASELINE — Logistic Regression
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 1: Baseline — Logistic Regression")
print("=" * 60)

baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
baseline.fit(X_train, y_train)

y_base_prob = baseline.predict_proba(X_test)[:, 1]
evaluate(y_test, y_base_prob, threshold=0.5, label="Baseline LogReg")


# =============================================================================
# 2. HYPERPARAMETER TUNING (RandomizedSearchCV — fast 10% subsample)
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 2: XGBoost Hyperparameter Tuning (RandomizedSearchCV)")
print("=" * 60)

# Note: SMOTE already resamples to ~0.5 ratio, so we do NOT set scale_pos_weight
# (double class-weighting hurts precision/recall balance)

# Use a 10% stratified subsample for faster tuning on large SMOTE data
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.9, random_state=42)
tune_idx, _ = next(sss.split(X_train, y_train))
X_tune, y_tune = X_train[tune_idx], y_train[tune_idx]
print(f"  Tuning on {len(X_tune):,} samples (10% subsample for speed)")

param_distributions = {
    "n_estimators":     randint(300, 1500),
    "max_depth":        randint(3, 9),        # capped at 8 to reduce overfitting
    "learning_rate":    uniform(0.01, 0.09),   # 0.01–0.10 (lower → less overfit)
    "subsample":        uniform(0.6, 0.4),
    "colsample_bytree": uniform(0.5, 0.5),
    "min_child_weight": randint(1, 11),
    "gamma":            uniform(0.0, 5.0),
    "reg_alpha":        uniform(0.001, 9.999),
    "reg_lambda":       uniform(0.001, 9.999),
}

base_xgb = XGBClassifier(
    eval_metric="aucpr",
    random_state=42,
    n_jobs=-1,
    tree_method="hist",
    verbosity=0,
    early_stopping_rounds=30,   # stop if val metric doesn't improve for 30 rounds
)

pr_auc_scorer = make_scorer(average_precision_score, needs_proba=True)

search = RandomizedSearchCV(
    estimator=base_xgb,
    param_distributions=param_distributions,
    n_iter=30,
    scoring=pr_auc_scorer,
    cv=3,
    random_state=42,
    n_jobs=-1,
    verbose=1,
)

print("  Running 30-iteration RandomizedSearchCV (optimising PR-AUC)...")
search.fit(X_tune, y_tune)

best_params = search.best_params_
best_params.update({
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
    "tree_method": "hist",
    "verbosity": 0,
    "early_stopping_rounds": 50,
})

print(f"\n  ✓ Best CV PR-AUC: {search.best_score_:.6f}")
print("  ✓ Best params:")
for k, v in search.best_params_.items():
    print(f"      {k}: {round(v, 4) if isinstance(v, float) else v}")


# =============================================================================
# 3. TRAIN FINAL MODEL on full training data
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 3: Training Final XGBoost")
print("=" * 60)

fraud_model = XGBClassifier(**best_params)
fraud_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],   # early stopping monitors validation performance
    verbose=False,
)
print(f"  ✓ Model trained  (best iteration: {fraud_model.best_iteration})")


# =============================================================================
# 4. THRESHOLD OPTIMISATION (on validation set — never on test)
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 4: Threshold Optimisation (validation set)")
print("=" * 60)

y_val_prob = fraud_model.predict_proba(X_val)[:, 1]

t_f1, score_f1        = best_threshold(y_val, y_val_prob, goal="f1")
t_recall, score_recall = best_threshold(y_val, y_val_prob, goal="recall")

print(f"  Best F1 threshold:     {t_f1:.2f}  (val F1     = {score_f1:.4f})")
print(f"  Best recall threshold: {t_recall:.2f}  (val recall = {score_recall:.4f})")

chosen_t = t_f1    # ← change to t_recall to catch more fraud
print(f"\n  → Using threshold = {chosen_t:.2f} for final test evaluation")


# =============================================================================
# 5. EVALUATE ON TEST SET
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 5: Final Test Evaluation")
print("=" * 60)

y_test_prob = fraud_model.predict_proba(X_test)[:, 1]

m_default = evaluate(y_test, y_test_prob, threshold=0.50,
                     label="XGBoost @ 0.50")
m_tuned   = evaluate(y_test, y_test_prob, threshold=chosen_t,
                     label=f"XGBoost @ {chosen_t:.2f}")


# =============================================================================
# 6. PLOTS
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 6: Saving Plots")
print("=" * 60)

save_curves(y_test, y_test_prob, label="XGBoost (tuned)")

# Feature importance
importances = fraud_model.feature_importances_
try:
    feat_names = pd.read_csv("processed_data/feature_names.csv")["feature_name"].tolist()
except Exception:
    feat_names = [f"f{i}" for i in range(len(importances))]

imp_df = (pd.DataFrame({"feature": feat_names, "importance": importances})
            .sort_values("importance", ascending=False)
            .head(20))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
ax.set_title("Top 20 Feature Importances")
ax.set_xlabel("Importance")
plt.tight_layout()
p = os.path.join(RESULTS_DIR, "feature_importance.png")
fig.savefig(p, dpi=150); plt.close(fig)
print(f"  ✓ Feature importance → {p}")


# =============================================================================
# 7. SHAP EXPLAINABILITY
# =============================================================================

if SHAP_AVAILABLE:
    print("\n" + "=" * 60)
    print("  STEP 7: SHAP Explainability")
    print("=" * 60)

    explainer = shap.TreeExplainer(fraud_model)
    idx = np.random.RandomState(42).choice(len(X_test),
                                           size=min(500, len(X_test)),
                                           replace=False)
    shap_vals = explainer.shap_values(X_test[idx])

    plt.figure()
    shap.summary_plot(shap_vals, X_test[idx],
                      feature_names=feat_names, show=False)
    p = os.path.join(RESULTS_DIR, "shap_summary.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✓ SHAP summary → {p}")


# =============================================================================
# 8. SAVE MODEL & RESULTS
# =============================================================================

print("\n" + "=" * 60)
print("  STEP 8: Saving Model & Results")
print("=" * 60)

fraud_model.save_model(os.path.join(RESULTS_DIR, "xgboost_fraud_model.json"))
print(f"  ✓ Model → {RESULTS_DIR}/xgboost_fraud_model.json")

results = {
    "baseline_roc_auc": round(roc_auc_score(y_test, y_base_prob), 6),
    "xgb_default":      m_default,
    "xgb_tuned":        m_tuned,
    "best_params":      {k: (round(v, 6) if isinstance(v, float) else v)
                         for k, v in best_params.items()},
}
with open(os.path.join(RESULTS_DIR, "results.json"), "w") as f:
    json.dump(results, f, indent=2)
print(f"  ✓ Results → {RESULTS_DIR}/results.json")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"""
{'='*60}
  DONE — COMPARISON
{'='*60}
  Baseline LogReg   ROC-AUC : {roc_auc_score(y_test, y_base_prob):.6f}
  XGBoost (t=0.50)  ROC-AUC : {m_default['roc_auc']}   F1: {m_default['f1']}   Recall: {m_default['recall']}
  XGBoost (t={chosen_t:.2f})  ROC-AUC : {m_tuned['roc_auc']}   F1: {m_tuned['f1']}   Recall: {m_tuned['recall']}

  Outputs saved to: {RESULTS_DIR}/
""")
