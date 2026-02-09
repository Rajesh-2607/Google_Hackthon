"""
Enterprise Fraud Risk Intelligence Dashboard
=============================================
Streamlit-based interactive dashboard for:
  1. Model Performance Overview
  2. Live Transaction Risk Scoring
  3. Per-Transaction SHAP Explanations
  4. Feature Importance Analysis
  5. Data & Model Drift Monitoring

Run:  streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import json, os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file (picks up GEMINI_API_KEY automatically)
load_dotenv()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve,
)

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# Local modules
from fraud_scorer import FraudScorer, RISK_ACTIONS
from drift_monitor import DriftMonitor
from gemini_explainer import GEMINI_AVAILABLE as _GEMINI_LIB, fallback_explanation

try:
    from gemini_explainer import GeminiExplainer
except Exception:
    pass

# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Fraud Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Caching loaders ─────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    model = XGBClassifier()
    model.load_model("results/xgboost_fraud_model.json")
    return model

@st.cache_resource
def load_scorer():
    return FraudScorer()

@st.cache_data
def load_test_data():
    X = np.load("processed_data/X_test.npy")
    y = np.load("processed_data/y_test.npy")
    return X, y

@st.cache_data
def load_val_data():
    X = np.load("processed_data/X_val.npy")
    y = np.load("processed_data/y_val.npy")
    return X, y

@st.cache_data
def load_train_data():
    return np.load("processed_data/X_train.npy")

@st.cache_data
def load_results():
    with open("results/results.json") as f:
        return json.load(f)

@st.cache_data
def load_feature_names():
    try:
        return pd.read_csv("processed_data/feature_names.csv")["feature_name"].tolist()
    except Exception:
        return None

@st.cache_data
def get_test_predictions(_model, X_test):
    return _model.predict_proba(X_test)[:, 1]


# ── Sidebar navigation ──────────────────────────────────────────────────────

st.sidebar.title("🏦 Fraud Risk Intelligence")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    [
        "📊 Performance Overview",
        "🔍 Transaction Scorer",
        "💡 SHAP Explanations",
        "📈 Feature Importance",
        "🔄 Drift Monitor",
    ],
)

st.sidebar.markdown("---")
st.sidebar.caption("Enterprise Fraud Risk Intelligence Platform v2.0")


# =============================================================================
# PAGE 1 — Model Performance Overview
# =============================================================================

if page == "📊 Performance Overview":
    st.title("📊 Model Performance Overview")
    st.markdown("Comprehensive evaluation of the fraud detection model on the held-out test set.")

    results = load_results()
    model = load_model()
    X_test, y_test = load_test_data()
    y_prob = get_test_predictions(model, X_test)

    # ── KPI cards ────────────────────────────────────────────────────────
    tuned = results.get("xgb_tuned", {})
    baseline_auc = results.get("baseline_roc_auc", 0)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{tuned.get('roc_auc', 0):.4f}",
              delta=f"+{tuned.get('roc_auc', 0) - baseline_auc:.4f} vs baseline")
    c2.metric("PR-AUC", f"{tuned.get('pr_auc', 0):.4f}")
    c3.metric("Precision", f"{tuned.get('precision', 0)*100:.1f}%")
    c4.metric("Recall", f"{tuned.get('recall', 0)*100:.1f}%")
    c5.metric("F1 Score", f"{tuned.get('f1', 0):.4f}")

    st.markdown("---")

    # ── ROC + PR Curves ──────────────────────────────────────────────────
    col_l, col_r = st.columns(2)

    with col_l:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.plot(fpr, tpr, lw=2, color="#3498db",
                 label=f"XGBoost (AUC={roc_auc_score(y_test, y_prob):.4f})")
        ax1.plot([0, 1], [0, 1], "k--", lw=1)
        ax1.set(xlabel="False Positive Rate", ylabel="True Positive Rate")
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        st.pyplot(fig1)
        plt.close(fig1)

    with col_r:
        st.subheader("Precision-Recall Curve")
        prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_prob)
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.plot(rec_curve, prec_curve, lw=2, color="#e74c3c",
                 label=f"XGBoost (AP={average_precision_score(y_test, y_prob):.4f})")
        ax2.set(xlabel="Recall", ylabel="Precision")
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)
        st.pyplot(fig2)
        plt.close(fig2)

    # ── Confusion Matrix ─────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Confusion Matrix")

    threshold = tuned.get("threshold", 0.5)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    cm_col1, cm_col2 = st.columns([1, 2])
    with cm_col1:
        fig3, ax3 = plt.subplots(figsize=(5, 4))
        im = ax3.imshow(cm, cmap="Blues")
        ax3.set_xticks([0, 1]); ax3.set_yticks([0, 1])
        ax3.set_xticklabels(["Legit", "Fraud"]); ax3.set_yticklabels(["Legit", "Fraud"])
        ax3.set_xlabel("Predicted"); ax3.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                         color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
        ax3.set_title(f"Threshold = {threshold}")
        plt.tight_layout()
        st.pyplot(fig3)
        plt.close(fig3)

    with cm_col2:
        st.markdown(f"""
        | Metric | Value |
        |--------|-------|
        | **True Positives (Fraud caught)** | {tp:,} |
        | **True Negatives (Legit correct)** | {tn:,} |
        | **False Positives (False alarms)** | {fp:,} |
        | **False Negatives (Missed fraud)** | {fn:,} |
        | **Optimal Threshold** | {threshold} |
        """)

    # ── Score distribution ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Prediction Score Distribution")
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.hist(y_prob[y_test == 0], bins=80, alpha=0.6, label="Legit", color="#2ecc71", density=True)
    ax4.hist(y_prob[y_test == 1], bins=80, alpha=0.6, label="Fraud", color="#e74c3c", density=True)
    ax4.axvline(threshold, color="black", ls="--", lw=1.5, label=f"Threshold={threshold}")
    ax4.set(xlabel="Predicted Fraud Probability", ylabel="Density")
    ax4.legend()
    ax4.grid(alpha=0.3)
    st.pyplot(fig4)
    plt.close(fig4)


# =============================================================================
# PAGE 2 — Transaction Risk Scorer
# =============================================================================

elif page == "🔍 Transaction Scorer":
    st.title("🔍 Real-Time Transaction Risk Scorer")
    st.markdown("Score individual or batch transactions and get instant risk assessments.")

    scorer = load_scorer()
    X_test, y_test = load_test_data()
    feature_names = load_feature_names()

    st.markdown("---")

    tab1, tab2 = st.tabs(["📝 Manual Input", "📂 Sample from Test Set"])

    with tab1:
        st.markdown("Enter feature values (comma-separated, one transaction per line):")
        raw_input = st.text_area(
            "Transaction features",
            placeholder=f"e.g. paste {X_test.shape[1]} comma-separated values",
            height=120,
        )
        if st.button("Score Transaction", key="manual_score"):
            if raw_input.strip():
                try:
                    lines = [l.strip() for l in raw_input.strip().split("\n") if l.strip()]
                    X_input = np.array([[float(v) for v in line.split(",")] for line in lines])
                    results = scorer.score(X_input)
                    for r in results:
                        lvl = r["risk_level"]
                        icon = "🔴" if lvl == "HIGH" else ("🟡" if lvl == "MEDIUM" else "🟢")
                        st.markdown(f"### {icon} Transaction {r['transaction_index']+1}")
                        st.markdown(f"**Risk Score:** {r['risk_pct']}  |  "
                                    f"**Level:** {r['risk_level']}  |  "
                                    f"**Action:** {r['action']}")
                except Exception as e:
                    st.error(f"Error parsing input: {e}")
            else:
                st.warning("Please enter transaction data.")

    with tab2:
        n_samples = st.slider("Number of test transactions to score", 5, 200, 20)
        if st.button("Score Sample", key="sample_score"):
            idx = np.random.RandomState(42).choice(len(X_test), size=n_samples, replace=False)
            sample_X = X_test[idx]
            sample_y = y_test[idx]

            results = scorer.score(sample_X)
            df = pd.DataFrame(results)
            df["actual_label"] = ["Fraud" if y == 1 else "Legit" for y in sample_y]

            # Colour-coded risk
            def _colour(level):
                if level == "HIGH":   return "background-color: #ffcccc"
                if level == "MEDIUM": return "background-color: #fff3cd"
                return "background-color: #d4edda"

            st.dataframe(
                df[["transaction_index", "risk_pct", "risk_level", "action", "actual_label"]],
                use_container_width=True,
            )

            # Summary
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            col1.metric("🔴 High Risk", f"{(df['risk_level']=='HIGH').sum()}")
            col2.metric("🟡 Medium Risk", f"{(df['risk_level']=='MEDIUM').sum()}")
            col3.metric("🟢 Low Risk", f"{(df['risk_level']=='LOW').sum()}")


# =============================================================================
# PAGE 3 — SHAP Explanations
# =============================================================================

elif page == "💡 SHAP Explanations":
    st.title("💡 Explainable AI — SHAP Analysis")

    if not SHAP_AVAILABLE:
        st.error("SHAP is not installed. Run `pip install shap` to enable this page.")
    else:
        scorer = load_scorer()
        X_test, y_test = load_test_data()
        feature_names = load_feature_names()
        model = load_model()

        st.markdown("---")

        tab_global, tab_local, tab_gemini = st.tabs(
            ["🌐 Global Explanation", "🔬 Per-Transaction", "🤖 Gemini Business Explanation"]
        )

        with tab_global:
            st.subheader("SHAP Summary Plot (Global Feature Impact)")
            shap_img = "results/shap_summary.png"
            if os.path.exists(shap_img):
                st.image(shap_img, use_container_width=True)
            else:
                st.info("Generating SHAP summary (may take a moment)...")
                explainer = shap.TreeExplainer(model)
                sample_idx = np.random.RandomState(42).choice(len(X_test), size=min(500, len(X_test)), replace=False)
                shap_vals = explainer.shap_values(X_test[sample_idx])
                fig, ax = plt.subplots()
                shap.summary_plot(shap_vals, X_test[sample_idx],
                                  feature_names=feature_names, show=False)
                st.pyplot(fig)
                plt.close(fig)

        with tab_local:
            st.subheader("Per-Transaction Explanation")
            txn_idx = st.number_input(
                "Transaction index (from test set)", min_value=0,
                max_value=len(X_test)-1, value=0, step=1,
            )

            if st.button("Explain Transaction"):
                with st.spinner("Computing SHAP values..."):
                    explanation = scorer.explain(X_test[txn_idx:txn_idx+1], top_k=10)[0]

                lvl = explanation["risk_level"]
                icon = "🔴" if lvl == "HIGH" else ("🟡" if lvl == "MEDIUM" else "🟢")

                st.markdown(f"### {icon} Risk: {explanation['risk_pct']} — {explanation['risk_level']}")
                st.markdown(f"**Action:** {explanation['action']}")
                st.markdown(f"**Actual Label:** {'Fraud' if y_test[txn_idx]==1 else 'Legit'}")

                st.markdown("---")
                st.markdown("**Top Contributing Features:**")

                for feat in explanation["top_features"]:
                    direction_icon = "🔺" if feat["shap_value"] > 0 else "🔻"
                    st.markdown(
                        f"- {direction_icon} **{feat['feature']}** = {feat['feature_value']:.4f} "
                        f"(SHAP: {feat['shap_value']:+.4f}) — *{feat['direction']}*"
                    )

                st.markdown("---")
                st.code(explanation["narrative"], language=None)

                # Waterfall plot
                st.subheader("SHAP Waterfall Plot")
                explainer = shap.TreeExplainer(model)
                sv = explainer(X_test[txn_idx:txn_idx+1])
                fig_w, ax_w = plt.subplots()
                shap.plots.waterfall(sv[0], max_display=15, show=False)
                st.pyplot(fig_w)
                plt.close(fig_w)


        with tab_gemini:
            st.subheader("🤖 Gemini-Powered Business Explanations")
            st.markdown(
                "Converts technical SHAP output into plain-English business "
                "narratives using Google Gemini. The API key is loaded from "
                "the `.env` file. If not set, a rule-based fallback is used."
            )

            # ── API Key Status Check ─────────────────────────────
            _env_key = os.environ.get("GEMINI_API_KEY", "")
            if _env_key and _env_key not in ("your-api-key-here", "<your-key>"):
                st.success("✅ Gemini API key loaded from `.env` file.")
            else:
                st.warning(
                    "⚠️ No valid Gemini API key found. "
                    "Add your key to the `.env` file as `GEMINI_API_KEY=AIza...` — "
                    "[Get a free key](https://aistudio.google.com/apikey). "
                    "Rule-based fallback will be used."
                )

            gem_txn_idx = st.number_input(
                "Transaction index", min_value=0,
                max_value=len(X_test)-1, value=0, step=1,
                key="gem_txn_idx",
            )

            if st.button("Generate Business Explanation", key="gemini_btn"):
                with st.spinner("Analysing transaction..."):
                    explanation = scorer.explain(X_test[gem_txn_idx:gem_txn_idx+1], top_k=5)[0]

                lvl = explanation["risk_level"]
                icon = "🔴" if lvl == "HIGH" else ("🟡" if lvl == "MEDIUM" else "🟢")
                st.markdown(f"### {icon} Risk: {explanation['risk_pct']} — {lvl}")
                st.markdown(f"**Actual Label:** {'Fraud' if y_test[gem_txn_idx]==1 else 'Legit'}")

                # Technical SHAP summary
                st.markdown("---")
                st.markdown("**Technical Risk Factors (SHAP):**")
                for feat in explanation["top_features"]:
                    arrow = "🔺" if feat["shap_value"] > 0 else "🔻"
                    st.markdown(
                        f"- {arrow} **{feat['feature']}** = {feat['feature_value']:.4f} "
                        f"(SHAP: {feat['shap_value']:+.4f})"
                    )

                # Business explanation
                st.markdown("---")
                st.markdown("**📝 Business Explanation:**")

                use_gemini = False
                api_key = os.environ.get("GEMINI_API_KEY", "")
                if api_key and _GEMINI_LIB:
                    try:
                        gem = GeminiExplainer(api_key=api_key)
                        with st.spinner("Generating Gemini explanation..."):
                            biz_text = gem.explain(explanation)
                        use_gemini = True
                    except Exception as e:
                        st.warning(f"Gemini unavailable ({e}). Using rule-based fallback.")
                        biz_text = fallback_explanation(explanation)
                else:
                    biz_text = fallback_explanation(explanation)

                st.info(biz_text)

                if use_gemini:
                    st.caption("💡 Explanation generated by Google Gemini")
                else:
                    st.caption(
                        "💡 Rule-based explanation (set GEMINI_API_KEY in .env file "
                        "for AI-powered narratives)"
                    )


# =============================================================================
# PAGE 4 — Feature Importance
# =============================================================================

elif page == "📈 Feature Importance":
    st.title("📈 Feature Importance Analysis")

    model = load_model()
    feature_names = load_feature_names()
    importances = model.feature_importances_

    names = feature_names or [f"f{i}" for i in range(len(importances))]
    imp_df = (pd.DataFrame({"Feature": names, "Importance": importances})
              .sort_values("Importance", ascending=False))

    st.markdown("---")

    top_n = st.slider("Show top N features", 5, len(imp_df), 20)
    top = imp_df.head(top_n)

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.35)))
    ax.barh(top["Feature"][::-1], top["Importance"][::-1], color="#3498db")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top {top_n} Feature Importances")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.markdown("---")
    st.subheader("Full Feature Ranking")
    st.dataframe(imp_df.reset_index(drop=True), use_container_width=True)


# =============================================================================
# PAGE 5 — Drift Monitor
# =============================================================================

elif page == "🔄 Drift Monitor":
    st.title("🔄 Data & Model Drift Monitor")
    st.markdown("Detect distribution shifts between training and test data that signal model degradation.")

    monitor = DriftMonitor()
    model = load_model()
    X_test, y_test = load_test_data()
    X_val, y_val = load_val_data()

    y_prob_test = get_test_predictions(model, X_test)
    y_prob_val = model.predict_proba(X_val)[:, 1]

    st.markdown("---")

    tab_feat, tab_pred, tab_health = st.tabs([
        "📊 Feature Drift", "📉 Prediction Drift", "💚 Model Health"
    ])

    with tab_feat:
        st.subheader("Feature Drift Analysis (Train → Test)")

        with st.spinner("Computing PSI & KS statistics..."):
            feat_df = monitor.feature_drift(X_test)

        # Summary KPIs
        n_stable   = (feat_df["psi_status"] == "STABLE").sum()
        n_moderate = (feat_df["psi_status"] == "MODERATE_DRIFT").sum()
        n_severe   = (feat_df["psi_status"] == "SIGNIFICANT_DRIFT").sum()

        kc1, kc2, kc3 = st.columns(3)
        kc1.metric("✅ Stable Features", n_stable)
        kc2.metric("⚠️ Moderate Drift", n_moderate)
        kc3.metric("🚨 Significant Drift", n_severe)

        # PSI Chart
        fig_psi, ax_psi = plt.subplots(figsize=(10, max(5, len(feat_df) * 0.3)))
        colours = ["#e74c3c" if s == "SIGNIFICANT_DRIFT"
                    else "#f39c12" if s == "MODERATE_DRIFT"
                    else "#2ecc71" for s in feat_df["psi_status"]]
        ax_psi.barh(feat_df["feature"], feat_df["psi"], color=colours)
        ax_psi.axvline(0.10, color="orange", ls="--", lw=1, label="Moderate (0.10)")
        ax_psi.axvline(0.25, color="red",    ls="--", lw=1, label="Significant (0.25)")
        ax_psi.set_xlabel("PSI")
        ax_psi.set_title("Population Stability Index per Feature")
        ax_psi.legend()
        plt.tight_layout()
        st.pyplot(fig_psi)
        plt.close(fig_psi)

        st.markdown("---")
        st.subheader("Detailed Drift Table")
        st.dataframe(feat_df, use_container_width=True)

    with tab_pred:
        st.subheader("Prediction Distribution Drift (Val → Test)")

        pred_drift = monitor.prediction_drift(y_prob_val, y_prob_test)

        pc1, pc2, pc3 = st.columns(3)
        psi_status = pred_drift["prediction_psi_status"]
        status_icon = "✅" if psi_status == "STABLE" else ("⚠️" if "MODERATE" in psi_status else "🚨")
        pc1.metric("Prediction PSI", f"{pred_drift['prediction_psi']:.4f}")
        pc2.metric("Status", f"{status_icon} {psi_status}")
        pc3.metric("KS Statistic", f"{pred_drift['prediction_ks_stat']:.4f}")

        # Overlay histograms
        fig_pd, ax_pd = plt.subplots(figsize=(10, 4))
        ax_pd.hist(y_prob_val, bins=60, alpha=0.5, density=True, label="Validation", color="#3498db")
        ax_pd.hist(y_prob_test, bins=60, alpha=0.5, density=True, label="Test",       color="#e74c3c")
        ax_pd.set(xlabel="Predicted Fraud Probability", ylabel="Density",
                  title="Prediction Distribution: Validation vs Test")
        ax_pd.legend()
        ax_pd.grid(alpha=0.3)
        st.pyplot(fig_pd)
        plt.close(fig_pd)

        st.markdown("---")
        st.json(pred_drift)

    with tab_health:
        st.subheader("Model Health Monitor")

        results = load_results()
        tuned = results.get("xgb_tuned", {})

        # Compute current test metrics
        threshold = tuned.get("threshold", 0.5)
        y_pred = (y_prob_test >= threshold).astype(int)
        current_metrics = {
            "roc_auc":   round(roc_auc_score(y_test, y_prob_test), 6),
            "pr_auc":    round(average_precision_score(y_test, y_prob_test), 6),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 6),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 6),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 6),
        }

        decay = DriftMonitor.performance_decay(tuned, current_metrics, decay_threshold=0.05)

        for metric_name, info in decay.items():
            col_m, col_b, col_c, col_s = st.columns(4)
            col_m.write(f"**{metric_name.upper()}**")
            col_b.metric("Baseline", f"{info['baseline']:.4f}")
            col_c.metric("Current", f"{info['current']:.4f}",
                         delta=f"{-info['drop']:.4f}" if info['drop'] != 0 else "0")
            status_emoji = "✅" if info["status"] == "HEALTHY" else "🚨"
            col_s.metric("Status", f"{status_emoji} {info['status']}")

        st.markdown("---")

        # Overall health verdict
        any_degraded = any(v["status"] == "DEGRADED" for v in decay.values())
        if any_degraded:
            st.error("🚨 **Model Health: DEGRADED** — Performance has dropped significantly. "
                     "Consider retraining the model with fresh data.")
        else:
            st.success("✅ **Model Health: HEALTHY** — All metrics are within acceptable bounds.")
