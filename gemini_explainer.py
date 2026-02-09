"""
Gemini Business Explanation Engine
===================================
Converts technical ML risk scores + SHAP factors into natural-language
business explanations using Google Gemini.

Flow:
  ML Model → Risk Score (e.g. 86%)
  SHAP     → Top risk drivers
  Gemini   → Human-friendly business explanation

Setup:
  1. pip install google-generativeai
  2. Set env var: GEMINI_API_KEY=<your-key>
     Or pass api_key to GeminiExplainer()

Get a free API key at: https://aistudio.google.com/apikey
"""

import os
from dotenv import load_dotenv

# Load .env file so GEMINI_API_KEY is available via os.environ
load_dotenv()
from typing import Dict, List, Optional

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


# ── Prompt template ──────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a fraud-risk analyst at a major bank. You receive technical output
from an ML fraud-detection system and must translate it into a clear,
concise business explanation that a non-technical compliance officer or
customer-service agent can understand.

Rules:
- Write 2-4 sentences maximum.
- Avoid technical jargon (no "SHAP", "PCA", "feature values").
- Refer to factors in plain business language (e.g. "transaction amount",
  "time of transaction", "spending pattern").
- State the risk level and recommended action clearly.
- If relevant features are PCA components (V1-V28), describe them as
  "unusual spending patterns" or "anomalous transaction characteristics".
- Be professional and factual — no speculation.
"""

USER_TEMPLATE = """\
Transaction Risk Assessment:
- Risk Score: {risk_pct}
- Risk Level: {risk_level}
- Recommended Action: {action}

Top contributing factors:
{factors_text}

Write a business-friendly explanation of why this transaction was flagged.
"""


class GeminiExplainer:
    """Generates business explanations using Google Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "Gemini 2.5 Flash Native Audio Dialog",
    ):
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai is required. Install with:\n"
                "  pip install google-generativeai"
            )

        self.api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self.api_key or self.api_key in ("your-api-key-here", "<your-key>"):
            raise ValueError(
                "Gemini API key not found or is a placeholder. Either:\n"
                "  1. Set GEMINI_API_KEY in your .env file\n"
                "  2. Pass api_key='...' to GeminiExplainer()\n"
                "Get a free key at: https://aistudio.google.com/apikey"
            )

        # Quick validation: Gemini keys typically start with 'AIza'
        if not self.api_key.startswith("AIza"):
            raise ValueError(
                f"Invalid API key format (expected 'AIza...'). "
                f"Check your .env file or passed key."
            )

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=SYSTEM_PROMPT,
        )

    # ── Core explain method ──────────────────────────────────────────────

    def explain(self, explanation: Dict) -> str:
        """
        Convert a single SHAP explanation dict into a business narrative.

        Parameters
        ----------
        explanation : dict
            Output from FraudScorer.explain() — must contain:
            risk_pct, risk_level, action, top_features

        Returns
        -------
        str : Business-friendly explanation from Gemini
        """
        factors_lines = []
        for f in explanation.get("top_features", []):
            name = f["feature"]
            # Make PCA components human-readable
            if name.startswith("V") and name[1:].isdigit():
                name = f"Anomalous pattern (component {name})"
            direction = "↑ increases risk" if f["shap_value"] > 0 else "↓ decreases risk"
            factors_lines.append(f"  - {name}: {direction} (value={f['feature_value']:.4f})")

        prompt = USER_TEMPLATE.format(
            risk_pct=explanation.get("risk_pct", "N/A"),
            risk_level=explanation.get("risk_level", "N/A"),
            action=explanation.get("action", "N/A"),
            factors_text="\n".join(factors_lines) if factors_lines else "  (no factors available)",
        )

        response = self.model.generate_content(prompt)
        return response.text.strip()

    # ── Batch explain ────────────────────────────────────────────────────

    def explain_batch(self, explanations: List[Dict]) -> List[str]:
        """Generate business explanations for multiple transactions."""
        results = []
        for exp in explanations:
            try:
                results.append(self.explain(exp))
            except Exception as e:
                results.append(f"[Gemini unavailable: {e}]")
        return results


# ── Fallback for when Gemini is unavailable ──────────────────────────────────

def fallback_explanation(explanation: Dict) -> str:
    """
    Generate a rule-based business explanation without Gemini.
    Used as fallback when API key is not set.
    """
    risk_pct = explanation.get("risk_pct", "N/A")
    risk_level = explanation.get("risk_level", "UNKNOWN")
    action = explanation.get("action", "N/A")
    features = explanation.get("top_features", [])

    # Translate features to business language
    risk_factors = []
    protective_factors = []

    for f in features[:5]:
        name = f["feature"]
        # Map technical names to business language
        if name == "Amount" or name == "Amount_Log":
            label = "transaction amount"
        elif name == "Hour":
            label = "time of transaction"
        elif name.startswith("Period_"):
            label = f"transaction during {name.replace('Period_', '').lower()} hours"
        elif name.startswith("AmountCat_"):
            label = f"{name.replace('AmountCat_', '').lower()} transaction value category"
        elif name == "Time":
            label = "timing relative to other transactions"
        elif name.startswith("V") and name[1:].isdigit():
            label = "unusual spending pattern"
        else:
            label = name.lower().replace("_", " ")

        if f["shap_value"] > 0:
            risk_factors.append(label)
        else:
            protective_factors.append(label)

    # Build narrative
    parts = []
    if risk_level == "HIGH":
        parts.append(f"This transaction has been flagged as HIGH risk ({risk_pct}).")
    elif risk_level == "MEDIUM":
        parts.append(f"This transaction shows MODERATE risk indicators ({risk_pct}).")
    else:
        parts.append(f"This transaction appears LOW risk ({risk_pct}).")

    if risk_factors:
        # De-duplicate (e.g. multiple V* features → one "unusual spending pattern")
        unique_risks = list(dict.fromkeys(risk_factors))
        parts.append(
            "Key risk drivers include: " + ", ".join(unique_risks[:3]) + "."
        )

    if protective_factors and risk_level != "HIGH":
        unique_safe = list(dict.fromkeys(protective_factors))
        parts.append(
            "However, " + ", ".join(unique_safe[:2]) + " are consistent with normal behaviour."
        )

    parts.append(f"Recommended action: {action}.")
    return " ".join(parts)


# ── CLI test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Gemini Business Explanation Engine — Test")
    print("=" * 60)

    # Test fallback first (no API key needed)
    sample_explanation = {
        "risk_pct": "86.32%",
        "risk_level": "HIGH",
        "action": "BLOCK — immediate investigation required",
        "top_features": [
            {"feature": "V14", "shap_value": 2.5, "feature_value": -5.234, "direction": "increases fraud risk"},
            {"feature": "Amount_Log", "shap_value": 1.8, "feature_value": 3.12, "direction": "increases fraud risk"},
            {"feature": "Hour", "shap_value": 0.9, "feature_value": 2.0, "direction": "increases fraud risk"},
            {"feature": "V12", "shap_value": -0.6, "feature_value": -1.05, "direction": "decreases fraud risk"},
            {"feature": "V4", "shap_value": 0.7, "feature_value": 3.12, "direction": "increases fraud risk"},
        ],
    }

    print("\n── Fallback (rule-based) explanation ──")
    print(fallback_explanation(sample_explanation))

    # Test Gemini if API key is available
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if api_key and GEMINI_AVAILABLE:
        print("\n── Gemini explanation ──")
        explainer = GeminiExplainer(api_key=api_key)
        print(explainer.explain(sample_explanation))
    else:
        print("\n  ℹ Set GEMINI_API_KEY to test Gemini integration.")
        if not GEMINI_AVAILABLE:
            print("  ℹ Install with: pip install google-generativeai")
