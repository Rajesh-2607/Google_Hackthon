"""
Google Drive Data Loader
========================
Downloads processed_data/ and results/ folders from Google Drive
for Streamlit Cloud deployment.

Setup:
  1. Upload processed_data/ and results/ folders to Google Drive
  2. Share each FOLDER as "Anyone with the link can view"
  3. Copy the share link (or folder ID)
  4. Set them in your .env file:
       GDRIVE_PROCESSED_DATA_FOLDER=https://drive.google.com/drive/folders/xxx
       GDRIVE_RESULTS_FOLDER=https://drive.google.com/drive/folders/yyy

  Accepts either:
    - Full URL:  https://drive.google.com/drive/folders/<FOLDER_ID>?usp=sharing
    - Just the ID:  1AbCdEfGhIjKlMnOpQrStUvWxYz
"""

import os
import gdown
import streamlit as st
from pathlib import Path
from typing import Optional


# ── Folders the app needs at runtime ────────────────────────────────────────

REQUIRED_FOLDERS = {
    "processed_data": "GDRIVE_PROCESSED_DATA_FOLDER",
    "results":        "GDRIVE_RESULTS_FOLDER",
}

# Key files that MUST exist for the app to work
REQUIRED_FILES = [
    "processed_data/X_train.npy",
    "processed_data/X_test.npy",
    "processed_data/X_val.npy",
    "processed_data/y_train.npy",
    "processed_data/y_test.npy",
    "processed_data/y_val.npy",
    "processed_data/feature_names.csv",
    "results/xgboost_fraud_model.json",
    "results/results.json",
]


def _extract_folder_id(value: str) -> str:
    """
    Accept a full Google Drive folder URL or a raw folder ID.
    Extracts and returns just the folder ID.
    """
    value = value.strip()
    # Full URL: https://drive.google.com/drive/folders/<ID>?usp=sharing
    if "/folders/" in value:
        return value.split("/folders/")[1].split("?")[0].split("/")[0]
    return value


def _get_secret(key: str) -> Optional[str]:
    """Get a value from .env / environment variables or Streamlit secrets."""
    # Check environment variables first (.env loaded by app.py via load_dotenv)
    val = os.environ.get(key)
    if val and val.strip():
        return _extract_folder_id(val)
    # Fall back to Streamlit secrets (for Streamlit Cloud)
    try:
        val = st.secrets.get(key)
        if val:
            return _extract_folder_id(str(val))
    except Exception:
        pass
    return None


def download_folder_from_gdrive(folder_id: str, destination: str) -> bool:
    """
    Download an entire Google Drive folder using gdown.
    The folder must be shared as 'Anyone with the link'.
    """
    url = f"https://drive.google.com/drive/folders/{folder_id}"
    os.makedirs(destination, exist_ok=True)
    try:
        gdown.download_folder(
            url=url,
            output=destination,
            quiet=False,
            use_cookies=False,
        )
        return True
    except Exception as e:
        st.warning(f"Folder download failed for `{destination}`: {e}")
        return False


def ensure_data_available() -> bool:
    """
    Check if all required data files exist locally.
    If not, download folders from Google Drive.

    Returns True if all required files are available.
    """
    # Check if everything is already present
    if files_exist_locally():
        return True

    # Download missing folders
    download_count = 0
    for local_folder, secret_key in REQUIRED_FOLDERS.items():
        # Skip if folder already has files
        if os.path.isdir(local_folder) and os.listdir(local_folder):
            continue

        folder_id = _get_secret(secret_key)
        if not folder_id:
            continue

        st.toast(f"⬇️ Downloading {local_folder}/ from Google Drive...")
        if download_folder_from_gdrive(folder_id, local_folder):
            download_count += 1

    if download_count > 0:
        st.toast(f"✅ Downloaded {download_count} folder(s) from Google Drive")

    # Final check — are all required files present now?
    missing = [f for f in REQUIRED_FILES if not os.path.exists(f)]
    if missing:
        # Build helpful error message
        missing_secrets = []
        for local_folder, secret_key in REQUIRED_FOLDERS.items():
            if not _get_secret(secret_key):
                missing_secrets.append(f"**{secret_key}**")

        msg = "**Missing required data files!**\n\n"
        if missing_secrets:
            msg += (
                "Set these Google Drive folder IDs in **Streamlit Cloud Secrets**:\n\n"
                + "\n".join(f"- {s}" for s in missing_secrets)
                + "\n\n"
            )
        msg += (
            "Missing files:\n"
            + "\n".join(f"- `{f}`" for f in missing[:5])
        )
        if len(missing) > 5:
            msg += f"\n- ... and {len(missing) - 5} more"

        st.error(msg)
        return False

    return True


def files_exist_locally() -> bool:
    """Quick check — are all required files already present?"""
    return all(os.path.exists(p) for p in REQUIRED_FILES)
