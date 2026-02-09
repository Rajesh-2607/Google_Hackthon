# Streamlit Cloud Deployment Guide

## Overview

Deploy the **Enterprise Fraud Risk Intelligence Platform** to [Streamlit Community Cloud](https://share.streamlit.io/) for a free public URL. Data files are stored on **Google Drive** and auto-downloaded at app startup.

---

## Architecture

```
GitHub Repo (code only)          Google Drive (your Google_Hackthon folder)
├── app.py                       ├── processed_data/   ← share this folder
├── fraud_scorer.py              │   ├── X_train.npy
├── drift_monitor.py             │   ├── X_test.npy
├── gemini_explainer.py          │   ├── ... (all .npy + .csv files)
├── data_loader.py               ├── results/          ← share this folder
├── requirements.txt             │   ├── xgboost_fraud_model.json
├── .streamlit/config.toml       │   ├── results.json
└── .gitignore                   │   └── ... (plots + reports)
                                 └── creditcard.csv (not needed at runtime)
         ↓                                ↓
    Streamlit Cloud ←── gdown downloads folders on first run ──┘
    (public URL)
```

You only need **2 folder IDs** — no individual file IDs needed!

---

## Step-by-Step Deployment

### Step 1: Share Your Google Drive Folders

You already have `processed_data/` and `results/` in your `Google_Hackthon` Drive folder. Now share them:

**For the `processed_data` folder:**
1. Right-click `processed_data` → **Share**
2. Under "General access", change to **"Anyone with the link"**
3. Set permission to **Viewer**
4. Click **Copy link**

The link looks like:
```
https://drive.google.com/drive/folders/1AbCdEfGhIjKlMnOpQrStUvWxYz?usp=sharing
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        This is your FOLDER ID
```

**Repeat for the `results` folder.**

Save both folder IDs — you'll need them in Step 4.

### Step 2: Push Code to GitHub

```bash
# Initialize git (if not already)
git init

# Add remote (create a repo on GitHub first)
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git

# Stage and commit
git add .
git commit -m "Fraud Risk Intelligence Platform"

# Push
git push -u origin main
```

**Files that will be pushed** (code only — data is in .gitignore):
- `app.py`, `fraud_scorer.py`, `drift_monitor.py`, `gemini_explainer.py`
- `preprocessed.py`, `XGBoost.py`, `data_loader.py`
- `requirements.txt`, `README.md`, `DEPLOY.md`
- `.streamlit/config.toml`
- `.gitignore`

### Step 3: Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with your **GitHub account**
3. Click **"New app"**
4. Select:
   - **Repository:** `YOUR_USERNAME/YOUR_REPO`
   - **Branch:** `main`
   - **Main file path:** `app.py`
5. Click **"Advanced settings"** → go to **"Secrets"** tab
6. Paste the secrets from Step 4 below
7. Click **"Deploy!"**

### Step 4: Configure Secrets

In the **Secrets** tab (Step 3.5 above), paste this — replacing with your actual folder IDs:

```toml
# Google Drive Folder IDs (REQUIRED)
GDRIVE_PROCESSED_DATA_FOLDER = "1AbCdEfGhIjKlMnOpQrStUvWxYz"
GDRIVE_RESULTS_FOLDER        = "1ZyXwVuTsRqPoNmLkJiHgFeDcBa"

# Gemini API Key (OPTIONAL — for AI-powered explanations)
# Get free key at: https://aistudio.google.com/apikey
GEMINI_API_KEY = ""
```

That's it! Just **2 folder IDs** and you're done.

> You can also update secrets after deployment: **App menu (⋮)** → **Settings** → **Secrets**

---

## How It Works at Runtime

1. App starts on Streamlit Cloud
2. `data_loader.py` checks if `processed_data/` and `results/` exist locally
3. If missing → reads folder IDs from Streamlit secrets
4. Uses `gdown` to download entire folders from Google Drive
5. App loads normally with all data available
6. Files persist until the app sleeps (~7 days of inactivity)

---

## Local Development

Your `processed_data/` and `results/` folders already exist locally, so the app skips downloading:

```bash
streamlit run app.py
```

If you want to test the download flow locally, create `.streamlit/secrets.toml`:
```bash
copy .streamlit\secrets.example.toml .streamlit\secrets.toml
# Edit and fill in your folder IDs
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "Missing required data files" | Check that both folder IDs are set in Streamlit Cloud secrets |
| Download fails | Ensure folders are shared as **"Anyone with the link"** (not restricted) |
| App is slow on first load | Normal — downloading ~120 MB of data. Subsequent loads use cached files |
| App crashes with memory error | Streamlit free tier has ~1 GB RAM — should be fine for this dataset |
| Gemini explanations not working | Set `GEMINI_API_KEY` in secrets (optional — rule-based fallback works without it) |
| `ModuleNotFoundError` | Ensure `requirements.txt` is committed and includes all dependencies |
| App sleeps after inactivity | Normal on free tier — data re-downloads when someone visits again |

---

## Free Tier Limits (Streamlit Community Cloud)

- **Apps:** Unlimited public apps
- **RAM:** ~1 GB
- **Storage:** Ephemeral (data re-downloads after app sleeps)
- **Sleep:** App sleeps after ~7 days of inactivity (wakes on visit)
- **URL format:** `https://your-app-name.streamlit.app`
