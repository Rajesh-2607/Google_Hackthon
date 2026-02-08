print("ML TRAIN SCRIPT STARTED")
import numpy as np

# Load processed data
X_train = np.load("processed_data/X_train.npy")
X_val   = np.load("processed_data/X_val.npy")
X_test  = np.load("processed_data/X_test.npy")

y_train = np.load("processed_data/y_train.npy")
y_val   = np.load("processed_data/y_val.npy")
y_test  = np.load("processed_data/y_test.npy")

print("Train:", X_train.shape, y_train.shape)
print("Val:", X_val.shape, y_val.shape)
print("Test:", X_test.shape, y_test.shape)

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

baseline = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

baseline.fit(X_train, y_train)

y_pred_prob = baseline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)


print("\nBASELINE RESULTS")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42,
    n_jobs=-1
)

xgb_model.fit(X_train, y_train)

y_pred_prob = xgb_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nXGBOOST RESULTS")
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_pred_prob))

def decision(score):
    if score < 0.3:
        return "ALLOW"
    elif score < 0.7:
        return "REVIEW"
    else:
        return "BLOCK"

for i in range(5):
    print(f"Transaction {i} → Fraud Score: {y_pred_prob[i]:.3f} → Decision: {decision(y_pred_prob[i])}")

