import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
import xgboost as xgb
import joblib
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow
import os

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment("Engine-Predictive-Maintenance-Experiment")

# Hugging Face API authentication
api = HfApi(token=os.getenv("HF_TOKEN_"))

dataset_repo = "Money2277/Engine-Predictive-Maintenance"

Xtrain_path = f"hf://datasets/{dataset_repo}/processed_data/Xtrain.csv"
Xtest_path = f"hf://datasets/{dataset_repo}/processed_data/Xtest.csv"
ytrain_path = f"hf://datasets/{dataset_repo}/processed_data/ytrain.csv"
ytest_path = f"hf://datasets/{dataset_repo}/processed_data/ytest.csv"

Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).iloc[:, 0]
ytest = pd.read_csv(ytest_path).iloc[:, 0]

numeric_features = [
    'Engine rpm', 'Lub oil pressure', 'Fuel pressure',
    'Coolant pressure', 'lub oil temp', 'Coolant temp',
]

neg, pos = (ytrain == 0).sum(), (ytrain == 1).sum()
scale_pos_weight = neg / pos

xgb_model = xgb.XGBClassifier(
    scale_pos_weight=scale_pos_weight, eval_metric="logloss", random_state=42,
)
model_pipeline = Pipeline(steps=[("xgb", xgb_model)])

param_grid = {
    'xgb__n_estimators': [100, 150, 200],
    'xgb__max_depth': [2, 3, 4],
    'xgb__learning_rate': [0.01, 0.05, 0.1],
    'xgb__colsample_bytree': [0.6, 0.8],
    'xgb__reg_lambda': [1.0, 2.0, 5.0],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=cv, n_jobs=-1, scoring='roc_auc')
    grid_search.fit(Xtrain, ytrain)

    mlflow.log_params(grid_search.best_params_)
    mlflow.log_param("scale_pos_weight", scale_pos_weight)

    best_model = grid_search.best_estimator_
    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]

    mlflow.log_metrics({
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "test_precision": precision_score(ytest, y_pred_test),
        "test_recall": recall_score(ytest, y_pred_test),
        "test_f1": f1_score(ytest, y_pred_test),
        "test_roc_auc": roc_auc_score(ytest, y_proba_test),
    })

    print("Best Params:\n", grid_search.best_params_)
    print("\nTest Classification Report:")
    print(classification_report(ytest, y_pred_test, digits=3))
    print("Test Confusion Matrix:")
    print(confusion_matrix(ytest, y_pred_test))

    model_path = "best_engine_maintenance_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")

    repo_id = "Money2277/Engine-Predictive-Maintenance"
    repo_type = "model"
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
