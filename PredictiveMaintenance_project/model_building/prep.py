# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Define constants for the dataset and output paths
api = HfApi(token=os.getenv("HF_TOKEN_"))
DATASET_PATH = "hf://datasets/Money2277/Engine-Predictive-Maintenance/data/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Create a copy of the data
data = df.copy()
target = "Engine Condition"

# drop duplicates
data = data.drop_duplicates()

# As seen in data analysis, droping the measurements or row where coolant temperature is very high > 100
data = data[data["Coolant temp"] <= 100].reset_index(drop=True)

# ----------------------------
# Combine features to form X (feature matrix)
# ----------------------------
X = data.drop(columns=[target_col])

# ----------------------------
# Define target vector y
# ----------------------------
y = data[target]

# ----------------------------
# Split dataset into training and test sets
# ----------------------------
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

Xtrain.to_csv("Xtrain.csv",index=False)
Xtest.to_csv("Xtest.csv",index=False)
ytrain.to_csv("ytrain.csv",index=False)
ytest.to_csv("ytest.csv",index=False)


files = ["Xtrain.csv","Xtest.csv","ytrain.csv","ytest.csv"]

for file_path in files:
    filename = file_path.split("/")[-1]
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed_data/{filename}",
        repo_id="Money2277/Engine-Predictive-Maintenance",
        repo_type="dataset",
    )
