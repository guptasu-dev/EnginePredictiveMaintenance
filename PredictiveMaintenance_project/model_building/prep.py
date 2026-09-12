# for data manipulation
import pandas as pd
# for creating a folder
import os
# for train/test split
from sklearn.model_selection import train_test_split
# for hugging face space authentication to upload files
from huggingface_hub import HfApi

# Define constants for the dataset and output paths
hf_token = os.getenv("HF_TOKEN_")
if not hf_token:
    raise ValueError("HF_TOKEN environment variable is not set — check the GitHub Actions secret.")

api = HfApi(token=hf_token)
whoami = api.whoami()
print(f"Authenticated to Hugging Face as: {whoami['name']}")
DATASET_PATH = "hf://datasets/Money2277/Engine-Predictive-Maintenance/data/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully.")

# Create a copy of the data
data = df.copy()
target = "Engine Condition"

# drop duplicates
data = data.drop_duplicates()

# As seen in EDA, dropping the two rows where coolant temperature is
# physically implausible (> 100 degrees C) — identified as sensor error
data = data[data["Coolant temp"] <= 100].reset_index(drop=True)

# ----------------------------
# No columns dropped: EDA (correlation analysis) found no multicollinearity
# and no redundant features among the six sensor readings, so all are
# retained for modeling.
# ----------------------------

# Combine features to form X (feature matrix)
X = data.drop(columns=[target])

# Define target vector y
y = data[target]

# Split dataset into training and test sets
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)

files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]

for file_path in files:
    filename = file_path.split("/")[-1]
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"processed_data/{filename}",
        repo_id="Money2277/Engine-Predictive-Maintenance",
        repo_type="dataset",
    )

print("Processed train/test data uploaded to Hugging Face.")
