from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
from huggingface_hub import HfApi, create_repo
from pathlib import Path
import os

repo_id = "Money2277/Engine-Predictive-Maintenance"
repo_type = "dataset"

# get the path for Data folder
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN_"))

# Step 1: Check if the space exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Space '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Space '{repo_id}' not found. Creating new space...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Space '{repo_id}' created.")

api.upload_folder(
    folder_path=str(DATA_DIR),
    repo_id=repo_id,
    repo_type=repo_type,
)
