from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOKEN_"))
api.upload_folder(
    folder_path="PredictiveMaintenance_project/deployment",     # the local folder containing your files
    repo_id="Money2277/Engine-Predictive-Maintenance",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
