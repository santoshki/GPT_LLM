import os

from huggingface_hub import HfApi, upload_folder

repo_id = os.getenv("HF_REPO_ID", "Santoshki/mini-llm-alphav1")
token = os.getenv("HF_TOKEN")

if not token:
    raise RuntimeError(
        "HF_TOKEN is not set. Add the Hugging Face access token as a GitHub secret."
    )

api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

upload_folder(
    folder_path="outputs",
    repo_id=repo_id,
    repo_type="model",
    token=token,
    commit_message="Upload trained SimpleGPT model"
)

print(f"Upload complete to {repo_id}!")