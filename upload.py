
import os
from pathlib import Path

from huggingface_hub import HfApi

# Repository root: the directory containing upload.py
ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"

REPO_ID = os.getenv(
    "HF_REPO_ID",
    "Santoshki/mini-llm-alphav1"
)
HF_TOKEN = os.getenv("HF_TOKEN")

# These paths must match the filenames saved by train.py.
files_to_upload = {
    OUTPUT_DIR / "mini_llm_v2.pt": "mini_llm_v2.pt",
    OUTPUT_DIR / "vocab2.json": "vocab2.json",
}

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing.")

# Fail before uploading if either file is missing.
for local_path in files_to_upload:
    if not local_path.is_file():
        raise FileNotFoundError(
            f"Required training output not found: {local_path}"
        )

api = HfApi(token=HF_TOKEN)

for local_path, remote_path in files_to_upload.items():
    print(
        f"Uploading {local_path.name} "
        f"({local_path.stat().st_size:,} bytes)..."
    )

    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message=f"Update {remote_path} after training",
    )

    print(f"Uploaded successfully: {remote_path}")

print(f"Training outputs uploaded to https://huggingface.co/{REPO_ID}")
