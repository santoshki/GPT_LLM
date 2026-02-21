from huggingface_hub import upload_folder

upload_folder(
    folder_path=".",
    repo_id="Santoshki/mini-llm-alphav1",
    repo_type="model"
)

print("Upload complete!")