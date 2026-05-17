from huggingface_hub import snapshot_download
import os

# os.environ["HF_TOKEN"] = ""
snapshot_download(
    repo_id="Qwen/Qwen2.5-14B-Instruct",
    local_dir="./models/Qwen2.5-14B-Instruct",
)