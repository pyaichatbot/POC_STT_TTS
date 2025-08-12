"""
Download Whisper Small model weights from Hugging Face to a local directory for offline use.
Usage:
    python download_whisper_small.py
"""
from huggingface_hub import snapshot_download
import os

def main():
    target_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "models/whisper-small"))
    print(f"Downloading 'openai/whisper-small' to {target_dir} ...")
    snapshot_download(
        repo_id="openai/whisper-small",
        repo_type="model",
        local_dir=target_dir,
        local_dir_use_symlinks=False
    )
    print("Download complete.")

if __name__ == "__main__":
    main()

#python download_whisper_small.py


# 1. Set Proxy Environment Variables
# In your terminal, set the proxy variables (replace with your actual proxy address):

# 2. Use huggingface-cli with Proxy
# Install the CLI if needed:
# pip install huggingface_hub
# Then download the model
# huggingface-cli download --repo-type model openai/whisper-small --local-dir ./models/whisper-small --local-dir-use-symlinks False