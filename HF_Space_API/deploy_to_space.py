import os
from huggingface_hub import HfApi

# Put your write-access token here (ensure it has 'write' permissions to your spaces)
HF_TOKEN = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

# Your space repository ID
REPO_ID = "HemanthNasaram/my-sentiment-api"
REPO_TYPE = "space"

# The folder where your Space files are located
FOLDER_TO_UPLOAD = os.path.dirname(os.path.abspath(__file__))

def deploy_space():
    api = HfApi(token=HF_TOKEN)
    
    print(f"Deploying files from {FOLDER_TO_UPLOAD} to Space: {REPO_ID}...")
    
    # Files to upload
    files_to_upload = ["app.py", "requirements.txt", "Dockerfile"]
    
    for filename in files_to_upload:
        file_path = os.path.join(FOLDER_TO_UPLOAD, filename)
        if os.path.exists(file_path):
            print(f"Uploading {filename}...")
            api.upload_file(
                path_or_fileobj=file_path,
                path_in_repo=filename,
                repo_id=REPO_ID,
                repo_type=REPO_TYPE,
                commit_message=f"Upload {filename} via auto-deploy script"
            )
        else:
            print(f"Warning: {filename} not found!")

    print("\n✅ Deployment successful! Your Space should start building now.")
    print(f"Watch the build progress here: https://huggingface.co/spaces/{REPO_ID}")
    
if __name__ == "__main__":
    deploy_space()
