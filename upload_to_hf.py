import os
from huggingface_hub import HfApi

# 1. Provide your Hugging Face username and the name you want for your model
username = "HemanthNasaram"  # Change to your actual HF username if different
model_name = "restaurant-sentiment-model"

repo_id = f"{username}/{model_name}"

# 2. Define the path where your locally trained model is saved
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(BASE_DIR, "Sentimental_Analysis_Loads", "custom_llm_model")

print(f"🚀 Preparing to upload model from: {SAVE_DIR}")
print(f"📦 Uploading to Hugging Face Hub: {repo_id}")

try:
    api = HfApi()
    
    # Create the repository if it doesn't exist
    api.create_repo(repo_id=repo_id, exist_ok=True, repo_type="model")
    print("✅ Repository created or found!")

    # Upload the entire folder
    api.upload_folder(
        folder_path=SAVE_DIR,
        repo_id=repo_id,
        repo_type="model",
    )
    
    print(f"🎉 Success! Your model is now live at: https://huggingface.co/{repo_id}")
    print("\nNext steps:")
    print(f"1. Generate an Access Token in your Hugging Face account settings.")
    print(f"2. Add your token to your Render environment variables as 'HF_TOKEN'.")
    print(f"3. Add 'HF_MODEL_URL' to your Render environment variables with the value: https://api-inference.huggingface.co/models/{repo_id}")

except Exception as e:
    print(f"❌ Error uploading to Hugging Face: {e}")
    print("Make sure you are logged in to the huggingface-cli by running: huggingface-cli login")
