from huggingface_hub import login, upload_folder

# (optional) Login with your Hugging Face credentials
login()

# Push your model files
upload_folder(folder_path="C:/Users/adebo/OneDrive/Desktop/FNOL PROJECT/models", repo_id="Bunmi01/Ultimate_claim_cost_model", repo_type="model")
