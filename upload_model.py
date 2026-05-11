import os
import socket
from huggingface_hub import HfApi

# NETWORK PATCH: Forces Python to use IPv4 only.
# This bypasses 'Connection Reset' errors common on some Windows network configurations
# when connecting to Hugging Face servers via IPv6.
orig_getaddrinfo = socket.getaddrinfo
def patched_getaddrinfo(*args, **kwargs):
    responses = orig_getaddrinfo(*args, **kwargs)
    return [res for res in responses if res[0] == socket.AF_INET]
socket.getaddrinfo = patched_getaddrinfo

def upload_model():
    """
    Robust script to upload the final GGUF model to the Hugging Face Hub.
    Uses token-based authentication directly to bypass system-level login issues.
    """
    
    print("Welcome to the Robust Model Uploader (Forcing IPv4)")
    token = input("Enter your Hugging Face Token: ").strip()
    
    if not token:
        print("No token provided. Exiting.")
        return

    # Initialize the API with the token
    api = HfApi(token=token)

    try:
        # Verify connection and get user details
        print("Verifying connection to Hugging Face...")
        user_info = api.whoami()
        username = user_info['name']
        print(f"Verified account: {username}")
        
        # Create or verify the model repository
        repo_name = f"{username}/qwen-cs-researcher-0.5B"
        print(f"Checking repository: {repo_name}")
        api.create_repo(repo_id=repo_name, repo_type="model", exist_ok=True)

        # Look for the GGUF file in the current directory
        gguf_file = "qwen-resercher.gguf"
        if os.path.exists(gguf_file):
            print(f"Uploading {gguf_file} to {repo_name}...")
            # Upload the large file to the root of the repository
            api.upload_file(
                path_or_fileobj=gguf_file,
                path_in_repo=gguf_file,
                repo_id=repo_name,
                repo_type="model"
            )
            print("Upload complete. The model is now public.")
            print(f"Visit: https://huggingface.co/{repo_name}")
        else:
            print(f"File {gguf_file} not found. Please ensure the conversion script finished successfully.")

    except Exception as e:
        print(f"An error occurred during upload: {e}")
        print("If connection issues persist, please use the manual upload option in the browser.")

if __name__ == "__main__":
    upload_model()
