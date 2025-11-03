import os
import argparse
from huggingface_hub import HfApi, upload_file

def main():
    parser = argparse.ArgumentParser(description="Push a dataset to the HuggingFace Hub.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the local dataset directory.")
    parser.add_argument("--repo_id", type=str, required=True, help="Name of the repository on HuggingFace Hub.")
    args = parser.parse_args()

    api = HfApi()
    # Create a repository
    repo_url = api.create_repo(repo_id=args.repo_id, repo_type="dataset", exist_ok=True)
    print(f"Repository created or already exists: {repo_url}")

    # Upload the folder
    upload_file(
        path_or_fileobj=args.dataset_path,
        repo_id=args.repo_id,
        repo_type="dataset",
        path_in_repo=os.path.basename(args.dataset_path)
    )
    print("Dataset uploaded successfully!")

if __name__ == "__main__":
    main()
