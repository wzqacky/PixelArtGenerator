import os
import argparse
from huggingface_hub import HfApi, upload_file, upload_folder

def main():
    parser = argparse.ArgumentParser(description="Push a file or folder to the HuggingFace Hub.")
    parser.add_argument("--path", type=str, required=True, help="Path to the local file or directory.")
    parser.add_argument("--repo_id", type=str, required=True, help="Name of the repository on HuggingFace Hub.")
    parser.add_argument("--repo_type", type=str, required=True, choices=["dataset", "model"], help="Type of the repository.")
    args = parser.parse_args()

    api = HfApi()
    # Create a repository
    repo_url = api.create_repo(repo_id=args.repo_id, repo_type=args.repo_type, exist_ok=True)
    print(f"Repository created or already exists: {repo_url}")

    if os.path.isdir(args.path):
        # Upload the folder
        upload_folder(
            folder_path=args.path,
            repo_id=args.repo_id,
            repo_type=args.repo_type
        )
        print("Folder uploaded successfully!")
    elif os.path.isfile(args.path):
        # Upload the file
        upload_file(
            path_or_fileobj=args.path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            path_in_repo=os.path.basename(args.path)
        )
        print("File uploaded successfully!")
    else:
        print(f"Error: Path '{args.path}' does not exist or is not a file or directory.")
        return

if __name__ == "__main__":
    main()
