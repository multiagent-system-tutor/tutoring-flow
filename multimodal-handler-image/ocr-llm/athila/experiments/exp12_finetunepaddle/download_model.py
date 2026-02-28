import os
import requests
import tarfile
from pathlib import Path

# Config
URL = "https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_train.tar"
DEST_DIR = Path(r"C:\projekdosen\tutoring\PaddleOCR")
TAR_PATH = DEST_DIR / "en_PP-OCRv4_rec_train.tar"
EXTRACT_DIR = DEST_DIR / "en_PP-OCRv4_rec_train"

def download_file(url, dest_path):
    print(f"Downloading from {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    else:
        print(f"Failed to download. Status code: {response.status_code}")
        exit(1)

def extract_tar(tar_path, dest_dir):
    print(f"Extracting {tar_path}...")
    if not tar_path.exists():
        print(f"File {tar_path} not found!")
        return
        
    try:
        with tarfile.open(tar_path, "r") as tar:
            tar.extractall(path=dest_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Error extracting: {e}")

def main():
    if not DEST_DIR.exists():
        print(f"Creating directory {DEST_DIR}...")
        DEST_DIR.mkdir(parents=True, exist_ok=True)

    if not TAR_PATH.exists():
        download_file(URL, TAR_PATH)
    else:
        print(f"{TAR_PATH} already exists. Skipping download.")

    # Extract to DEST_DIR (tar usually contains the directory structure)
    extract_tar(TAR_PATH, DEST_DIR)

    # Check contents
    if EXTRACT_DIR.exists():
        print(f"\nContents of {EXTRACT_DIR}:")
        for f in EXTRACT_DIR.iterdir():
            print(f.name)
    else:
        print(f"Warning: {EXTRACT_DIR} not found after extraction.")

if __name__ == "__main__":
    main()
