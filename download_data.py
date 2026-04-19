"""
CMAPSS Dataset Downloader
NASA Prognostics Center of Excellence — Real turbofan engine degradation data.

Run this script BEFORE the notebook:
    python download_data.py
"""

import os
import sys
import zipfile
import urllib.request

DATA_DIR = "data"
REQUIRED_FILES = [
    "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
    "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
    "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
    "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
]

# Direct NASA download URL (CMAPSSData.zip)
NASA_URL = "https://ti.arc.nasa.gov/c/6/"


def check_data_exists():
    return all(os.path.exists(os.path.join(DATA_DIR, f)) for f in REQUIRED_FILES)


def try_nasa_download():
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, "CMAPSSData.zip")
    print("Attempting download from NASA Prognostics Repository...")
    try:
        urllib.request.urlretrieve(NASA_URL, zip_path)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(DATA_DIR)
        os.remove(zip_path)
        print("NASA download successful.")
        return True
    except Exception as e:
        print(f"NASA direct download failed: {e}")
        return False


def try_kaggle_download():
    """
    Kaggle dataset: behrad3d/nasa-cmaps
    Requires: kaggle CLI installed and ~/.kaggle/kaggle.json configured
    Install: pip install kaggle
    Setup:   https://www.kaggle.com/docs/api
    """
    try:
        import subprocess
        print("Attempting Kaggle download...")
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", "behrad3d/nasa-cmaps",
             "-p", DATA_DIR, "--unzip"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            print("Kaggle download successful.")
            return True
        else:
            print(f"Kaggle download failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("Kaggle CLI not installed.")
        return False


def print_manual_instructions():
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD INSTRUCTIONS")
    print("="*60)
    print("\nOption A — NASA Direct (recommended):")
    print("  1. Go to: https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/")
    print("  2. Find 'Turbofan Engine Degradation Simulation Dataset'")
    print("  3. Download CMAPSSData.zip")
    print(f"  4. Unzip into '{os.path.abspath(DATA_DIR)}/' folder")
    print("\nOption B — Kaggle:")
    print("  1. https://www.kaggle.com/datasets/behrad3d/nasa-cmaps")
    print("  2. Download and unzip into data/ folder")
    print("\nRequired files after unzip:")
    for f in REQUIRED_FILES:
        print(f"  data/{f}")
    print("="*60)


if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)

    if check_data_exists():
        print("All CMAPSS data files found. Ready to run experiments.")
        sys.exit(0)

    print("CMAPSS data not found. Attempting download...")

    if try_nasa_download() and check_data_exists():
        print("Data ready.")
        sys.exit(0)

    if try_kaggle_download() and check_data_exists():
        print("Data ready.")
        sys.exit(0)

    print_manual_instructions()
    sys.exit(1)
