"""
Quick-start script: download MASAD from Google Drive and extract it.
Usage: python scripts/download_masad.py

Requires: pip install gdown
"""

import subprocess
import sys
from pathlib import Path
import zipfile

GDRIVE_ID = "19YJ8vEYCb-uEKUqSGFmysUTvNzxhVKFE"
OUT_DIR = Path("data/raw")
ZIP_PATH = OUT_DIR / "masad.zip"


def download():
    import urllib.request
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try direct URL with confirmation first (fast and reliable without extra dependencies)
    url = f"https://drive.usercontent.google.com/download?id={GDRIVE_ID}&confirm=t"
    print(f"[INFO] Downloading MASAD archive from Google Drive (2.4 GB)...")
    print(f"Target path: {ZIP_PATH}")
    
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as resp, open(ZIP_PATH, "wb") as out_file:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            block_size = 1024 * 1024  # 1MB blocks
            while True:
                buffer = resp.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                if total > 0:
                    percent = downloaded / total * 100
                    print(f"\rProgress: {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({percent:.1f}%)", end="", flush=True)
                else:
                    print(f"\rDownloaded: {downloaded / (1024*1024):.1f} MB", end="", flush=True)
            print()
        print(f"[INFO] Download completed successfully: {ZIP_PATH}")
    except Exception as e:
        print(f"\n[WARNING] Direct download encountered error: {e}. Falling back to gdown...")
        try:
            import gdown
        except ImportError:
            print("[INFO] Installing gdown…")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "gdown", "-q"])
            import gdown
        gdown_url = f"https://drive.google.com/uc?id={GDRIVE_ID}"
        gdown.download(gdown_url, str(ZIP_PATH), quiet=False)
        print(f"[INFO] Downloaded to {ZIP_PATH}")


def extract():
    print(f"[INFO] Extracting {ZIP_PATH} → {OUT_DIR}/")
    with zipfile.ZipFile(ZIP_PATH, "r") as zf:
        zf.extractall(OUT_DIR)
    ZIP_PATH.unlink()   # Remove zip to save space
    print(f"[INFO] Extraction complete. Contents of {OUT_DIR}:")
    for p in sorted(OUT_DIR.rglob("*"))[:30]:
        print(f"  {p}")


if __name__ == "__main__":
    download()
    extract()
    print("\n[DONE] MASAD is ready in data/raw/")
    print("Next step: python train.py --mode all")
