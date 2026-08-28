"""
FIREBASE STORAGE DATASET UPLOADER
=================================
Uploads ML training, raw, and feature datasets directly to Firebase Storage bucket:
  Destination: dataset/<filename>
  Bucket: ml-model-a98ee.firebasestorage.app / ml-model-a98ee.appspot.com
"""

import os
import mimetypes
import urllib.request
import urllib.error
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")

BUCKET = os.environ.get("FIREBASE_STORAGE_BUCKET", "ml-model-a98ee.firebasestorage.app")
# Strip gs:// or trailing slashes if present
BUCKET = BUCKET.replace("gs://", "").strip("/")
API_KEY = os.environ.get("FIREBASE_API_KEY", "")

DATASET_FILES = [
    os.path.join(DATA_DIR, "raw", "raw_signup_events.csv"),
    os.path.join(DATA_DIR, "processed", "full_dataset_with_features.csv"),
    os.path.join(DATA_DIR, "processed", "train_set.csv"),
    os.path.join(DATA_DIR, "processed", "val_set.csv"),
    os.path.join(DATA_DIR, "processed", "test_set.csv"),
    os.path.join(DATA_DIR, "processed", "features_v2.csv"),
    os.path.join(DATA_DIR, "processed", "scored_dataset.csv"),
]

def upload_file_to_firebase_storage(file_path: str, storage_path: str) -> bool:
    if not os.path.exists(file_path):
        print(f"[SKIP] Local file not found: {file_path}")
        return False

    encoded_name = urllib.parse.quote(storage_path, safe="")
    url = f"https://firebasestorage.googleapis.com/v0/b/{BUCKET}/o?uploadType=media&name={encoded_name}"
    if API_KEY:
        url += f"&key={API_KEY}"

    with open(file_path, "rb") as f:
        data = f.read()

    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "text/csv",
            "Content-Length": str(len(data))
        },
        method="POST"
    )

    try:
        with urllib.request.urlopen(req) as resp:
            status = resp.getcode()
            print(f"[SUCCESS] Uploaded {os.path.basename(file_path)} -> gs://{BUCKET}/{storage_path} (HTTP {status})")
            return True
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="ignore")
        print(f"[HTTP {e.code}] Failed to upload {os.path.basename(file_path)}: {body}")
        # Try appspot.com variant if firebasestorage.app failed
        if "appspot.com" not in BUCKET:
            alt_bucket = "ml-model-a98ee.appspot.com"
            alt_url = f"https://firebasestorage.googleapis.com/v0/b/{alt_bucket}/o?uploadType=media&name={encoded_name}"
            if API_KEY:
                alt_url += f"&key={API_KEY}"
            try:
                alt_req = urllib.request.Request(alt_url, data=data, headers={"Content-Type": "text/csv"}, method="POST")
                with urllib.request.urlopen(alt_req) as alt_resp:
                    print(f"[SUCCESS] Uploaded to alt bucket {os.path.basename(file_path)} -> gs://{alt_bucket}/{storage_path}")
                    return True
            except Exception as alt_err:
                print(f"[ALT ERROR] {alt_err}")
        return False
    except Exception as ex:
        print(f"[ERROR] Exception during upload of {os.path.basename(file_path)}: {ex}")
        return False

def main():
    print("=" * 60)
    print(f"UPLOADING DATASETS TO FIREBASE STORAGE: {BUCKET}")
    print("=" * 60)

    success_count = 0
    for fpath in DATASET_FILES:
        fname = os.path.basename(fpath)
        dest = f"dataset/{fname}"
        if upload_file_to_firebase_storage(fpath, dest):
            success_count += 1

    print(f"\nCompleted: {success_count}/{len(DATASET_FILES)} datasets uploaded to Firebase Storage.")

if __name__ == "__main__":
    main()
