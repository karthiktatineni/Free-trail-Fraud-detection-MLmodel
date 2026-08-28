"""
DATASET UPLOADER TO FIREBASE (FIRESTORE & REALTIME DB)
======================================================
Uploads historical raw and processed signup datasets (raw_signup_events.csv,
train_set.csv, test_set.csv) directly into Firebase Firestore using the REST API.
"""

import os
import sys
import json
import urllib.request
import urllib.error
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIREBASE_PROJECT_ID = os.environ.get("FIREBASE_PROJECT_ID", "ml-model-a98ee")


def upload_to_firestore_rest(csv_relative_path: str, collection_name: str, max_records: int = 50):
    """Uploads rows directly to Cloud Firestore collection via REST API."""
    csv_path = os.path.join(DATA_DIR, csv_relative_path)
    if not os.path.exists(csv_path):
        print(f"[Firestore Upload] File not found: {csv_path}")
        return

    print(f"\n[Firestore Upload] Reading {csv_path} ...")
    df = pd.read_csv(csv_path).head(max_records)
    print(f"[Firestore Upload] Uploading {len(df)} documents to Firestore collection '{collection_name}' in project '{FIREBASE_PROJECT_ID}'...")

    success_count = 0
    for idx, row in df.iterrows():
        doc_id = str(row.get("user_id", f"doc_{idx:04d}"))
        url = f"https://firestore.googleapis.com/v1/projects/{FIREBASE_PROJECT_ID}/databases/(default)/documents/{collection_name}?documentId={doc_id}"

        # Convert fields to Firestore REST JSON format
        fields = {}
        for k, v in row.items():
            if pd.isna(v):
                continue
            if isinstance(v, (int, bool)):
                fields[k] = {"integerValue": str(int(v))}
            elif isinstance(v, float):
                fields[k] = {"doubleValue": float(v)}
            else:
                fields[k] = {"stringValue": str(v)}

        payload_bytes = json.dumps({"fields": fields}).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=payload_bytes,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                success_count += 1
        except urllib.error.HTTPError as e:
            if idx == 0:
                print(f"[Firestore Upload] Note on doc {doc_id}: HTTP {e.code} ({e.read().decode('utf-8')[:120]})")
        except Exception as e:
            if idx == 0:
                print(f"[Firestore Upload] Connection note: {e}")

    print(f"[Firestore Upload] Finished collection '{collection_name}' ({success_count}/{len(df)} documents synced).")


def main():
    print(f"=== FIREBASE FIRESTORE DATASET SYNC ({FIREBASE_PROJECT_ID}) ===")
    upload_to_firestore_rest(os.path.join("raw", "raw_signup_events.csv"), "historical_signup_events", max_records=50)
    upload_to_firestore_rest(os.path.join("processed", "test_set.csv"), "benchmark_test_events", max_records=50)


if __name__ == "__main__":
    main()
