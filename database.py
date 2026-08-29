"""
MULTI-TENANT SAAS DATABASE LAYER WITH CRYPTOGRAPHIC KEY HASHING
================================================================
Handles:
  - User Account registration & session validation (Firebase Auth / Local)
  - Cryptographic SHA-256 API Key Generation & Hashed Storage
  - Private Customer scoring history stored per user (users/{uid}/customers)
  - Real-time customer search & duplicate existence check
  - Safe, lock-free SQLite connection management with automatic commit & close
"""

import os
import time
import json
import sqlite3
import secrets
import hashlib
from contextlib import contextmanager
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)
DB_PATH = os.path.join(DATA_DIR, "fraud_saas.db")

# ----------------- FIREBASE INITIALIZATION -----------------
FIREBASE_INITIALIZED = False
firestore_client = None

try:
    import firebase_admin
    from firebase_admin import credentials, firestore

    service_account = os.environ.get("FIREBASE_SERVICE_ACCOUNT_JSON")
    if service_account and os.path.exists(service_account):
        cred = credentials.Certificate(service_account)
        firebase_admin.initialize_app(cred)
        firestore_client = firestore.client()
        FIREBASE_INITIALIZED = True
    elif os.environ.get("FIREBASE_PROJECT_ID"):
        try:
            firebase_admin.initialize_app(options={"projectId": os.environ.get("FIREBASE_PROJECT_ID")})
            firestore_client = firestore.client()
            FIREBASE_INITIALIZED = True
        except Exception:
            pass
except Exception:
    pass


# ----------------- SQLITE CONNECTION CONTEXT MANAGER -----------------
@contextmanager
def db_session():
    """Safe SQLite connection context ensuring immediate transaction handling and no locks."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=60.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def init_db():
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("PRAGMA journal_mode = WAL;")
        cursor.execute("PRAGMA synchronous = NORMAL;")

        # 1. Users Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            uid TEXT PRIMARY KEY,
            email TEXT UNIQUE NOT NULL,
            display_name TEXT,
            created_at TEXT NOT NULL
        );
        """)

        # 2. API Keys Table (Zero Plaintext Storage — strictly SHA-256 key_hash and masked_key)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS api_keys (
            key_hash TEXT PRIMARY KEY,
            key_id TEXT NOT NULL,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            key_type TEXT NOT NULL,
            masked_key TEXT NOT NULL,
            rate_limit_per_min INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 1,
            FOREIGN KEY (user_id) REFERENCES users(uid)
        );
        """)

        # Auto-migrate legacy tables to drop raw_key column if present
        cursor.execute("PRAGMA table_info(api_keys);")
        cols = [row[1] for row in cursor.fetchall()]
        if "raw_key" in cols:
            try:
                cursor.execute("ALTER TABLE api_keys DROP COLUMN raw_key;")
            except Exception:
                pass

        # 3. Tenant Customers Table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            customer_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            ip_address TEXT NOT NULL,
            device_id TEXT NOT NULL,
            payment_token TEXT NOT NULL,
            area TEXT,
            device_os TEXT,
            payment_country TEXT,
            risk_score REAL NOT NULL,
            verdict TEXT NOT NULL,
            recommended_action TEXT NOT NULL,
            severity TEXT,
            model_confidence_pct REAL,
            model_probability REAL,
            signal_breakdown TEXT,
            raw_features TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(uid)
        );
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_keys_user ON api_keys(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_user ON customers(user_id);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_email ON customers(user_id, email);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_ip ON customers(user_id, ip_address);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_customers_payment ON customers(user_id, payment_token);")


init_db()


# ----------------- USER & AUTH METHODS -----------------
def get_or_create_user(uid: str, email: str, display_name: Optional[str] = None) -> Dict[str, Any]:
    """Gets an existing user by UID or email, or registers a new user."""
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    email_clean = (email or "user@enterprise.io").strip().lower()

    with db_session() as conn:
        cursor = conn.cursor()
        # 1. Check by UID first
        cursor.execute("SELECT * FROM users WHERE uid = ?", (uid,))
        row = cursor.fetchone()

        if row:
            return dict(row)

        # 2. Check by Email if UID changed (e.g. across auth providers or tests)
        cursor.execute("SELECT * FROM users WHERE LOWER(email) = ?", (email_clean,))
        row = cursor.fetchone()

        if row:
            cursor.execute("UPDATE users SET uid = ? WHERE LOWER(email) = ?", (uid, email_clean))
            cursor.execute("SELECT * FROM users WHERE uid = ?", (uid,))
            row = cursor.fetchone()
            return dict(row) if row else {"uid": uid, "email": email_clean, "display_name": display_name or email_clean.split("@")[0].capitalize(), "created_at": now_iso}

        # 3. Insert new user
        name = display_name or email_clean.split("@")[0].capitalize()
        cursor.execute(
            "INSERT INTO users (uid, email, display_name, created_at) VALUES (?, ?, ?, ?)",
            (uid, email_clean, name, now_iso)
        )
        user_dict = {"uid": uid, "email": email_clean, "display_name": name, "created_at": now_iso}

        if FIREBASE_INITIALIZED and firestore_client:
            try:
                firestore_client.collection("users").document(uid).set(user_dict)
            except Exception:
                pass

    return user_dict


def get_user_by_id(uid: str) -> Optional[Dict[str, Any]]:
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE uid = ?", (uid,))
        row = cursor.fetchone()
        return dict(row) if row else None


# ----------------- HASHED API KEY METHODS -----------------
def create_user_api_key(
    user_id: str,
    name: str = "Production API Key",
    key_type: str = "live",
    rate_limit_per_min: int = 30
) -> Dict[str, Any]:
    """
    Generates a secure cryptographic random key and stores its SHA-256 hash.
    Format: fk_live_<48-hex-chars>
    """
    prefix = "fk_live_" if key_type == "live" else "fk_test_"
    random_hex = secrets.token_hex(24)
    raw_key = f"{prefix}{random_hex}"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    masked_key = f"{raw_key[:12]}...{raw_key[-4:]}"
    key_id = f"key_{secrets.token_hex(4)}"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rate_limit = int(os.environ.get("DEFAULT_RATE_LIMIT_PER_MINUTE", 30))

    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT INTO api_keys (key_hash, key_id, user_id, name, key_type, masked_key, rate_limit_per_min, created_at, is_active)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
        """, (key_hash, key_id, user_id, name, key_type, masked_key, rate_limit, now_iso))

    key_record = {
        "api_key": raw_key,  # Returned once to creator only; never stored
        "key_hash": key_hash,
        "key_id": key_id,
        "user_id": user_id,
        "name": name,
        "key_type": key_type,
        "masked_key": masked_key,
        "rate_limit_per_min": rate_limit,
        "created_at": now_iso,
        "is_active": 1
    }

    if FIREBASE_INITIALIZED and firestore_client:
        try:
            firestore_client.collection("users").document(user_id).collection("api_keys").document(key_id).set({
                "key_id": key_id,
                "key_hash": key_hash,
                "masked_key": masked_key,
                "name": name,
                "key_type": key_type,
                "rate_limit_per_min": rate_limit,
                "created_at": now_iso,
                "is_active": 1
            })
        except Exception:
            pass

    return key_record


def list_user_api_keys(user_id: str) -> List[Dict[str, Any]]:
    """Returns keys created by this user from SQLite with direct Firestore sync fallback."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT key_hash, key_id, user_id, name, key_type, masked_key, rate_limit_per_min, created_at
        FROM api_keys
        WHERE user_id = ? AND is_active = 1
        ORDER BY created_at DESC
        """, (user_id,))
        rows = cursor.fetchall()
        keys = [dict(r) for r in rows]

    if not keys and FIREBASE_INITIALIZED and firestore_client:
        try:
            docs = firestore_client.collection("users").document(user_id).collection("api_keys").stream()
            fs_keys = []
            for doc in docs:
                data = doc.to_dict()
                if data.get("is_active", 1) == 1:
                    fs_keys.append(data)
                    with db_session() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                        INSERT OR REPLACE INTO api_keys (key_hash, key_id, user_id, name, key_type, masked_key, rate_limit_per_min, created_at, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """, (
                            data.get("key_hash", ""),
                            data.get("key_id", doc.id),
                            user_id,
                            data.get("name", "Production API Key"),
                            data.get("key_type", "live"),
                            data.get("masked_key", "fk_live_..."),
                            data.get("rate_limit_per_min", 30),
                            data.get("created_at", "")
                        ))
            if fs_keys:
                return fs_keys
        except Exception:
            pass

    return keys


def validate_api_key(api_key: str) -> Optional[Dict[str, Any]]:
    """Validates incoming API key by computing its SHA-256 hash, with Firestore cloud fallback."""
    if not api_key:
        return None

    key_hash = hashlib.sha256(api_key.encode("utf-8")).hexdigest()

    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT a.*, u.email as user_email, u.display_name as user_name
        FROM api_keys a
        JOIN users u ON a.user_id = u.uid
        WHERE a.key_hash = ? AND a.is_active = 1
        """, (key_hash,))
        row = cursor.fetchone()
        if row:
            return dict(row)

    if FIREBASE_INITIALIZED and firestore_client:
        try:
            docs = firestore_client.collection_group("api_keys").where("key_hash", "==", key_hash).limit(1).stream()
            for doc in docs:
                data = doc.to_dict()
                if data.get("is_active", 1) == 1:
                    uid = data.get("user_id") or doc.reference.parent.parent.id
                    with db_session() as conn:
                        cursor = conn.cursor()
                        cursor.execute("""
                        INSERT OR REPLACE INTO api_keys (key_hash, key_id, user_id, name, key_type, masked_key, rate_limit_per_min, created_at, is_active)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
                        """, (
                            key_hash,
                            data.get("key_id", doc.id),
                            uid,
                            data.get("name", "Production API Key"),
                            data.get("key_type", "live"),
                            data.get("masked_key", "fk_live_..."),
                            data.get("rate_limit_per_min", 30),
                            data.get("created_at", "")
                        ))
                    return {
                        "key_hash": key_hash,
                        "key_id": data.get("key_id", doc.id),
                        "user_id": uid,
                        "name": data.get("name", "Production API Key"),
                        "key_type": data.get("key_type", "live"),
                        "masked_key": data.get("masked_key", "fk_live_..."),
                        "rate_limit_per_min": data.get("rate_limit_per_min", 30),
                        "is_active": 1
                    }
        except Exception:
            pass

    return None


def revoke_user_api_key(user_id: str, key_id: str) -> bool:
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("UPDATE api_keys SET is_active = 0 WHERE user_id = ? AND key_id = ?", (user_id, key_id))

    if FIREBASE_INITIALIZED and firestore_client:
        try:
            firestore_client.collection("users").document(user_id).collection("api_keys").document(key_id).set({"is_active": 0}, merge=True)
        except Exception:
            pass

    return True


def delete_user_api_key(user_id: str, key_id: str) -> bool:
    """Permanently deletes an API key from database and Firestore."""
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("DELETE FROM api_keys WHERE user_id = ? AND key_id = ?", (user_id, key_id))

    if FIREBASE_INITIALIZED and firestore_client:
        try:
            firestore_client.collection("users").document(user_id).collection("api_keys").document(key_id).delete()
        except Exception as e:
            print(f"[DB] Firestore delete key note: {e}")

    return True


# ----------------- CUSTOMER STORAGE & MULTI-TENANT SEARCH -----------------
def record_customer_signup(user_id: str, event_data: Dict[str, Any], score_result: Dict[str, Any]) -> str:
    cust_id = event_data.get("user_id") or f"cust_{secrets.token_hex(6)}"
    now_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    name = str(event_data.get("name", "")).strip()
    email = str(event_data.get("email", "")).strip().lower()
    ip = str(event_data.get("ip_address", "")).strip()
    device = str(event_data.get("device_id", "")).strip()
    payment = str(event_data.get("payment_token", "")).strip()
    area = str(event_data.get("area", "")).strip().lower()
    os_name = str(event_data.get("device_os", "")).strip().lower()
    payment_country = str(event_data.get("payment_country", "")).strip().upper()

    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO customers (
            customer_id, user_id, name, email, ip_address, device_id, payment_token,
            area, device_os, payment_country, risk_score, verdict, recommended_action,
            severity, model_confidence_pct, model_probability, signal_breakdown, raw_features, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            cust_id, user_id, name, email, ip, device, payment,
            area, os_name, payment_country,
            float(score_result["risk_score"]),
            score_result["verdict"],
            score_result["recommended_action"],
            score_result.get("severity", "low"),
            float(score_result.get("model_confidence_pct", 0.0)),
            float(score_result.get("model_probability", 0.0) or 0.0),
            json.dumps(score_result.get("signal_breakdown", {})),
            json.dumps(score_result.get("raw_features", {})),
            now_iso
        ))

    if FIREBASE_INITIALIZED and firestore_client:
        try:
            firestore_client.collection("users").document(user_id).collection("customers").document(cust_id).set({
                "customer_id": cust_id,
                "name": name,
                "email": email,
                "ip_address": ip,
                "device_id": device,
                "payment_token": payment,
                "area": area,
                "risk_score": float(score_result["risk_score"]),
                "verdict": score_result["verdict"],
                "recommended_action": score_result["recommended_action"],
                "created_at": now_iso
            })
        except Exception:
            pass

    return cust_id


def list_user_customers(user_id: str, limit: int = 50) -> List[Dict[str, Any]]:
    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT customer_id, name, email, ip_address, device_id, payment_token, area,
               risk_score, verdict, recommended_action, severity, created_at, signal_breakdown
        FROM customers
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
        """, (user_id, limit))
        rows = cursor.fetchall()

    out = []
    for r in rows:
        d = dict(r)
        try:
            d["signal_breakdown"] = json.loads(d["signal_breakdown"])
        except Exception:
            d["signal_breakdown"] = {}
        out.append(d)
    return out


def search_user_customer(user_id: str, query: str) -> Dict[str, Any]:
    q = f"%{query.strip().lower()}%"

    with db_session() as conn:
        cursor = conn.cursor()
        cursor.execute("""
        SELECT *
        FROM customers
        WHERE user_id = ? AND (
            LOWER(email) LIKE ? OR
            LOWER(name) LIKE ? OR
            ip_address LIKE ? OR
            payment_token LIKE ? OR
            device_id LIKE ?
        )
        ORDER BY created_at DESC
        LIMIT 20
        """, (user_id, q, q, q, q, q))
        rows = cursor.fetchall()

    matches = []
    for r in rows:
        d = dict(r)
        try:
            d["signal_breakdown"] = json.loads(d["signal_breakdown"])
            d["raw_features"] = json.loads(d["raw_features"])
        except Exception:
            pass
        matches.append(d)

    return {
        "exists": len(matches) > 0,
        "match_count": len(matches),
        "customers": matches
    }


def push_initial_dataset_to_firebase(batch_limit: int = 200) -> Dict[str, Any]:
    if not FIREBASE_INITIALIZED or not firestore_client:
        return {"status": "skipped", "message": "Firebase Firestore not connected."}

    raw_csv = os.path.join(DATA_DIR, "raw", "raw_signup_events.csv")
    if not os.path.exists(raw_csv):
        return {"status": "error", "message": "raw_signup_events.csv missing"}

    import pandas as pd
    df = pd.read_csv(raw_csv).head(batch_limit)
    batch = firestore_client.batch()
    col_ref = firestore_client.collection("historical_signup_events")

    count = 0
    for _, row in df.iterrows():
        doc_ref = col_ref.document(str(row["user_id"]))
        batch.set(doc_ref, {
            "user_id": str(row["user_id"]),
            "name": str(row["name"]),
            "email": str(row["email"]),
            "ip_address": str(row["ip_address"]),
            "device_id": str(row["device_id"]),
            "payment_token": str(row["payment_token"]),
            "area": str(row["area"]),
            "is_repeat_user": int(row["is_repeat_user"]),
            "signup_time": str(row["signup_time"])
        })
        count += 1

    batch.commit()
    return {"status": "success", "synced_records": count}


def load_all_production_customers() -> List[Dict[str, Any]]:
    """
    Load all previously scored customers from Firestore (persistent) or SQLite (fallback).
    Used to rebuild engine velocity counters on server restart so the model
    remembers past signups and detects repeat abuse across Render cold-starts.
    """
    records = []

    # --- Try Firestore first (survives Render restarts) ---
    if FIREBASE_INITIALIZED and firestore_client:
        try:
            all_users_ref = firestore_client.collection("users").stream()
            for user_doc in all_users_ref:
                uid = user_doc.id
                try:
                    cust_docs = firestore_client.collection("users").document(uid).collection("customers").stream()
                    for doc in cust_docs:
                        data = doc.to_dict()
                        if data and data.get("payment_token"):
                            records.append({
                                "name": data.get("name", ""),
                                "email": data.get("email", ""),
                                "ip_address": data.get("ip_address", ""),
                                "device_id": data.get("device_id", ""),
                                "payment_token": data.get("payment_token", ""),
                                "area": data.get("area", ""),
                                "created_at": data.get("created_at", ""),
                            })
                except Exception:
                    continue
            if records:
                print(f"[DB] Loaded {len(records)} production customers from Firestore")
                return records
        except Exception as e:
            print(f"[DB] Firestore customer load failed: {e}")

    # --- Fallback to SQLite (ephemeral on Render but works locally) ---
    try:
        with db_session() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            SELECT name, email, ip_address, device_id, payment_token, area, created_at
            FROM customers
            ORDER BY created_at ASC
            """)
            rows = cursor.fetchall()
            for r in rows:
                records.append(dict(r))
        if records:
            print(f"[DB] Loaded {len(records)} production customers from SQLite")
    except Exception as e:
        print(f"[DB] SQLite customer load failed: {e}")

    return records

