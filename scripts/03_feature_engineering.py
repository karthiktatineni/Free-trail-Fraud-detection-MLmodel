"""
STEP 3 - ENHANCED Feature Engineering
======================================
Implements the full causal feature set per PRD Section 5 + Section 9:
  1. Entity-Graph Linkage: connected-component size via causal union-find
  2. BIN-Country Geo Mismatch: payment country vs IP country
  3. Rolling Velocity: 24h & 1h sliding windows per entity
  4. Time-of-Day Risk: odd-hour signups (0-5am)
  5. Meta-Reuse Count: distinct identity families reused (0-4)

Input:  data/raw/raw_signup_events.csv
Output: data/processed/features_v2.csv
        data/processed/full_dataset_with_features.csv
        data/processed/full_dataset_with_features_v2.csv
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DATA_RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "raw_signup_events.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

df = pd.read_csv(DATA_RAW_PATH, parse_dates=["signup_time"])
df = df.sort_values("signup_time").reset_index(drop=True)
print(f"Loaded {len(df)} events from {DATA_RAW_PATH}")

disposable_domains = {
    "mailinator.com", "tempmail.com", "10minutemail.com",
    "guerrillamail.com", "yopmail.com"
}
free_domains = {"gmail.com", "yahoo.com", "outlook.com", "hotmail.com"}

area_to_country = {
    "mumbai": "IN", "delhi": "IN", "bangalore": "IN", "hyderabad": "IN",
    "chennai": "IN", "pune": "IN", "kolkata": "IN", "ahmedabad": "IN",
    "new_york": "US", "san_francisco": "US", "london": "GB",
    "toronto": "CA", "singapore": "SG", "dubai": "AE"
}

rng = np.random.default_rng(7)
df["ip_country"] = df["area"].map(area_to_country)

# payment_country is now in the raw data (generated label-independently in 01_generate_data.py)
# Simply compute the mismatch from existing columns — no label lookup
df["payment_ip_country_mismatch"] = (df["ip_country"] != df["payment_country"]).astype(int)


def ip_subnet(ip):
    return ".".join(ip.split(".")[:3])

df["ip_subnet"] = df["ip_address"].apply(ip_subnet)
df["name_norm"] = df["name"].str.lower().str.replace(r"[^a-z ]", "", regex=True).str.strip()
df["signup_hour"] = df["signup_time"].dt.hour

# Incremental Union-Find
parent = {}
size = {}

def find(x):
    parent.setdefault(x, x)
    size.setdefault(x, 1)
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

def union(a, b):
    ra, rb = find(a), find(b)
    if ra == rb:
        return
    if size[ra] < size[rb]:
        ra, rb = rb, ra
    parent[rb] = ra
    size[ra] += size[rb]

def component_size(x):
    return size[find(x)]

# Running State Dictionaries
seen_payment, seen_ip, seen_subnet, seen_device, seen_name = {}, {}, {}, {}, {}
hour_bucket_device = {}
window_24h_payment = {}
window_24h_device = {}
window_24h_ip_subnet = {}

print("Computing causal features...")
feat_rows = []
total = len(df)

for idx, row in df.iterrows():
    if idx % 2000 == 0:
        print(f"  Processing row {idx}/{total} ({idx/total*100:.0f}%)")

    pay = row.payment_token
    ip = row.ip_address
    subnet = row.ip_subnet
    dev = row.device_id
    name = row.name_norm
    t = row.signup_time
    hour_bucket = t.floor("h")

    payment_reuse_count = seen_payment.get(pay, 0)
    ip_reuse_count = seen_ip.get(ip, 0)
    subnet_reuse_count = seen_subnet.get(subnet, 0)
    device_reuse_count = seen_device.get(dev, 0)

    def count_and_prune(window_dict, key, now):
        lst = window_dict.get(key, [])
        lst = [ts for ts in lst if (now - ts).total_seconds() <= 24 * 3600]
        cnt = len(lst)
        window_dict[key] = lst
        return cnt

    payment_signups_last_24h = count_and_prune(window_24h_payment, pay, t)
    device_signups_last_24h = count_and_prune(window_24h_device, dev, t)
    subnet_signups_last_24h = count_and_prune(window_24h_ip_subnet, subnet, t)

    best_sim = 0.0
    if seen_name:
        candidates = list(seen_name.keys())[-300:]
        for cand in candidates:
            s = SequenceMatcher(None, name, cand).ratio()
            if s > best_sim:
                best_sim = s
            if best_sim > 0.97:
                break

    key = (dev, hour_bucket)
    device_signups_last_hour = hour_bucket_device.get(key, 0)

    email_domain = row.email_domain
    is_disposable_domain = int(email_domain in disposable_domains)
    is_free_domain = int(email_domain in free_domains)
    email_local = row.email.split("@")[0]
    email_local_has_digits = int(any(ch.isdigit() for ch in email_local))
    email_local_has_plus_tag = int("+" in email_local)

    geo_mismatch = int(row.ip_country != row.payment_country)

    graph_component_size = max(
        component_size(pay),
        component_size(dev),
        component_size(subnet)
    )

    attrs_reused_count = (
        int(payment_reuse_count > 0) +
        int(subnet_reuse_count > 0) +
        int(device_reuse_count > 0) +
        int(best_sim > 0.85)
    )

    is_odd_hour = int(row.signup_hour in [0, 1, 2, 3, 4, 5])

    feat_rows.append(dict(
        payment_reuse_count=payment_reuse_count,
        ip_reuse_count=ip_reuse_count,
        ip_subnet_reuse_count=subnet_reuse_count,
        device_reuse_count=device_reuse_count,
        device_signups_last_hour=device_signups_last_hour,
        payment_signups_last_24h=payment_signups_last_24h,
        device_signups_last_24h=device_signups_last_24h,
        subnet_signups_last_24h=subnet_signups_last_24h,
        name_similarity_score=round(best_sim, 3),
        is_disposable_email_domain=is_disposable_domain,
        is_free_email_domain=is_free_domain,
        email_local_has_digits=email_local_has_digits,
        email_local_has_plus_tag=email_local_has_plus_tag,
        payment_ip_country_mismatch=geo_mismatch,
        graph_component_size=graph_component_size,
        attrs_reused_count=attrs_reused_count,
        signup_hour=row.signup_hour,
        is_odd_hour=is_odd_hour,
    ))

    seen_payment[pay] = payment_reuse_count + 1
    seen_ip[ip] = ip_reuse_count + 1
    seen_subnet[subnet] = subnet_reuse_count + 1
    seen_device[dev] = device_reuse_count + 1
    seen_name[name] = seen_name.get(name, 0) + 1
    hour_bucket_device[key] = device_signups_last_hour + 1
    window_24h_payment.setdefault(pay, []).append(t)
    window_24h_device.setdefault(dev, []).append(t)
    window_24h_ip_subnet.setdefault(subnet, []).append(t)

    union(pay, dev)
    union(dev, subnet)

feats = pd.DataFrame(feat_rows)
full = pd.concat([df.reset_index(drop=True), feats], axis=1)

full["area_freq"] = full["area"].map(full["area"].value_counts(normalize=True))
full["device_os_freq"] = full["device_os"].map(full["device_os"].value_counts(normalize=True))

FEATURE_COLS = [
    "payment_reuse_count", "ip_reuse_count", "ip_subnet_reuse_count",
    "device_reuse_count", "device_signups_last_hour",
    "payment_signups_last_24h", "device_signups_last_24h", "subnet_signups_last_24h",
    "name_similarity_score",
    "is_disposable_email_domain", "is_free_email_domain",
    "email_local_has_digits", "email_local_has_plus_tag",
    "payment_ip_country_mismatch",
    "graph_component_size", "attrs_reused_count",
    "signup_hour", "is_odd_hour",
    "area_freq", "device_os_freq",
]
TARGET_COL = "is_repeat_user"

features_path = os.path.join(PROCESSED_DIR, "features_v2.csv")
full_path = os.path.join(PROCESSED_DIR, "full_dataset_with_features.csv")
full_v2_path = os.path.join(PROCESSED_DIR, "full_dataset_with_features_v2.csv")

full[["user_id"] + FEATURE_COLS + [TARGET_COL]].to_csv(features_path, index=False)
full.to_csv(full_path, index=False)
full.to_csv(full_v2_path, index=False)

print(f"\n{'='*60}")
print("FEATURE ENGINEERING COMPLETE")
print(f"{'='*60}")
print(f"Saved {features_path} with {len(FEATURE_COLS)} features")
print(f"Dataset shape: {full.shape}")
print(f"\nTop correlation with target (is_repeat_user):")
corr = full[FEATURE_COLS + [TARGET_COL]].corr()[TARGET_COL].sort_values(ascending=False)
print(corr.to_string())
