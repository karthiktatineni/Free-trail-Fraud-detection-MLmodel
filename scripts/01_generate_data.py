"""
STEP 1 - Synthetic Data Generation
===================================
Generates a realistic free-trial signup event dataset per PRD Section 4:
  - 6,500 genuine single-signup users
  - 700 abuse rings (2-6 accounts each, ~2,448 abuse events)
  - Deliberately imbalanced (~70% genuine / 30% abuse)

Abuse rings rotate identity attributes with realistic probabilities:
  - Payment token: kept 75% of the time (hardest to replace)
  - Device ID: kept 65% of the time
  - IP address: rotated 50% of the time within same /24 subnet
  - Area: kept 85% of the time
  - Name & email: always perturbed (trivially cheap to change)

Output: data/raw/raw_signup_events.csv
"""

import pandas as pd
import numpy as np
import os
import hashlib

# Configuration
SEED = 42
N_GENUINE = 6500
N_RINGS = 700
MIN_RING_SIZE = 2
MAX_RING_SIZE = 6
DATE_START = "2026-01-01"
DATE_END = "2026-06-29"

KEEP_PAYMENT = 0.75
KEEP_DEVICE = 0.65
KEEP_IP = 0.50        # rotates within same /24 subnet
KEEP_AREA = 0.85
GENUINE_MISMATCH_RATE = 0.04  # ~4% of genuine users show geo mismatch (travel, VPN, gift cards)

AREA_TO_COUNTRY = {
    "mumbai": "IN", "delhi": "IN", "bangalore": "IN", "hyderabad": "IN",
    "chennai": "IN", "pune": "IN", "kolkata": "IN", "ahmedabad": "IN",
    "new_york": "US", "san_francisco": "US", "london": "GB",
    "toronto": "CA", "singapore": "SG", "dubai": "AE"
}
ALL_COUNTRIES = list(set(AREA_TO_COUNTRY.values()))

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DATA_DIR, exist_ok=True)

FIRST_NAMES = [
    "aarav","aditi","aisha","akash","amit","ananya","arjun","deepika","gaurav","isha",
    "john","jane","michael","sarah","david","emily","robert","jessica","william","olivia",
    "priya","rahul","ravi","sneha","vikram","neha","karthik","pooja","rajesh","sanjay",
    "mei","wei","yuki","hiroshi","chen","li","kumar","patel","singh","sharma",
    "mohammed","fatima","ali","hassan","omar","layla","james","emma","noah","sophia",
]
LAST_NAMES = [
    "sharma","patel","singh","kumar","nair","gupta","joshi","reddy","mehta","chopra",
    "smith","johnson","williams","brown","jones","garcia","miller","davis","wilson","moore",
    "chen","wang","li","zhang","liu","yang","kim","lee","park","wong",
    "khan","ali","ahmed","hussain","rahman","verma","mishra","pandey","das","iyer",
]
AREAS = [
    "mumbai","delhi","bangalore","hyderabad","chennai","pune",
    "kolkata","ahmedabad","new_york","san_francisco","london",
    "toronto","singapore","dubai"
]
DEVICE_OS_LIST = ["android","ios","windows","macos","linux"]
LEGIT_DOMAINS = ["gmail.com","yahoo.com","outlook.com","hotmail.com","proton.me","company.com"]
DISPOSABLE_DOMAINS = ["mailinator.com","tempmail.com","guerrillamail.com","yopmail.com"]

rng = np.random.default_rng(SEED)


def random_name():
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"

def random_ip():
    return ".".join(str(rng.integers(1, 255)) for _ in range(4))

def subnet_rotate_ip(base_ip):
    """Generate a new IP in the same /24 subnet."""
    parts = base_ip.split(".")
    parts[3] = str(rng.integers(1, 255))
    return ".".join(parts)

def random_device_id():
    return hashlib.md5(rng.bytes(16)).hexdigest()[:16]

def random_payment_token():
    return "pm_" + hashlib.md5(rng.bytes(16)).hexdigest()[:12]

def make_email(name, domain, tag=None):
    local = name.lower().replace(" ", ".")
    noise_roll = rng.random()
    if noise_roll < 0.3:
        local += str(rng.integers(1, 999))
    if tag:
        local += f"+{tag}"
    return f"{local}@{domain}"

def perturb_name(base_name):
    roll = rng.random()
    parts = base_name.split()
    if roll < 0.25 and len(parts) >= 2:
        return f"{parts[0]} {parts[1][0]}."
    elif roll < 0.5:
        return f"{base_name}{rng.integers(1, 99)}"
    elif roll < 0.75 and len(parts) >= 2:
        return ".".join(parts)
    else:
        return base_name.title() if rng.random() < 0.5 else base_name.upper()


# Generate Genuine Users
print("Generating genuine users...")
genuine_records = []
start_ts = pd.Timestamp(DATE_START)
end_ts = pd.Timestamp(DATE_END)
date_range_seconds = int((end_ts - start_ts).total_seconds())

for i in range(N_GENUINE):
    name = random_name()
    domain = rng.choice(LEGIT_DOMAINS)
    area = rng.choice(AREAS)
    ip_country = AREA_TO_COUNTRY[area]
    # ~4% of genuine users show a mismatch (travel, VPN, gift cards, expat bank)
    if rng.random() < GENUINE_MISMATCH_RATE:
        payment_country = rng.choice([c for c in ALL_COUNTRIES if c != ip_country])
    else:
        payment_country = ip_country
    genuine_records.append({
        "user_id": f"u_{i}",
        "signup_time": start_ts + pd.Timedelta(seconds=int(rng.integers(0, date_range_seconds))),
        "name": name,
        "email": make_email(name, domain),
        "email_domain": domain,
        "ip_address": random_ip(),
        "device_id": random_device_id(),
        "device_os": rng.choice(DEVICE_OS_LIST),
        "payment_token": random_payment_token(),
        "area": area,
        "payment_country": payment_country,
        "is_repeat_user": 0,
        "ring_id": -1,
    })

# Generate Abuse Rings
print("Generating abuse rings...")
abuse_records = []
user_counter = N_GENUINE

for ring_idx in range(N_RINGS):
    ring_size = rng.integers(MIN_RING_SIZE, MAX_RING_SIZE + 1)

    base_name = random_name()
    base_ip = random_ip()
    base_device = random_device_id()
    base_payment = random_payment_token()
    base_area = rng.choice(AREAS)
    base_os = rng.choice(DEVICE_OS_LIST)
    # BIN issuing country is tied to the ring leader's home area
    base_payment_country = AREA_TO_COUNTRY[base_area]

    ring_start = start_ts + pd.Timedelta(seconds=int(rng.integers(0, date_range_seconds)))

    for account_idx in range(ring_size):
        time_offset = pd.Timedelta(minutes=int(rng.integers(2, 90) * account_idx))
        signup_time = ring_start + time_offset
        if signup_time > end_ts:
            signup_time = end_ts - pd.Timedelta(minutes=rng.integers(1, 60))

        ip = base_ip if rng.random() < KEEP_IP else subnet_rotate_ip(base_ip)
        device = base_device if rng.random() < KEEP_DEVICE else random_device_id()
        payment = base_payment if rng.random() < KEEP_PAYMENT else random_payment_token()
        area = base_area if rng.random() < KEEP_AREA else rng.choice(AREAS)

        # Payment country follows the payment token:
        # - If keeping base payment → base_payment_country (ring leader's BIN)
        # - If new payment token → country of the current area (new card)
        if payment == base_payment:
            payment_country = base_payment_country
        else:
            payment_country = AREA_TO_COUNTRY[area]

        name = perturb_name(base_name)

        if rng.random() < 0.40:
            domain = rng.choice(DISPOSABLE_DOMAINS)
        else:
            domain = rng.choice(LEGIT_DOMAINS)
        tag = f"trial{account_idx}" if rng.random() < 0.5 else None
        email = make_email(name, domain, tag=tag)

        abuse_records.append({
            "user_id": f"u_{user_counter}",
            "signup_time": signup_time,
            "name": name,
            "email": email,
            "email_domain": domain,
            "ip_address": ip,
            "device_id": device,
            "device_os": base_os if rng.random() < 0.8 else rng.choice(DEVICE_OS_LIST),
            "payment_token": payment,
            "area": area,
            "payment_country": payment_country,
            "is_repeat_user": 1,
            "ring_id": ring_idx,
        })
        user_counter += 1

# Combine and Save
df = pd.DataFrame(genuine_records + abuse_records)
df = df.sort_values("signup_time").reset_index(drop=True)

output_path = os.path.join(RAW_DATA_DIR, "raw_signup_events.csv")
df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print("SYNTHETIC DATASET GENERATED SUCCESSFULLY")
print(f"{'='*60}")
print(f"Total events:        {len(df)}")
print(f"Genuine users:       {(df['is_repeat_user']==0).sum()}")
print(f"Abuse-linked:        {(df['is_repeat_user']==1).sum()}")
print(f"Abuse rings:         {N_RINGS}")
print(f"Class balance:       {(df['is_repeat_user']==0).mean()*100:.1f}% genuine / {(df['is_repeat_user']==1).mean()*100:.1f}% abuse")
print(f"Date range:          {df['signup_time'].min()} to {df['signup_time'].max()}")
print(f"Saved to:            {output_path}")
