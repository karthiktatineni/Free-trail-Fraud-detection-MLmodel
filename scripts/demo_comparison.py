"""
Live Model Demonstration on Genuine vs Fraud Signups
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import FraudRiskEngine

engine = FraudRiskEngine(warm_start=True)

# 1. Genuine / Legitimate User Signup (Brand new user, organic Gmail, unique card/device, London)
genuine_event = {
    "user_id": "demo_genuine_user_01",
    "name": "David Smith",
    "email": "david.smith@gmail.com",
    "ip_address": "203.0.113.45",
    "device_id": "dev_unique_macbook_pro",
    "payment_token": "pm_unique_barclays_card",
    "area": "london",
    "device_os": "macos",
    "payment_country": "GB",
    "signup_time": "2026-07-15 14:30:00"
}

# 2. Fraud / Syndicate Abuse User (Reused card + shared /24 subnet + disposable email + tag)
fraud_event = {
    "user_id": "demo_fraud_syndicate_03",
    "name": "Sanjay Nair 2",
    "email": "sanjay.nair+trial4@mailinator.com",
    "ip_address": "39.173.180.190",
    "device_id": "f21faa72fe17c06d",
    "payment_token": "pm_424776171fe7",
    "area": "ahmedabad",
    "device_os": "android",
    "payment_country": "IN",
    "signup_time": "2026-07-15 02:45:00"
}

print("\n" + "="*80)
print("              PART 1: SIDE-BY-SIDE SCENARIO COMPARISON")
print("="*80)

for title, event in [("SCENARIO 1: GENUINE (NON-FRAUD) SIGNUP", genuine_event),
                     ("SCENARIO 2: COORDINATED FRAUD / MULTI-ACCOUNTING ATTACK", fraud_event)]:
    res = engine.score_event(event, update_state=False)
    print(f"\n>>> {title}")
    print("-" * 80)
    print(f"  User Name / ID       : {event['name']} ({event['user_id']})")
    print(f"  Email Address        : {event['email']}")
    print(f"  IP & Subnet Prefix   : {event['ip_address']} (/24: {'.'.join(event['ip_address'].split('.')[:3])})")
    print(f"  Payment Token        : {event['payment_token']}")
    print(f"  Device ID            : {event['device_id']}")
    print(f"  ----------------------------------------------------------------------")
    print(f"  RISK SCORE           : {res['risk_score']} / 100")
    print(f"  DECISION VERDICT     : {res['verdict']}")
    print(f"  RECOMMENDED ACTION   : {res['recommended_action']}")
    print(f"  MODEL CONFIDENCE     : {res['model_confidence_pct']}%")
    print(f"  CONNECTED GRAPH SIZE : {res['raw_features']['graph_component_size']} linked identity nodes")
    print(f"  ----------------------------------------------------------------------")
    print("  TOP CONTRIBUTING SIGNALS:")
    for sig, weight in list(res["signal_breakdown"].items())[:5]:
        raw_val = res["raw_features"].get(sig, "--")
        print(f"    * {sig:<28}: contribution = {weight:>6.2f}  | raw_value = {raw_val}")

print("\n" + "="*80)
print("              PART 2: EVASION ATTACK SEQUENCE (GRAPH LINKAGE IN ACTION)")
print("="*80)

# Simulate a 3-account ring rotating email & IP but sharing card & subnet
acc1 = {
    "user_id": "syndicate_acc_01",
    "name": "Vikram Patel",
    "email": "vikram.patel@gmail.com",
    "ip_address": "45.33.32.10",
    "device_id": "dev_ring_device_99",
    "payment_token": "pm_ring_card_99",
    "area": "delhi",
    "device_os": "windows",
    "signup_time": "2026-08-01 01:00:00"
}

acc2 = {
    "user_id": "syndicate_acc_02",
    "name": "Vikram P.",
    "email": "vikram.p+trial2@mailinator.com",
    "ip_address": "45.33.32.55",  # Rotated IP in same /24 subnet
    "device_id": "dev_ring_device_99",  # Same device
    "payment_token": "pm_ring_card_99",  # Same card
    "area": "delhi",
    "device_os": "windows",
    "signup_time": "2026-08-01 01:15:00"
}

acc3 = {
    "user_id": "syndicate_acc_03",
    "name": "V. Patel99",
    "email": "vpatel99@tempmail.com",
    "ip_address": "45.33.32.88",  # Rotated IP in same /24 subnet
    "device_id": "dev_new_phone_01",  # Rotated device
    "payment_token": "pm_ring_card_99",  # Same card
    "area": "delhi",
    "device_os": "windows",
    "signup_time": "2026-08-01 01:30:00"
}

for i, acc in enumerate([acc1, acc2, acc3], 1):
    res = engine.score_event(acc, update_state=True)
    print(f"\n[Syndicate Attempt #{i}] User: {acc['user_id']} ({acc['name']})")
    print(f"  Email: {acc['email']} | IP: {acc['ip_address']} | Card: {acc['payment_token']}")
    print(f"  -> RISK SCORE          : {res['risk_score']} / 100")
    print(f"  -> VERDICT             : {res['verdict']}")
    print(f"  -> RECOMMENDED ACTION  : {res['recommended_action']}")
    print(f"  -> GRAPH CLUSTER SIZE  : {res['raw_features']['graph_component_size']} connected entities")
    print(f"  -> 24h SUBNET VELOCITY : {res['raw_features']['subnet_signups_last_24h']} signups in /24")
    print(f"  -> CARD REUSE COUNT    : {res['raw_features']['payment_reuse_count']}")

print("\n" + "="*80)
