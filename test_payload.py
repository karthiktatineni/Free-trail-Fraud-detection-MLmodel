from predict import FraudRiskEngine
import json

# Initialize engine with clean runtime state (model inference active, no training-data pre-population)
engine = FraudRiskEngine(warm_start=False)

print("=" * 80)
print("DEMONSTRATION: TRAINING-DATA NAMES & FIRST-TIME SIGNUPS EVALUATION")
print("=" * 80)

# 1. First-time signup with Ava (Name / email present in historical synthetic dataset)
payload_ava = {
    "name": "Ava",
    "email": "ava.anderson@gmail.com",
    "ip_address": "198.51.100.143",
    "device_id": "dev_iphone_819",
    "payment_token": "pm_mastercard_9371",
    "area": "seattle"
}

print("\n[1] Scoring 'Ava' (Name & Domain from training data, first-time signup):")
res_ava = engine.score_event(payload_ava, update_state=True)
print(f"  * Verdict            : {res_ava['verdict']}")
print(f"  * Risk Score         : {res_ava['risk_score']} / 100.0")
print(f"  * Recommended Action : {res_ava['recommended_action']}")
print(f"  * Severity           : {res_ava['severity']}")
print(f"  * Model Confidence   : {res_ava['model_confidence_pct']}%")

# 2. Another training name: 'David Smith'
payload_david = {
    "name": "David Smith",
    "email": "david.smith@gmail.com",
    "ip_address": "203.0.113.45",
    "device_id": "dev_macbook_pro_101",
    "payment_token": "pm_visa_card_101",
    "area": "london"
}

print("\n[2] Scoring 'David Smith' (First-time genuine signup):")
res_david = engine.score_event(payload_david, update_state=True)
print(f"  * Verdict            : {res_david['verdict']}")
print(f"  * Risk Score         : {res_david['risk_score']} / 100.0")
print(f"  * Recommended Action : {res_david['recommended_action']}")
print(f"  * Severity           : {res_david['severity']}")

# 3. Simulate Repeat Trial Abuse (Attempting to sign up again with Ava's payment token and device)
print("\n[3] Simulating Repeat Trial Abuse (Same device & payment token used a 2nd time):")
res_abuse = engine.score_event(payload_ava, update_state=True)
print(f"  * Verdict            : {res_abuse['verdict']}")
print(f"  * Risk Score         : {res_abuse['risk_score']} / 100.0")
print(f"  * Recommended Action : {res_abuse['recommended_action']}")
print(f"  * Top Risk Signals   : {dict(list(res_abuse['signal_breakdown'].items())[:3])}")
