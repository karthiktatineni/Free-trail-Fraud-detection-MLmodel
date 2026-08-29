import requests

url = "https://free-trail-fraud-detection-mlmodel.onrender.com/api/v1/score"
headers = {
    "Content-Type": "application/json",
    "X-API-Key": "fk_live_0bb674919a9999ac87ceb1f590db4da9d18669c4b451c8df"
}

# New Genuine User 1: Maya Patel (London)
payload_1 = {
    "name": "Maya Patel",
    "email": "maya.patel.ldn@gmail.com",
    "ip_address": "82.165.197.101",
    "device_id": "dev_macbook_m3_maya_881",
    "payment_token": "pm_barclays_visa_maya_920",
    "area": "london",
    "payment_country": "GB"
}

# New Genuine User 2: Ethan Wright (San Francisco)
payload_2 = {
    "name": "Ethan Wright",
    "email": "ethan.wright.tech@gmail.com",
    "ip_address": "66.220.149.25",
    "device_id": "dev_pixel_9_ethan_334",
    "payment_token": "pm_chase_freedom_ethan_104",
    "area": "san_francisco",
    "payment_country": "US"
}

# New Genuine User 3: Chloe Tremblay (Toronto)
payload_3 = {
    "name": "Chloe Tremblay",
    "email": "chloe.tremblay.to@gmail.com",
    "ip_address": "142.250.190.46",
    "device_id": "dev_iphone_16_chloe_512",
    "payment_token": "pm_rbc_avion_chloe_789",
    "area": "toronto",
    "payment_country": "CA"
}

# Test with payload_1 (switch to payload_2 or payload_3 as needed)
res = requests.post(url, json=payload_1, headers=headers)
decision = res.json()

print(f"User Tested : {payload_1['name']}")
print("Verdict     :", decision.get("verdict"))
print("Risk Score  :", decision.get("risk_score"))
print("Action      :", decision.get("recommended_action"))
