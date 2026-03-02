import httpx
import json

payload = {
    "administrative": 10,
    "administrative_duration": 100.0,
    "informational": 2,
    "informational_duration": 50.0,
    "product_related": 50,
    "product_related_duration": 2000.0,
    "bounce_rates": 0.0,
    "exit_rates": 0.0,
    "page_values": 50.0,
    "special_day": 0.0,
    "month": "Nov",
    "operating_systems": 2,
    "browser": 2,
    "region": 1,
    "traffic_type": 1,
    "visitor_type": "Returning_Visitor",
    "weekend": True
}

try:
    response = httpx.post("http://localhost:8000/predict", json=payload, timeout=10.0)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
except Exception as e:
    print(f"Error: {e}")
