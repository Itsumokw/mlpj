import requests

payload = {
            "dataset_name": "Air Passengers (Default)",
            "time_col": "Month",
            "value_col": "#Passengers",
            "custom_data": None,
            "p": 12,
            "q": 1
        }

response = requests.post(
            f"http://127.0.0.1:8080/preprocess",
            json=payload,
            timeout=30
        )

print(response.json())