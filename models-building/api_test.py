import requests, json

data = json.load(open("sample.json"))
payload = {"pair": "EURUSD", "data": data}

res = requests.post("http://localhost:8000/predict", json=payload)
print(res.json())
