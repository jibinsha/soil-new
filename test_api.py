import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "lat": 10.5,
    "lon": 76.2
}

response = requests.post(url, json=data)

print(response.json())