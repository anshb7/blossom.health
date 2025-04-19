import requests

r = requests.post(
    "https://a0d6-2a09-bac1-36c0-40-00-243-6.ngrok-free.app/api/generate",
    headers={"Content-Type": "application/json"},
    json={"model": "llama3", "prompt": "Hi Blossom!"}
)
print(r.status_code)
