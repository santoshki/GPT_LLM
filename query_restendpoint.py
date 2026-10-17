import requests

# Replace with your actual Render URL
URL = "https://gpt-llm-2.onrender.com/chat"

payload = {
    "message": "I am not feeling well"
}

try:
    response = requests.post(URL, json=payload, timeout=60)

    print("Status Code:", response.status_code)
    print("Response JSON:")

    try:
        print(response.json())
    except Exception:
        print(response.text)

except requests.exceptions.RequestException as e:
    print("Request failed:", e)