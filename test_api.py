from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

if not API_KEY:
    raise ValueError("FOOTBALL_DATA_API_KEY not found — check your .env file")

print("API key loaded:", API_KEY is not None)

url = "https://api.football-data.org/v4/competitions/PL"
headers = {"X-Auth-Token": API_KEY}

response = requests.get(url, headers=headers, timeout=10)

print("Status Code: ", response.status_code)
print(response.json())