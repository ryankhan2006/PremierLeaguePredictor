from dotenv import load_dotenv
import os
import requests

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

print(API_KEY)


url = "https://api.football-data.org/v4/competitions/PL"

headers = {
    "X-Auth-Token": API_KEY
}

response = requests.get(url, headers=headers)

print("Status Code: ", response.status_code)
print(response.json())
