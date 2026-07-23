import requests
import csv
import time
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

seasons = [2022, 2023, 2024]
all_matches = []

for season in seasons:
    print(f"Fetching season {season}...")

    url = f"https://api.football-data.org/v4/competitions/PL/matches?season={season}"
    headers = {"X-Auth-Token": API_KEY}

    try:
        response = requests.get(url, headers=headers, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"  Request failed for season {season}: {e}")
        continue

    if response.status_code != 200:
        print(f"  Could not get season {season}, skipping...")
        continue

    matches = response.json()["matches"]
    print(f"  Got {len(matches)} matches")

    for match in matches:
        if match["status"] != "FINISHED":
            continue

        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        # Skip if scores are missing (e.g. abandoned/void matches)
        if home_goals is None or away_goals is None:
            continue

        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        date = match["utcDate"][:10]
        matchday = match["matchday"]

        if home_goals > away_goals:
            result = "H"
        elif home_goals < away_goals:
            result = "A"
        else:
            result = "D"

        all_matches.append({
            "date": date,
            "season": season,
            "matchday": matchday,
            "home_team": home_team,
            "away_team": away_team,
            "home_goals": home_goals,
            "away_goals": away_goals,
            "result": result
        })

    time.sleep(7)

os.makedirs("data", exist_ok=True)

with open("data/matches.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "season", "matchday", "home_team", "away_team", "home_goals", "away_goals", "result"])
    writer.writeheader()
    writer.writerows(all_matches)

print(f"\nDone! Saved {len(all_matches)} matches to data/matches.csv")