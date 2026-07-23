import requests
import csv
import time
import os
from dotenv import load_dotenv

# Load the API key from .env file
load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

# The seasons we want to get data for
seasons = [2022, 2023, 2024]

# This will hold all our matches
all_matches = []

# Loop through each season and fetch matches
for season in seasons:
    print(f"Fetching season {season}...")

    url = f"https://api.football-data.org/v4/competitions/PL/matches?season={season}"
    headers = {"X-Auth-Token": API_KEY}

    response = requests.get(url, headers=headers)

    # Check if the request worked
    if response.status_code != 200:
        print(f"  Could not get season {season}, skipping...")
        continue

    # Get the list of matches from the response
    matches = response.json()["matches"]
    print(f"  Got {len(matches)} matches")

    # Loop through each match and grab what we need
    for match in matches:

        # Skip matches that haven't been played yet
        if match["status"] != "FINISHED":
            continue

        home_team = match["homeTeam"]["name"]
        away_team = match["awayTeam"]["name"]
        date = match["utcDate"][:10]
        matchday = match["matchday"]
        home_goals = match["score"]["fullTime"]["home"]
        away_goals = match["score"]["fullTime"]["away"]

        # Figure out the result
        if home_goals > away_goals:
            result = "H"  # Home win
        elif home_goals < away_goals:
            result = "A"  # Away win
        else:
            result = "D"  # Draw

        # Add this match to our list
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

    # Wait 7 seconds before next request so we don't get rate limited
    time.sleep(7)

# Save everything to a CSV file
os.makedirs("data", exist_ok=True)

with open("data/matches.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["date", "season", "matchday", "home_team", "away_team", "home_goals", "away_goals", "result"])
    writer.writeheader()
    writer.writerows(all_matches)

print(f"\nDone! Saved {len(all_matches)} matches to data/matches.csv")