import os
import pandas as pd

# Seasons used to train the model (historical data)
SEASONS = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]


TEAM_NAME_MAP = {
    "Arsenal": "Arsenal FC",
    "Aston Villa": "Aston Villa FC",
    "Bournemouth": "AFC Bournemouth",
    "Brentford": "Brentford FC",
    "Brighton": "Brighton & Hove Albion FC",
    "Chelsea": "Chelsea FC",
    "Coventry": "Coventry City FC",
    "Crystal Palace": "Crystal Palace FC",
    "Everton": "Everton FC",
    "Fulham": "Fulham FC",
    "Hull": "Hull City AFC",
    "Ipswich": "Ipswich Town FC",
    "Leeds": "Leeds United FC",
    "Liverpool": "Liverpool FC",
    "Man City": "Manchester City FC",
    "Man United": "Manchester United FC",
    "Newcastle": "Newcastle United FC",
    "Nott'm Forest": "Nottingham Forest FC",
    "Sunderland": "Sunderland AFC",
    "Tottenham": "Tottenham Hotspur FC",

    # Historical teams (not in prem right now)
    "West Ham": "West Ham United FC",
    "Burnley": "Burnley FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Leicester": "Leicester City FC",
    "Southampton": "Southampton FC",
    "Sheffield United": "Sheffield United FC",
    "Norwich": "Norwich City FC",
    "Watford": "Watford FC",
}

def fetch_season(season_code):
    print(f"Fetching {season_code[:2]}/{season_code[2:]} season...")
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    df = pd.read_csv(url)
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()

    df.columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]
    df["home_team"] = (df["home_team"].map(TEAM_NAME_MAP).fillna(df["home_team"]))
    df["away_team"] = (df["away_team"].map(TEAM_NAME_MAP).fillna(df["away_team"]))
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce", utc=True)

    df = df.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals", "result"])

    df["season"] = season_code
    print(f" Got {len(df)} matches")
    return df

def main():
    all_seasons = []

    for season in SEASONS:
        season_df = fetch_season(season)
        all_seasons.append(season_df)

    matches = pd.concat(all_seasons, ignore_index=True)

    matches = matches.sort_values(by="date").reset_index(drop=True)

    os.makedirs("data", exist_ok=True)

    matches.to_csv("data/matches.csv", index=False)

    print(f"\nDone! Saved {len(matches)} matches to data/historical_matches.csv")

if __name__ == "__main__":
    main()




