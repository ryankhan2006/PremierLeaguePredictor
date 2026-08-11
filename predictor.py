"""
predictor.py — Premier League 2026/27 standings predictor

Pipeline:
1. Load several past seasons of match results (football-data.co.uk CSVs —
   works even though football-data.org's free API only allows the current
   season's data)
2. Engineer features (recent team form)
3. Train a classifier (Win/Draw/Loss)
4. Fetch every scheduled 2026/27 fixture from football-data.org
5. Predict every fixture, then simulate the resulting league table
"""

import os

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

load_dotenv()
API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")
BASE_URL = "https://api.football-data.org/v4"
HEADERS = {"X-Auth-Token": API_KEY}

# football-data.co.uk season codes: "2425" = 2024/25 season, etc.
# Using the last 4 completed seasons gives a solid training set (~1,520 matches)
HISTORICAL_SEASON_CODES = ["2223", "2324", "2425", "2526"]

# football-data.co.uk uses short team names 
# football-data.org's API uses full names 
# We standardize everything to the football-data.org names so historical
# data and the 26/27 fixture list can be joined together.
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
    # Recently-relegated teams that still appear in historical data —
    # kept so past matches involving them parse correctly, even though
    # they won't appear in the 26/27 fixture list.
    "West Ham": "West Ham United FC",
    "Burnley": "Burnley FC",
    "Wolves": "Wolverhampton Wanderers FC",
    "Leicester": "Leicester City FC",
    "Southampton": "Southampton FC",
}

# The 20 teams playing in the 2026/27 season 
TEAMS_2026_27 = [
    "Arsenal FC", "Aston Villa FC", "AFC Bournemouth", "Brentford FC",
    "Brighton & Hove Albion FC", "Chelsea FC", "Coventry City FC",
    "Crystal Palace FC", "Everton FC", "Fulham FC", "Hull City FC",
    "Ipswich Town FC", "Leeds United FC", "Liverpool FC",
    "Manchester City FC", "Manchester United FC", "Newcastle United FC",
    "Nottingham Forest FC", "Sunderland AFC", "Tottenham Hotspur FC",
]


# Load historical match data

def fetch_historical_season(season_code):
    """Download one season's match results as a DataFrame from football-data.co.uk.

    season_code example: '2425' for the 2024/25 season.
    """
    url = f"https://www.football-data.co.uk/mmz4281/{season_code}/E0.csv"
    df = pd.read_csv(url)

    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    df.columns = ["date", "home_team", "away_team", "home_goals", "away_goals", "result"]

    # Standardize team names to match football-data.org's naming
    df["home_team"] = df["home_team"].map(TEAM_NAME_MAP).fillna(df["home_team"])
    df["away_team"] = df["away_team"].map(TEAM_NAME_MAP).fillna(df["away_team"])

    # football-data.co.uk dates are day-first (DD/MM/YYYY)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, utc=True)
    return df

# Downloading all seaosons
def fetch_historical_data(season_codes=HISTORICAL_SEASON_CODES):
    """Download and combine multiple seasons of historical match data."""
    all_seasons = []
    for code in season_codes:
        print(f"Fetching {code[:2]}/{code[2:]} season data...")
        all_seasons.append(fetch_historical_season(code))

    df = pd.concat(all_seasons, ignore_index=True)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} historical matches total.")
    return df


# Step 2: Feature engineering (unchanged logic from the single-season version)

def get_team_form(df, team, before_date, n=5):
    """Points per game for a team's last n matches before a given date."""
    past = df[
        ((df["home_team"] == team) | (df["away_team"] == team))
        & (df["date"] < before_date)
    ].tail(n)

    if len(past) == 0:
        return {
            "ppg": 1.0,
            "goals_for": 1.0,
            "goals_against": 1.0,
            "goal_diff": 0.0,
            "win_rate": 0.33,
        }  # neutral default (roughly 1 pt/game) if no history yet

    points = 0
    goals_for = 0
    goals_against = 0
    wins = 0
    
    for _, row in past.iterrows():
        if row["home_team"] == team:
            gf = row["home_goals"]
            ga = row["away_goals"]

            if row["result"] == "H":
                points += 3
                wins += 1
            elif row["result"] == "D":
                points += 1
        else:
            gf = row["away_goals"]
            ga = row["home_goals"]

            if row["result"] == "A":
                points += 3
                wins += 1
            elif row["result"] == "D":
                points += 1
        
        goals_for += gf
        goals_against += ga
    
    games = len(past)

    return {
        "ppg": points / games,
        "goals_for": goals_for / games,
        "goals_against": goals_against / games,
        "goal_diff": (goals_for - goals_against) / games,
        "win_rate": wins / games,
    }

def build_features(df):
    """Build a feature row (home form, away form, form diff) for every match."""
    feature_rows = []
    for _, row in df.iterrows():
        
        home5 = get_team_form(
            df,
            row["home_team"],
            row["date"],
            n=5
        )

        away5 = get_team_form(
            df,
            row["away_team"],
            row["date"],
            n=5
        )

        home10 = get_team_form(
            df,
            row["home_team"],
            row["date"],
            n=10
        )

        away10 = get_team_form(
            df,
            row["away_team"],
            row["date"],
            n=10
        )

        feature_rows.append({
            "home_ppg_5": home5["ppg"],
            "away_ppg_5": away5["ppg"],
            "home_ppg_10": home10["ppg"],
            "away_ppg_10": away10["ppg"],
            "home_gf": home10["goals_for"],
            "away_gf": away10["goals_for"],
            "home_ga": home10["goals_against"],
            "away_ga": away10["goals_against"],
            "home_gd": home10["goal_diff"],
            "away_gd": away10["goal_diff"],
            "home_win_rate": home10["win_rate"],
            "away_win_rate": away10["win_rate"],
            "ppg_diff": home10["ppg"] - away10["ppg"],
            "gd_diff": home10["goal_diff"] - away10["goal_diff"],
            "result": row["result"],
        })
        
    return pd.DataFrame(feature_rows)


# Train the model

def train_model(features_df):
    """Train a RandomForest classifier on Win/Draw/Loss and save it to disk."""
    feature_columns = [
        "home_ppg_5", "away_ppg_5", "home_ppg_10", "away_ppg_10",
        "home_gf", "away_gf", "home_ga", "away_ga",
        "home_gd", "away_gd", "home_win_rate", "away_win_rate",
        "ppg_diff", "gd_diff"
    ]

    X = features_df[feature_columns]
    y = features_df["result"]

    split = int(len(features_df) * 0.8)

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model trained. Test set accuracy: {accuracy:.2%}")
    joblib.dump(model, "model.pkl")

    return model


def load_or_train_model(df, force_retrain=False):
    """Load a saved model if one exists, otherwise train a new one."""
    if not force_retrain:
        try:
            model = joblib.load("model.pkl")
            print("Loaded existing model.pkl")
            return model
        except FileNotFoundError:
            pass
    print("Training a new model...")
    features_df = build_features(df)
    return train_model(features_df)


# Fetch the real 26/27 fixture list

def fetch_fixtures(competition="PL", status="SCHEDULED"):
    """Pull the upcoming fixture list for the current (2026/27) season.

    This works on the free tier because it's the CURRENT season — the free
    tier restriction only blocks pulling *past* seasons, not upcoming
    fixtures in the season that's live right now.
    """
    url = f"{BASE_URL}/competitions/{competition}/matches"
    params = {"status": status}
    response = requests.get(url, headers=HEADERS, params=params)
    if response.status_code != 200:
        print(f"API error {response.status_code}: {response.text}")
    response.raise_for_status()
    matches = response.json()["matches"]
    if not matches:
        raise ValueError(
            f"No fixtures returned for status={status}. "
            "The fixture list may not be published yet."
        )

    rows = []
    for m in matches:
        rows.append({
            "date": m["utcDate"],
            "home_team": m["homeTeam"]["name"],
            "away_team": m["awayTeam"]["name"],
        })
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} scheduled fixtures for the 26/27 season.")
    return df


# Predict every fixture, then simulate the standings table

def predict_fixture(model, hist_df, home_team, away_team, match_date):
    """Predict one fixture's outcome using form calculated as of match_date."""
    
    #Last 5 matches
    home5 = get_team_form(hist_df, home_team, match_date, n=5)
    away5 = get_team_form(hist_df, away_team, match_date, n=5)

    #Last 10 matches
    home10 = get_team_form(hist_df, home_team, match_date, n=10)
    away10 = get_team_form(hist_df, away_team, match_date, n=10)

    # Must match the exact features used in train_model()
    feature_data = {
        "home_ppg_5": home5["ppg"],
        "away_ppg_5": away5["ppg"],
        "home_ppg_10": home10["ppg"],
        "away_ppg_10": away10["ppg"],
        "home_gf": home10["goals_for"],
        "away_gf": away10["goals_for"],
        "home_ga": home10["goals_against"],
        "away_ga": away10["goals_against"],
        "home_gd": home10["goal_diff"],
        "away_gd": away10["goal_diff"],
        "home_win_rate": home10["win_rate"],
        "away_win_rate": away10["win_rate"],
        "ppg_diff": (home10["ppg"] - away10["ppg"]),
        "gd_diff": (home10["goal_diff"] - away10["goal_diff"]),
    }

    X_new = pd.DataFrame([feature_data])
    prediction = model.predict(X_new)[0]
    probabilities = dict(zip(model.classes_, model.predict_proba(X_new)[0]))

    return prediction, probabilities


def predict_season(model, hist_df, fixtures_df):
    """Predict every fixture in the season. Returns fixtures_df with predictions added.

    NOTE: this predicts every match using only PRE-SEASON form (since none of
    these matches have been played yet), so all fixtures for a given team
    early in the season will look similar. This is a simplification —
    a more advanced version could update form as predicted results roll in.
    """
    
    predictions = []
    probs_list = []
    for _, row in fixtures_df.iterrows():
        pred, probs = predict_fixture(
            model, hist_df, row["home_team"], row["away_team"], row["date"]
        )
        predictions.append(pred)
        probs_list.append(probs)

    fixtures_df = fixtures_df.copy()
    fixtures_df["predicted_result"] = predictions
    fixtures_df["probabilities"] = probs_list
    return fixtures_df


def simulate_standings(predicted_fixtures, teams=None):
    """Turn match-level predictions into a full league table.

    3 points for a predicted win, 1 for a predicted draw, 0 for a loss —
    same scoring as the real Premier League table.

    teams: list of team names to include. If not given, this is derived
    directly from the fixture list itself — this avoids silent bugs where
    a hardcoded team name doesn't exactly match what the API returns
    (e.g. "Hull City FC" vs the API's actual "Hull City AFC").
    """
    if teams is None:
        teams = sorted(set(predicted_fixtures["home_team"]) | set(predicted_fixtures["away_team"]))

    table = {
        team: {"played": 0, "wins": 0, "draws": 0, "losses": 0, "points": 0,
               "match_results": []}
        for team in teams
    }

    for _, row in predicted_fixtures.iterrows():
        home, away, result = row["home_team"], row["away_team"], row["predicted_result"]

        if home not in table or away not in table:
            continue  # skip any team not in our current-season list

        table[home]["played"] += 1
        table[away]["played"] += 1

        if result == "H":
            table[home]["wins"] += 1
            table[home]["points"] += 3
            table[away]["losses"] += 1
            table[home]["match_results"].append(("W", away))
            table[away]["match_results"].append(("L", home))
        elif result == "A":
            table[away]["wins"] += 1
            table[away]["points"] += 3
            table[home]["losses"] += 1
            table[away]["match_results"].append(("W", home))
            table[home]["match_results"].append(("L", away))
        else:  # draw
            table[home]["draws"] += 1
            table[home]["points"] += 1
            table[away]["draws"] += 1
            table[away]["points"] += 1
            table[home]["match_results"].append(("D", away))
            table[away]["match_results"].append(("D", home))

    standings_df = pd.DataFrame([
        {
            "team": team,
            "played": stats["played"],
            "wins": stats["wins"],
            "draws": stats["draws"],
            "losses": stats["losses"],
            "points": stats["points"],
        }
        for team, stats in table.items()
    ])
    standings_df = standings_df.sort_values(
        ["points", "wins"], ascending=False
    ).reset_index(drop=True)
    standings_df.index += 1  # ranks start at 1, not 0

    return standings_df, table


def print_team_results(table, team):
    """Print every predicted match result for one team."""
    print(f"\n--- Predicted results for {team} ---")
    for outcome, opponent in table[team]["match_results"]:
        label = {"W": "beat", "D": "drew with", "L": "lost to"}[outcome]
        print(f"  {outcome}: {label} {opponent}")


if __name__ == "__main__":
    hist_df = fetch_historical_data()
    model = load_or_train_model(hist_df)

    fixtures_df = fetch_fixtures()
    predicted_fixtures = predict_season(model, hist_df, fixtures_df)

    standings_df, results_table = simulate_standings(predicted_fixtures)

    print("\n=== Predicted 2026/27 Premier League Standings ===")
    print(standings_df.to_string(index=True))

    # Example: show match-by-match results for one team
    print_team_results(results_table, "Arsenal FC")