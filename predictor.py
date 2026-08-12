"""
predictor.py — Premier League 2026/27 predictor

Pipeline:
1. Load historical matches from data/matches.csv
2. Build recent team-performance features
3. Train/evaluate a Random Forest classifier
4. Fetch the 2026/27 fixture list
5. Predict H/D/A probabilities for every fixture
6. Run thousands of Monte Carlo season simulations
7. Calculate expected standings and finishing probabilities
"""

import os

import joblib
import numpy as np
import pandas as pd
import requests

from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
)


# ==================================================
# CONFIG
# ==================================================

load_dotenv()

API_KEY = os.getenv("FOOTBALL_DATA_API_KEY")

BASE_URL = "https://api.football-data.org/v4"

HEADERS = {
    "X-Auth-Token": API_KEY
}


FEATURE_COLUMNS = [
    "home_ppg_5",
    "away_ppg_5",
    "home_ppg_10",
    "away_ppg_10",

    "home_gf",
    "away_gf",

    "home_ga",
    "away_ga",

    "home_gd",
    "away_gd",

    "home_win_rate",
    "away_win_rate",

    "ppg_diff",
    "gd_diff",
]


# Number of seasons to simulate
N_SIMULATIONS = 5000

RANDOM_SEED = 42


# ==================================================
# LOAD HISTORICAL DATA
# ==================================================

def load_historical_data(
    path="data/matches.csv"
):
    """Load cleaned historical Premier League matches."""

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} was not found. "
            "Run data_fetcher.py first."
        )

    df = pd.read_csv(path)

    df["date"] = pd.to_datetime(
        df["date"],
        utc=True
    )

    df = df.sort_values(
        "date"
    ).reset_index(drop=True)

    print(
        f"Loaded {len(df)} historical matches "
        f"from {path}."
    )

    return df


# ==================================================
# TEAM STATISTICS
# ==================================================

def get_team_stats(
    df,
    team,
    before_date,
    n=5
):
    """
    Calculate statistics from a team's
    previous n matches.
    """

    past = df[
        (
            (df["home_team"] == team)
            |
            (df["away_team"] == team)
        )
        &
        (df["date"] < before_date)
    ].tail(n)

    # Neutral default for teams without history
    if len(past) == 0:
        return {
            "ppg": 1.0,
            "goals_for": 1.0,
            "goals_against": 1.0,
            "goal_diff": 0.0,
            "win_rate": 0.33,
        }

    points = 0
    goals_for = 0
    goals_against = 0
    wins = 0

    for _, row in past.iterrows():

        # Team was home
        if row["home_team"] == team:

            goals_for += row["home_goals"]
            goals_against += row["away_goals"]

            if row["result"] == "H":
                points += 3
                wins += 1

            elif row["result"] == "D":
                points += 1

        # Team was away
        else:

            goals_for += row["away_goals"]
            goals_against += row["home_goals"]

            if row["result"] == "A":
                points += 3
                wins += 1

            elif row["result"] == "D":
                points += 1

    games = len(past)

    return {
        "ppg":
            points / games,

        "goals_for":
            goals_for / games,

        "goals_against":
            goals_against / games,

        "goal_diff":
            (
                goals_for
                - goals_against
            ) / games,

        "win_rate":
            wins / games,
    }


# ==================================================
# FEATURE ENGINEERING
# ==================================================

def create_feature_row(
    df,
    home_team,
    away_team,
    match_date
):
    """
    Create the exact feature set required
    by the model for one fixture.
    """

    home5 = get_team_stats(
        df,
        home_team,
        match_date,
        n=5
    )

    away5 = get_team_stats(
        df,
        away_team,
        match_date,
        n=5
    )

    home10 = get_team_stats(
        df,
        home_team,
        match_date,
        n=10
    )

    away10 = get_team_stats(
        df,
        away_team,
        match_date,
        n=10
    )

    return {
        "home_ppg_5":
            home5["ppg"],

        "away_ppg_5":
            away5["ppg"],

        "home_ppg_10":
            home10["ppg"],

        "away_ppg_10":
            away10["ppg"],

        "home_gf":
            home10["goals_for"],

        "away_gf":
            away10["goals_for"],

        "home_ga":
            home10["goals_against"],

        "away_ga":
            away10["goals_against"],

        "home_gd":
            home10["goal_diff"],

        "away_gd":
            away10["goal_diff"],

        "home_win_rate":
            home10["win_rate"],

        "away_win_rate":
            away10["win_rate"],

        "ppg_diff":
            (
                home10["ppg"]
                - away10["ppg"]
            ),

        "gd_diff":
            (
                home10["goal_diff"]
                - away10["goal_diff"]
            ),
    }


def build_features(df):
    """
    Build training features for every
    historical match.
    """

    rows = []

    print("Building features...")

    for _, match in df.iterrows():

        features = create_feature_row(
            df,
            match["home_team"],
            match["away_team"],
            match["date"]
        )

        features["result"] = (
            match["result"]
        )

        rows.append(
            features
        )

    return pd.DataFrame(
        rows
    )


# ==================================================
# TRAIN MODEL
# ==================================================

def train_model(features_df):
    """
    Train and evaluate the Random Forest.
    """

    X = features_df[
        FEATURE_COLUMNS
    ]

    y = features_df[
        "result"
    ]

    # Chronological split:
    # first 80% train
    # latest 20% test
    split = int(
        len(features_df) * 0.8
    )

    X_train = X.iloc[
        :split
    ]

    X_test = X.iloc[
        split:
    ]

    y_train = y.iloc[
        :split
    ]

    y_test = y.iloc[
        split:
    ]

    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=4,
        max_features="sqrt",
        random_state=42,
        n_jobs=-1,
    )

    print(
        "Training Random Forest..."
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    print(
        f"\nModel Accuracy: "
        f"{accuracy:.2%}"
    )

    print(
        "\n=== Classification Report ==="
    )

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    loss = log_loss(
        y_test,
        probabilities,
        labels=model.classes_
    )

    print(
        f"Log Loss: "
        f"{loss:.4f}"
    )

    joblib.dump(
        model,
        "model.pkl"
    )

    print(
        "\nModel saved to model.pkl"
    )

    return model


def load_or_train_model(
    hist_df,
    force_retrain=False
):
    """
    Load model.pkl if it exists.
    Otherwise train a new model.
    """

    if (
        os.path.exists("model.pkl")
        and not force_retrain
    ):

        print(
            "Loaded existing model.pkl"
        )

        return joblib.load(
            "model.pkl"
        )

    print(
        "Training a new model..."
    )

    features_df = build_features(
        hist_df
    )

    return train_model(
        features_df
    )


# ==================================================
# FETCH 2026/27 FIXTURES
# ==================================================

def fetch_fixtures(
    competition="PL",
    status="SCHEDULED"
):
    """
    Fetch upcoming Premier League fixtures
    from football-data.org.
    """

    if not API_KEY:
        raise ValueError(
            "FOOTBALL_DATA_API_KEY "
            "was not found in .env"
        )

    url = (
        f"{BASE_URL}/competitions/"
        f"{competition}/matches"
    )

    params = {
        "status": status
    }

    response = requests.get(
        url,
        headers=HEADERS,
        params=params,
        timeout=20
    )

    if response.status_code != 200:

        print(
            f"API error "
            f"{response.status_code}: "
            f"{response.text}"
        )

    response.raise_for_status()

    matches = response.json()[
        "matches"
    ]

    if not matches:

        raise ValueError(
            "No scheduled fixtures returned."
        )

    rows = []

    for match in matches:

        rows.append({
            "date":
                match["utcDate"],

            "home_team":
                match["homeTeam"]["name"],

            "away_team":
                match["awayTeam"]["name"],
        })

    fixtures_df = pd.DataFrame(
        rows
    )

    fixtures_df["date"] = (
        pd.to_datetime(
            fixtures_df["date"],
            utc=True
        )
    )

    fixtures_df = (
        fixtures_df
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Loaded {len(fixtures_df)} "
        "scheduled fixtures."
    )

    return fixtures_df


# ==================================================
# PREDICT ONE FIXTURE
# ==================================================

def predict_fixture(
    model,
    hist_df,
    home_team,
    away_team,
    match_date
):
    """
    Predict H/D/A probabilities for
    one match.
    """

    feature_data = create_feature_row(
        hist_df,
        home_team,
        away_team,
        match_date
    )

    X_new = pd.DataFrame(
        [feature_data]
    )

    X_new = X_new[
        FEATURE_COLUMNS
    ]

    probability_values = (
        model.predict_proba(
            X_new
        )[0]
    )

    probabilities = dict(
        zip(
            model.classes_,
            probability_values
        )
    )

    # Most likely outcome
    prediction = max(
        probabilities,
        key=probabilities.get
    )

    return (
        prediction,
        probabilities
    )


# ==================================================
# PREDICT ALL FIXTURES
# ==================================================

def predict_season(
    model,
    hist_df,
    fixtures_df
):
    """
    Calculate probabilities for all
    380 Premier League fixtures.
    """

    predictions = []
    probabilities = []

    print(
        "Predicting season fixtures..."
    )

    for _, fixture in fixtures_df.iterrows():

        prediction, probs = (
            predict_fixture(
                model,
                hist_df,
                fixture["home_team"],
                fixture["away_team"],
                fixture["date"]
            )
        )

        predictions.append(
            prediction
        )

        probabilities.append(
            probs
        )

    results_df = fixtures_df.copy()

    results_df[
        "predicted_result"
    ] = predictions

    results_df[
        "probabilities"
    ] = probabilities

    return results_df


# ==================================================
# DETERMINISTIC STANDINGS
# ==================================================

def simulate_standings(
    predicted_fixtures,
    teams=None
):
    """
    Create standings using only the single
    most likely result for every match.

    Kept mainly for API compatibility.
    Monte Carlo standings below should be
    used for the main season projection.
    """

    if teams is None:

        teams = sorted(
            set(
                predicted_fixtures[
                    "home_team"
                ]
            )
            |
            set(
                predicted_fixtures[
                    "away_team"
                ]
            )
        )

    table = {
        team: {
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
            "match_results": [],
        }
        for team in teams
    }

    for _, match in (
        predicted_fixtures.iterrows()
    ):

        home = match[
            "home_team"
        ]

        away = match[
            "away_team"
        ]

        result = match[
            "predicted_result"
        ]

        table[home]["played"] += 1
        table[away]["played"] += 1

        if result == "H":

            table[home]["wins"] += 1
            table[home]["points"] += 3

            table[away]["losses"] += 1

            table[home][
                "match_results"
            ].append(
                ("W", away)
            )

            table[away][
                "match_results"
            ].append(
                ("L", home)
            )

        elif result == "A":

            table[away]["wins"] += 1
            table[away]["points"] += 3

            table[home]["losses"] += 1

            table[away][
                "match_results"
            ].append(
                ("W", home)
            )

            table[home][
                "match_results"
            ].append(
                ("L", away)
            )

        else:

            table[home]["draws"] += 1
            table[away]["draws"] += 1

            table[home]["points"] += 1
            table[away]["points"] += 1

            table[home][
                "match_results"
            ].append(
                ("D", away)
            )

            table[away][
                "match_results"
            ].append(
                ("D", home)
            )

    standings_df = pd.DataFrame([
        {
            "team":
                team,

            "played":
                stats["played"],

            "wins":
                stats["wins"],

            "draws":
                stats["draws"],

            "losses":
                stats["losses"],

            "points":
                stats["points"],
        }

        for team, stats
        in table.items()
    ])

    standings_df = (
        standings_df
        .sort_values(
            [
                "points",
                "wins"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    standings_df.index += 1

    return (
        standings_df,
        table
    )


# ==================================================
# MONTE CARLO SEASON SIMULATION
# ==================================================

def monte_carlo_standings(
    predicted_fixtures,
    simulations=N_SIMULATIONS
):
    """
    Simulate the full Premier League season
    thousands of times.

    Instead of assuming the highest-probability
    result happens every time, sample results
    according to the model probabilities.
    """

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    teams = sorted(
        set(
            predicted_fixtures[
                "home_team"
            ]
        )
        |
        set(
            predicted_fixtures[
                "away_team"
            ]
        )
    )

    stats = {
        team: {
            "points_total": 0,
            "position_total": 0,
            "titles": 0,
            "top4": 0,
            "relegations": 0,
        }
        for team in teams
    }

    print(
        f"\nRunning {simulations:,} "
        "season simulations..."
    )

    for simulation in range(
        simulations
    ):

        season_table = {
            team: {
                "points": 0,
                "wins": 0,
            }
            for team in teams
        }

        for _, match in (
            predicted_fixtures.iterrows()
        ):

            home = match[
                "home_team"
            ]

            away = match[
                "away_team"
            ]

            probs = match[
                "probabilities"
            ]

            # Model probabilities
            p_home = float(
                probs.get("H", 0)
            )

            p_draw = float(
                probs.get("D", 0)
            )

            p_away = float(
                probs.get("A", 0)
            )

            probability_array = np.array(
                [
                    p_home,
                    p_draw,
                    p_away
                ],
                dtype=float
            )

            # Normalize in case of floating point error
            total = probability_array.sum()

            if total <= 0:

                probability_array = (
                    np.array(
                        [
                            1 / 3,
                            1 / 3,
                            1 / 3
                        ]
                    )
                )

            else:

                probability_array /= total

            result = rng.choice(
                [
                    "H",
                    "D",
                    "A"
                ],
                p=probability_array
            )

            if result == "H":

                season_table[
                    home
                ]["points"] += 3

                season_table[
                    home
                ]["wins"] += 1

            elif result == "A":

                season_table[
                    away
                ]["points"] += 3

                season_table[
                    away
                ]["wins"] += 1

            else:

                season_table[
                    home
                ]["points"] += 1

                season_table[
                    away
                ]["points"] += 1

        # Sort simulated league table
        ranked_teams = sorted(
            teams,
            key=lambda team: (
                season_table[
                    team
                ]["points"],
                season_table[
                    team
                ]["wins"]
            ),
            reverse=True
        )

        for position, team in enumerate(
            ranked_teams,
            start=1
        ):

            stats[
                team
            ]["points_total"] += (
                season_table[
                    team
                ]["points"]
            )

            stats[
                team
            ]["position_total"] += (
                position
            )

            if position == 1:

                stats[
                    team
                ]["titles"] += 1

            if position <= 4:

                stats[
                    team
                ]["top4"] += 1

            if position >= 18:

                stats[
                    team
                ]["relegations"] += 1

    rows = []

    for team in teams:

        rows.append({
            "team":
                team,

            "expected_points":
                (
                    stats[
                        team
                    ]["points_total"]
                    / simulations
                ),

            "average_position":
                (
                    stats[
                        team
                    ]["position_total"]
                    / simulations
                ),

            "title_chance":
                (
                    stats[
                        team
                    ]["titles"]
                    / simulations
                    * 100
                ),

            "top4_chance":
                (
                    stats[
                        team
                    ]["top4"]
                    / simulations
                    * 100
                ),

            "relegation_chance":
                (
                    stats[
                        team
                    ]["relegations"]
                    / simulations
                    * 100
                ),
        })

    standings = pd.DataFrame(
        rows
    )

    standings = (
        standings
        .sort_values(
            [
                "average_position",
                "expected_points"
            ],
            ascending=[
                True,
                False
            ]
        )
        .reset_index(drop=True)
    )

    standings.index += 1

    return standings


# ==================================================
# PRINT MONTE CARLO TABLE
# ==================================================

def print_monte_carlo_table(
    standings
):

    display = standings.copy()

    display[
        "expected_points"
    ] = display[
        "expected_points"
    ].round(1)

    display[
        "average_position"
    ] = display[
        "average_position"
    ].round(2)

    display[
        "title_chance"
    ] = display[
        "title_chance"
    ].round(1)

    display[
        "top4_chance"
    ] = display[
        "top4_chance"
    ].round(1)

    display[
        "relegation_chance"
    ] = display[
        "relegation_chance"
    ].round(1)

    print(
        "\n=== 2026/27 MONTE CARLO "
        "PREMIER LEAGUE PROJECTION ==="
    )

    print(
        display.to_string(
            index=True
        )
    )


# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    # Load historical training data
    hist_df = load_historical_data()

    # Train/load model
    model = load_or_train_model(
        hist_df
    )

    # Fetch upcoming fixtures
    fixtures_df = fetch_fixtures()

    # Generate probabilities for every match
    predicted_fixtures = predict_season(
        model,
        hist_df,
        fixtures_df
    )

    # Run Monte Carlo simulation
    monte_carlo_df = (
        monte_carlo_standings(
            predicted_fixtures,
            simulations=5000
        )
    )

    # Print realistic expected table
    print_monte_carlo_table(
        monte_carlo_df
    )