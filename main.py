"""
main.py — FastAPI app for the Premier League 26/27 predictor

Runs the full predictor.py pipeline ONCE at startup (downloading historical
data, training/loading the model, fetching fixtures, predicting the season)
and caches the results in memory. Every endpoint just reads from that cache,
so requests are fast even though the underlying computation isn't.

Run with: uvicorn main:app --reload
Then visit http://127.0.0.1:8000/docs for an interactive API explorer —
FastAPI builds that page automatically from this file.
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

import predictor

app = FastAPI(
    title="Premier League 26/27 Predictor",
    description="ML-based predictions for the 2026/27 Premier League season",
)

# This dict holds everything computed at startup. Endpoints read from it —
# they never recompute anything themselves.
cache = {
    "standings_df": None,
    "results_table": None,
    "hist_df": None,
    "model": None,
}


@app.on_event("startup")
def load_predictions():
    """Runs once when the server starts. Builds the model and predicts the
    whole season, storing everything in `cache` for endpoints to use."""
    print("Starting up — this runs the full prediction pipeline once...")

    hist_df = predictor.load_historical_data()
    model = predictor.load_or_train_model(hist_df)

    fixtures_df = predictor.fetch_fixtures()
    predicted_fixtures = predictor.predict_season(model, hist_df, fixtures_df)

    standings_df, results_table = predictor.simulate_standings(predicted_fixtures)

    cache["hist_df"] = hist_df
    cache["model"] = model
    cache["standings_df"] = standings_df
    cache["results_table"] = results_table

    print("Startup complete — predictions cached and ready.")


@app.get("/health")
def health_check():
    """Simple endpoint to confirm the API is running."""
    return {"status": "ok", "message": "Premier League predictor API is running"}


@app.get("/", response_class=HTMLResponse)
def standings_page():
    """A styled HTML page showing the predicted standings table."""
    if cache["standings_df"] is None:
        return HTMLResponse(
            "<h1 style='font-family:sans-serif;color:#fff;background:#0B1120;padding:2rem'>"
            "Still starting up — refresh in a moment.</h1>"
        )

    standings_df = cache["standings_df"]
    results_table = cache["results_table"]

    outcome_color = {"W": "#3FB68B", "D": "#6B7280", "L": "#E2574C"}

    rows_html = ""
    for rank, row in standings_df.iterrows():
        team = row["team"]
        last_5 = results_table[team]["match_results"][-5:]
        form_pills = "".join(
            f'<span class="pill" style="background:{outcome_color[outcome]}" '
            f'title="{outcome} vs {opponent}"></span>'
            for outcome, opponent in last_5
        )
        rows_html += f"""
        <tr>
          <td class="rank">{rank}</td>
          <td class="team">{team}</td>
          <td class="num">{row['played']}</td>
          <td class="num win">{row['wins']}</td>
          <td class="num draw">{row['draws']}</td>
          <td class="num loss">{row['losses']}</td>
          <td class="num points">{row['points']}</td>
          <td class="form">{form_pills}</td>
        </tr>
        """

    html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
      <meta charset="UTF-8">
      <title>2026/27 Predicted Premier League Standings</title>
      <link rel="preconnect" href="https://fonts.googleapis.com">
      <link href="https://fonts.googleapis.com/css2?family=Oswald:wght@500;700&family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
      <style>
        :root {{
          --bg: #0B1120;
          --panel: #131B2E;
          --border: #232E47;
          --text: #E8E6E1;
          --muted: #8892A6;
          --gold: #C9A227;
          --win: #3FB68B;
          --draw: #6B7280;
          --loss: #E2574C;
        }}
        * {{ box-sizing: border-box; }}
        body {{
          margin: 0;
          background: var(--bg);
          color: var(--text);
          font-family: 'Inter', sans-serif;
          padding: 3rem 1.5rem;
        }}
        .wrap {{ max-width: 780px; margin: 0 auto; }}
        .eyebrow {{
          font-family: 'Oswald', sans-serif;
          font-size: 0.8rem;
          letter-spacing: 0.15em;
          text-transform: uppercase;
          color: var(--gold);
          margin-bottom: 0.4rem;
        }}
        h1 {{
          font-family: 'Oswald', sans-serif;
          font-weight: 700;
          text-transform: uppercase;
          font-size: 2.1rem;
          letter-spacing: 0.02em;
          margin: 0 0 0.3rem 0;
        }}
        .subtitle {{
          color: var(--muted);
          font-size: 0.9rem;
          margin-bottom: 2rem;
        }}
        table {{
          width: 100%;
          border-collapse: collapse;
          background: var(--panel);
          border: 1px solid var(--border);
          border-radius: 8px;
          overflow: hidden;
        }}
        thead th {{
          font-family: 'Oswald', sans-serif;
          text-transform: uppercase;
          font-size: 0.72rem;
          letter-spacing: 0.08em;
          color: var(--muted);
          text-align: right;
          padding: 0.9rem 0.8rem;
          border-bottom: 1px solid var(--border);
        }}
        thead th:first-child, thead th:nth-child(2) {{ text-align: left; }}
        tbody td {{
          padding: 0.7rem 0.8rem;
          border-bottom: 1px solid var(--border);
          font-variant-numeric: tabular-nums;
          text-align: right;
        }}
        tbody tr:last-child td {{ border-bottom: none; }}
        tbody tr:hover {{ background: rgba(255,255,255,0.03); }}
        td.rank {{
          text-align: left;
          color: var(--muted);
          font-family: 'Oswald', sans-serif;
          width: 2rem;
        }}
        td.team {{
          text-align: left;
          font-weight: 600;
        }}
        td.points {{
          font-weight: 700;
          color: var(--gold);
        }}
        td.win {{ color: var(--win); }}
        td.loss {{ color: var(--loss); }}
        .pill {{
          display: inline-block;
          width: 8px;
          height: 8px;
          border-radius: 50%;
          margin-left: 3px;
        }}
        .form {{ text-align: right; white-space: nowrap; }}
        footer {{
          margin-top: 1.5rem;
          font-size: 0.78rem;
          color: var(--muted);
          text-align: center;
        }}
      </style>
    </head>
    <body>
      <div class="wrap">
        <div class="eyebrow">Premier League &middot; ML Projection</div>
        <h1>2026/27 Predicted Standings</h1>
        <div class="subtitle">Trained on 4 seasons of historical results &middot; form pills show each team's last 5 predicted results</div>
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Team</th>
              <th>P</th>
              <th>W</th>
              <th>D</th>
              <th>L</th>
              <th>Pts</th>
              <th>Form</th>
            </tr>
          </thead>
          <tbody>
            {rows_html}
          </tbody>
        </table>
        <footer>Predicted, not real results &middot; /docs for the raw API</footer>
      </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html)


@app.get("/standings")
def get_standings():
    """Return the full predicted 26/27 league table."""
    if cache["standings_df"] is None:
        raise HTTPException(status_code=503, detail="Predictions not ready yet — server is still starting up.")

    # .to_dict(orient="records") turns the DataFrame into a list of
    # JSON-friendly dictionaries, one per team.
    return cache["standings_df"].to_dict(orient="records")


@app.get("/team/{team_name}/results")
def get_team_results(team_name: str):
    """Return every predicted match result for one team.

    Example: /team/Arsenal FC/results
    (team names must match football-data.org's exact naming, e.g. "Arsenal FC")
    """
    table = cache["results_table"]
    if table is None:
        raise HTTPException(status_code=503, detail="Predictions not ready yet — server is still starting up.")

    if team_name not in table:
        raise HTTPException(
            status_code=404,
            detail=f"Team '{team_name}' not found. Check the exact name (e.g. 'Arsenal FC').",
        )

    matches = [
        {"outcome": outcome, "opponent": opponent}
        for outcome, opponent in table[team_name]["match_results"]
    ]
    return {
        "team": team_name,
        "wins": table[team_name]["wins"],
        "draws": table[team_name]["draws"],
        "losses": table[team_name]["losses"],
        "points": table[team_name]["points"],
        "matches": matches,
    }


@app.get("/predict/{home_team}/{away_team}")
def predict_matchup(home_team: str, away_team: str):
    """Predict any single matchup on demand, using current-day form.

    Example: /predict/Arsenal FC/Chelsea FC
    """
    if cache["model"] is None:
        raise HTTPException(status_code=503, detail="Model not ready yet — server is still starting up.")

    now = predictor.pd.Timestamp.now(tz="UTC")
    prediction, probabilities = predictor.predict_fixture(
        cache["model"], cache["hist_df"], home_team, away_team, now
    )

    return {
        "home_team": home_team,
        "away_team": away_team,
        "prediction": prediction,
        "probabilities": {k: round(float(v), 3) for k, v in probabilities.items()},
    }