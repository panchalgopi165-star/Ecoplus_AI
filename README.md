# EcoPulse AI — Cloud-Based Energy Waste Detector

A local-first research prototype that generates hourly electricity readings, forecasts the next 24 hours with Random Forest regression, flags unusual readings with Isolation Forest, gives evidence-linked recommendations, stores readings/analysis in SQLite, and supports CSV upload/download.

## 1. Run locally on Windows (VS Code)

Open the extracted `EcoPulse_AI` folder in VS Code. In Terminal → New Terminal, run one command at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

If activation is blocked by PowerShell, run this for the current terminal only, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Open the `Local URL` printed in the terminal (usually `http://localhost:8501`). If `python` is not recognized but `py` works, use `py` as shown above.

## 2. Testing before submission

The project includes automated smoke tests for sample-data generation, validation, 24-hour forecasting, anomaly output, recommendations, and SQLite save/reload. From the project root, run:

```powershell
py -m unittest discover -s tests -v
```

Expected result: 5 tests pass. The tests use a temporary database and do not erase the app's normal database.

Also perform this manual checklist:

- [ ] App opens at `localhost:8501` without a traceback.
- [ ] Overview shows metrics and energy charts.
- [ ] AI Forecast shows 24 forecast rows and MAE/RMSE holdout metrics.
- [ ] Anomaly Detection shows flagged readings or a clear no-anomaly message.
- [ ] Recommendations & History displays suggestions and saved analysis history.
- [ ] Download readings CSV and forecast CSV; open them to confirm they contain rows.
- [ ] Test CSV upload with at least 48 hourly rows and columns `timestamp,kwh`.
- [ ] Test an invalid CSV (missing columns or negative kWh) and confirm the app shows a validation message.

CSV example:

```csv
timestamp,kwh
2026-01-01 00:00:00,0.42
2026-01-01 01:00:00,0.38
```

The uploader accepts common alternatives such as `datetime`, `time`, `consumption`, and `energy_kwh`. Use hourly interval readings in kWh.

## 3. Deploy online for free with Streamlit Community Cloud

Cloud deployment requires your own GitHub account and your own Streamlit Community Cloud login. This package prepares the app, but it cannot publish to an account without your sign-in/authorization.

1. Create/sign in to GitHub and create a **new public repository** named `EcoPulse-AI` (or another name).
2. Upload the contents *inside* this `EcoPulse_AI` folder to the repository root. Confirm `app.py`, `requirements.txt`, `src/`, and `tests/` are at the top level—not inside an extra nested folder.
3. Open Streamlit Community Cloud: https://share.streamlit.io/ and sign in with GitHub.
4. Choose **Create app** / **Deploy an app**, select your repository and branch (usually `main`), and set **Main file path** to `app.py`.
5. In advanced settings, choose Python **3.11** if the selector is available, then deploy. Wait for package installation and app startup.
6. Open the generated public URL and test each of the four tabs plus CSV upload/download. If the build fails, open the app's logs and check the first Python/package error.

### Important cloud database limitation

The app currently uses SQLite at `data/ecopulse.db`. This is suitable for local testing, but a free cloud host's local filesystem may be temporary: saved readings/history can reset after restart or redeploy, and one SQLite file is not a robust shared database for multiple users. For a production-like cloud version, replace SQLite with a hosted database such as Supabase/PostgreSQL and store credentials in Streamlit Secrets—never commit passwords or secret keys to GitHub. This hosted-database integration is a separate step and requires setting up your own database account.

## 4. Folder structure

```text
EcoPulse_AI/
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── .python-version
├── data/
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── analysis.py
│   ├── recommendations.py
│   └── database.py
└── tests/
    └── test_project.py
```

## 5. How the AI works

- **Forecasting:** Random Forest uses hour, weekday, weekend flag, lag-1, lag-24, and rolling average features. It reports MAE and RMSE on a chronological holdout and then estimates the next 24 hours.
- **Anomaly detection:** Isolation Forest flags readings that differ from learned patterns. A flag means “review this reading,” not “confirmed energy waste.”
- **Recommendations:** Transparent rules produce evidence-linked suggestions; the prototype does not claim exact savings or identify individual appliances.

## Research limitations

Sample readings are synthetic and are only for demonstration. For research claims, evaluate on a public real-world dataset, compare forecasting with simple baselines, and report anomaly precision/recall/F1 if ground-truth labels are available. The prototype is not connected to a live smart meter and does not automatically control appliances.
