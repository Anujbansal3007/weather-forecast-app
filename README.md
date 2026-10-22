# WeatherWise ML — Weather Forecasting Mini Project

## What it does
- Trains a Random Forest **regression** model to estimate next-day temperature.
- Trains a Random Forest **classification** model to estimate rain risk.
- Displays predictions, basic evaluation metrics, a training-data chart, and feature importance.
- Uses a generated educational dataset so it can run without an API key or downloaded dataset.

> **Important:** This is an ML demonstration, not an official forecast. Its synthetic-data metrics do not represent real-world weather accuracy. For a scientifically valid project, train and test on verified historical weather observations.

## Run in VS Code (Windows)
1. Install **Python 3.10–3.12** and **Visual Studio Code**.
2. Extract this ZIP, then open the `WeatherWise_ML_Project` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -m venv .venv
   ```
5. Activate it in PowerShell:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
   If PowerShell blocks activation, you can skip activation and use the environment's Python directly:
   ```powershell
   .\.venv\Scripts\python.exe -m pip install -r requirements.txt
   .\.venv\Scripts\python.exe -m streamlit run app.py
   ```
6. Install packages:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```
7. Start the app:
   ```powershell
   python -m streamlit run app.py
   ```
8. Open the local URL printed in the terminal, usually `http://localhost:8501`.

### If `pip` times out
Check your internet/DNS first. Try:
```powershell
py -m pip install --default-timeout=120 -r requirements.txt
```
If the error says `getaddrinfo failed`, it usually means the computer cannot resolve the package host; retry on a working network or hotspot. Do not change Python code to fix a DNS problem.

## Deploy online with Streamlit Community Cloud
1. Create a GitHub repository and upload `app.py`, `requirements.txt`, and `README.md`.
2. Go to https://share.streamlit.io/ and sign in with GitHub.
3. Choose **Create app**, select the repository and branch, and set the main file path to `app.py`.
4. Click **Deploy**. The first build may take a few minutes.

## Suggested submission explanation
**Title:** WeatherWise ML: Next-Day Temperature and Rain-Risk Estimation Using Random Forest

**Objective:** Build an interactive ML application that predicts a next-day temperature estimate and rain likelihood from weather inputs.

**Algorithms:** Random Forest Regression and Random Forest Classification.

**Evaluation:** Mean Absolute Error (MAE) for temperature and accuracy for rain classification, evaluated on a held-out test split.

**Limitations:** Generated data is for demonstration only; real-world forecasting requires real observations and time-aware validation.
