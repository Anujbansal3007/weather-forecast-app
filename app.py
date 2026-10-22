import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="WeatherWise ML", page_icon="🌦️", layout="wide")

@st.cache_data
def make_dataset(n=3500, seed=42):
    """Create a reproducible educational dataset when no historical dataset/API is available."""
    rng = np.random.default_rng(seed)
    month = rng.integers(1, 13, n)
    day = rng.integers(1, 29, n)
    seasonal = 27 + 7*np.sin((month-3)*2*np.pi/12)
    temp = seasonal + rng.normal(0, 4.5, n)
    humidity = np.clip(rng.normal(62 + 10*np.sin((month-6)*2*np.pi/12), 17, n), 15, 100)
    pressure = np.clip(rng.normal(1012 - 0.15*(humidity-60), 7, n), 985, 1035)
    wind = np.clip(rng.gamma(2.1, 2.2, n), 0, 28)
    cloud = np.clip(0.72*humidity + rng.normal(0, 17, n), 0, 100)
    rain_logit = -5.0 + 0.055*humidity + 0.025*cloud + 0.10*wind - 0.045*pressure + 0.045*(pressure-1000)
    rain_prob = 1/(1+np.exp(-rain_logit))
    rain = rng.binomial(1, np.clip(rain_prob, 0.02, 0.95))
    next_temp = temp + 0.18*(seasonal-temp) - 0.035*(cloud-50) + rng.normal(0, 2.2, n)
    df = pd.DataFrame({
        "temperature": temp, "humidity": humidity, "pressure": pressure,
        "wind_speed": wind, "cloud_cover": cloud, "month": month,
        "next_day_temperature": next_temp, "rain": rain
    })
    return df

@st.cache_resource
def train_models():
    df = make_dataset()
    features = ["temperature", "humidity", "pressure", "wind_speed", "cloud_cover", "month"]
    X = df[features]
    y_temp = df["next_day_temperature"]
    y_rain = df["rain"]
    Xtr, Xte, yttr, ytte = train_test_split(X, y_temp, test_size=0.2, random_state=42)
    _, _, yrtr, yrte = train_test_split(X, y_rain, test_size=0.2, random_state=42)
    temp_model = RandomForestRegressor(n_estimators=140, random_state=42, min_samples_leaf=2)
    rain_model = RandomForestClassifier(n_estimators=140, random_state=42, min_samples_leaf=2, class_weight="balanced")
    temp_model.fit(Xtr, yttr)
    rain_model.fit(Xtr, yrtr)
    temp_mae = mean_absolute_error(ytte, temp_model.predict(Xte))
    rain_acc = accuracy_score(yrte, rain_model.predict(Xte))
    return temp_model, rain_model, temp_mae, rain_acc, df

temp_model, rain_model, temp_mae, rain_acc, dataset = train_models()

st.title("🌦️ WeatherWise ML")
st.caption("A machine-learning mini project • Next-day temperature and rain-risk estimate")
st.info("**Project note:** This submission-ready demo trains Random Forest models on a generated educational dataset. It demonstrates the ML workflow; predictions are not official weather forecasts. For real-world accuracy, replace the generated data with verified historical weather observations.")

with st.sidebar:
    st.header("Enter weather conditions")
    city = st.text_input("City (label only)", "Indore")
    month = st.selectbox("Month", list(range(1, 13)), index=9, format_func=lambda m: pd.Timestamp(2026, m, 1).strftime("%B"))
    temperature = st.slider("Current temperature (°C)", -5.0, 48.0, 29.0, 0.5)
    humidity = st.slider("Humidity (%)", 10, 100, 55)
    pressure = st.slider("Atmospheric pressure (hPa)", 980, 1040, 1010)
    wind_speed = st.slider("Wind speed (km/h)", 0.0, 40.0, 8.0, 0.5)
    cloud_cover = st.slider("Cloud cover (%)", 0, 100, 35)
    run = st.button("Predict tomorrow", type="primary", use_container_width=True)

st.write("Adjust the conditions in the sidebar, then run the model.")
if run:
    row = pd.DataFrame([{
        "temperature": temperature, "humidity": humidity, "pressure": pressure,
        "wind_speed": wind_speed, "cloud_cover": cloud_cover, "month": month
    }])
    predicted_temp = float(temp_model.predict(row)[0])
    rain_probability = float(rain_model.predict_proba(row)[0][1]) * 100
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated temperature tomorrow", f"{predicted_temp:.1f} °C", f"{predicted_temp-temperature:+.1f} °C vs now")
    c2.metric("Rain risk (model estimate)", f"{rain_probability:.0f}%")
    c3.metric("Location", city)
    if rain_probability >= 60:
        st.warning("Model signal: higher rain risk. Consider carrying an umbrella—but verify with an official forecast.")
    elif rain_probability >= 35:
        st.info("Model signal: moderate rain risk. Check a reliable local forecast before planning.")
    else:
        st.success("Model signal: lower rain risk in this demo. This is not a guarantee of dry weather.")

st.divider()
st.subheader("Model performance on held-out demo data")
m1, m2 = st.columns(2)
m1.metric("Temperature model MAE", f"{temp_mae:.2f} °C")
m2.metric("Rain classifier accuracy", f"{rain_acc*100:.1f}%")
st.caption("Metrics are measured against the generated demo dataset and should not be interpreted as real-world forecast accuracy.")

left, right = st.columns(2)
with left:
    st.subheader("Training data: temperature distribution")
    fig, ax = plt.subplots()
    ax.hist(dataset["temperature"], bins=28)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Number of samples")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.subheader("Feature importance")
    importance = pd.Series(temp_model.feature_importances_, index=["Temperature", "Humidity", "Pressure", "Wind speed", "Cloud cover", "Month"]).sort_values()
    fig, ax = plt.subplots()
    importance.plot(kind="barh", ax=ax)
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    plt.close(fig)

with st.expander("About this project"):
    st.markdown("""
    **Problem:** Estimate next-day temperature and rain likelihood from weather conditions.

    **Workflow:** Generate/load data → split into training and test sets → train Random Forest regression and classification models → evaluate with MAE and accuracy → show predictions in a Streamlit dashboard.

    **Algorithms:** Random Forest Regressor for temperature; Random Forest Classifier for rain/no-rain.

    **Future improvements:** Use real historical observations, add date/location features, compare with official forecast APIs, cross-validate, and monitor prediction error.
    """)
