import json
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load("car_price_model.joblib")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return model, meta


model, meta = load_artifacts()
m = meta["metrics"][meta["best_model"]]
cats = meta["categories"]
rng = meta["ranges"]  # [min, max, median] per numeric feature

st.title("🚗 Car Price Predictor")
st.caption(
    f"Best model: **{meta['best_model']}**  ·  trained on **{meta['n_rows']:,}** rows  ·  "
    f"R² = **{m['R2']:.3f}**"
)

# ---- model performance shown on the page (required) ----
c1, c2, c3 = st.columns(3)
c1.metric("R² (accuracy)", f"{m['R2']:.3f}")
c2.metric("MAE", f"{m['MAE']:,.0f}")
c3.metric("RMSE", f"{m['RMSE']:,.0f}")

st.divider()
st.subheader("Enter car details")

col1, col2 = st.columns(2)
with col1:
    brand = st.selectbox("Brand", cats["brand"])
    year = st.slider("Year", 1998, 2024, 2017)
    km_driven = st.number_input("Kilometers driven", min_value=0, max_value=500000,
                                value=60000, step=1000)
    fuel = st.selectbox("Fuel type", cats["fuel"])
    transmission = st.selectbox("Transmission", cats["transmission"])
with col2:
    engine_cc = st.slider("Engine (CC)", int(rng["engine_cc"][0]), int(rng["engine_cc"][1]),
                          int(rng["engine_cc"][2]))
    max_power_bhp = st.slider("Max power (bhp)", int(rng["max_power_bhp"][0]),
                              int(rng["max_power_bhp"][1]), int(rng["max_power_bhp"][2]))
    mileage_kmpl = st.slider("Mileage (kmpl)", float(rng["mileage_kmpl"][0]),
                             float(rng["mileage_kmpl"][1]), float(rng["mileage_kmpl"][2]))
    seats = st.selectbox("Seats", [4, 5, 6, 7, 8], index=1)
    owner = st.selectbox("Owner", cats["owner"])
    seller_type = st.selectbox("Seller type", cats["seller_type"])

if st.button("Predict price", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "car_age": 2024 - year,
        "km_driven": km_driven,
        "engine_cc": engine_cc,
        "max_power_bhp": max_power_bhp,
        "mileage_kmpl": mileage_kmpl,
        "seats": seats,
        "brand": brand,
        "fuel": fuel,
        "seller_type": seller_type,
        "transmission": transmission,
        "owner": owner,
    }])
    price = float(model.predict(row)[0])
    st.success(f"### Estimated price: {price:,.0f}")
    st.caption(
        "⚠️ Disclaimer: this is a model estimate, not a guaranteed market price. "
        f"Model accuracy (R²) is {m['R2']:.1%} on the test set; actual values may differ."
    )
