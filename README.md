# Car Price Prediction — AI & Big Data Final Project 2026

End-to-end supervised-learning (regression) project: **EDA → train 2 models →
evaluate → deploy a free web app**. Covers all four required steps.

## Files

| File | What it is |
|------|------------|
| `Car_Price_Prediction.ipynb` | Colab notebook — **Step 1–3** (EDA, training 2 models, evaluation). Already executed, outputs visible. |
| `car_dataset.csv` | Dataset (4,200 rows, 13 columns). Same columns as Kaggle *Car details v3* (CarDekho). |
| `car_price_model.joblib` | Best trained model (Random Forest pipeline) saved for deployment. |
| `model_meta.json` | Metrics + feature ranges/categories used by the web app. |
| `app.py` | **Step 4** — Streamlit web app (loads the model, takes user input, predicts, shows accuracy). |
| `requirements.txt` | Dependencies for deployment. |

## Results (test set, 80/20 split)

| Model | MAE | RMSE | R² |
|-------|-----|------|----|
| Linear Regression | ~20,100 | ~27,300 | 0.881 |
| **Random Forest (deployed)** | **~16,200** | **~24,000** | **0.908** |

## Step 4 — Deploy free on Streamlit Community Cloud

Streamlit Cloud is free and loads the `.joblib` model directly (much simpler than
Vercel/GitHub Pages, which run JavaScript and can't load a scikit-learn model without
extra conversion).

1. Create a **public GitHub repo** and upload: `app.py`, `requirements.txt`,
   `car_price_model.joblib`, `model_meta.json`.
2. Go to https://share.streamlit.io → **New app** → pick your repo → main file `app.py` → **Deploy**.
3. You get a public URL like `https://<your-app>.streamlit.app`. That's the link to submit.

Run locally first to check:
```bash
pip install -r requirements.txt
streamlit run app.py
```

The app exposes 11 inputs (well over the required 4), processes them through the saved
pipeline, shows the predicted price, and displays the model R²/MAE/RMSE on the page.

## Using the real Kaggle dataset

`car_dataset.csv` mirrors the exact columns of Kaggle's *Car details v3* (CarDekho):
`name, year, selling_price, km_driven, fuel, seller_type, transmission, owner,
mileage, engine, max_power, torque, seats`. To submit with the real dataset, download
*Car details v3.csv* from Kaggle, rename it to `car_dataset.csv`, and re-run the
notebook — **no code changes needed**.

## Final submission reminder

The assignment also asks for an **IEEE report**, **slides**, and a **~10-min English
presentation** — those aren't in this folder (you asked for the code only). Final ZIP
naming convention from the brief: `NIM Nama UAS AIBigData.zip`.

> ⚠️ Versions in `requirements.txt` are pinned to what trained the model
> (`scikit-learn==1.8.0`). If the deploy throws a version/install error, the
> scikit-learn pin is the one that must stay; pandas/numpy can be loosened.
