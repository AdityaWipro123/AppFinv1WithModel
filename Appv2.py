

import os
import numpy as np
import pandas as pd
import streamlit as st
import joblib
import altair as alt
import warnings
warnings.filterwarnings("ignore")

# ================= CONFIG =================
MODELS_DIR = 'models_saved'
LOGO_PATH = "Wiprohydraulics.png"   # <-- place logo in same folder

MODEL_OPTIONS = {
    "Linear Regression (all 5)": "LR_all_5",
    "Linear Regression (first 4)": "LR_first_4",
    "ElasticNet (all 5)": "ElasticNet_all_5",
    "XGBoost (all 5)": "XGB_all_5",
    "XGBoost (first 4)": "XGB_first_4"
}

INPUT_FEATURES = ['SCP', 'MCP', 'DCP', 'Proto', 'Spares']

OUTPUT_COLUMNS = [
    "Tools&Consumables", "SC cost", "SC cost-Localization", "Packing", "Test & Analysis",
    "Scrap cost", "Inv Var", "Power & Fuel", "Power & Fuel-R&D", "Outside Services",
    "Outside COVID", "Total VC1", "FCOW", "FCIW", "Manpower", "Welfare", "Training & Dev",
    "Repairs & Maintenance", "Telecom", "Tools of Capital nature", "Insurance",
    "EHS", "COVID related", "other", "Total Base Cost"
]

# ================= UI =================
st.set_page_config(page_title="Cost Sensitivity Analysis", layout="wide")

# -------- Logo --------
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=260)

st.title("Cost Sensitivity Analysis")

# -------- Model selection --------
model_label = st.selectbox(
    "Select model",
    list(MODEL_OPTIONS.keys())
)
model_name = MODEL_OPTIONS[model_label]

# -------- Inputs --------
st.subheader("Inputs")

cols = st.columns(5)
inputs = {}

for col, feature in zip(cols, INPUT_FEATURES):
    with col:
        inputs[feature] = st.slider(
            feature,
            min_value=0.0,
            max_value=500.0,
            value=100.0,
            step=0.1
        )

X_all_5 = np.array([[inputs[f] for f in INPUT_FEATURES]])
X_first_4 = X_all_5[:, :4]

total_input_sum = X_all_5.sum()

# -------- Predictions --------
st.subheader("Predicted Outputs")

results = []

for output in OUTPUT_COLUMNS:

    model_path = os.path.join(
        MODELS_DIR,
        output,
        f"{model_name}.pkl"
    )

    if not os.path.exists(model_path):
        y_pred = np.nan
    else:
        try:
            model = joblib.load(model_path)
            X = X_first_4 if "first_4" in model_name else X_all_5
            y_pred = abs(float(model.predict(X)[0]))
        except Exception:
            y_pred = np.nan

    pct_of_inputs = (
        (y_pred / total_input_sum) * 100
        if total_input_sum > 0 and not np.isnan(y_pred)
        else np.nan
    )

    results.append({
        "Output": output,
        "Prediction": y_pred,
        "% of Total Inputs": pct_of_inputs
    })

df_results = pd.DataFrame(results)

df_display = df_results.copy()
df_display["Prediction"] = df_display["Prediction"].round(2)
df_display["% of Total Inputs"] = df_display["% of Total Inputs"].round(2)

st.dataframe(df_display, use_container_width=True)

# -------- Bar Chart --------
st.subheader("Prediction Distribution")

chart_df = df_results.dropna(subset=["Prediction"])

bar_chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("Output:N", sort=OUTPUT_COLUMNS, axis=alt.Axis(labelAngle=-35)),
        y=alt.Y("Prediction:Q"),
        tooltip=[
            alt.Tooltip("Output:N"),
            alt.Tooltip("Prediction:Q", format=".2f"),
            alt.Tooltip("% of Total Inputs:Q", format=".2f")
        ]
    )
    .properties(height=450)
)

st.altair_chart(bar_chart, use_container_width=True)
