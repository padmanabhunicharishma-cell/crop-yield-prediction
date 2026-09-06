import streamlit as st
import pandas as pd
import joblib
import os

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

import numpy as np


# ==========================================================
# PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)


# ==========================================================
# BASE DIRECTORY
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)


# ==========================================================
# FILE PATHS
# ==========================================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "crop_yield_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "yield_df.csv"
)


# ==========================================================
# LOAD DATASET
# ==========================================================

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "raw"
)

if not os.path.exists(DATA_DIR):
    st.error("❌ Data folder not found!")
    st.code(DATA_DIR)
    st.stop()

csv_files = [
    file for file in os.listdir(DATA_DIR)
    if file.lower().endswith(".csv")
]

if len(csv_files) == 0:
    st.error("❌ No CSV file found inside the data folder!")
    st.code(DATA_DIR)
    st.stop()

# Use the first CSV file found
DATA_PATH = os.path.join(
    DATA_DIR,
    csv_files[0]
)

try:

    df = pd.read_csv(DATA_PATH)

    # Remove unwanted CSV index column
    df = df.drop(
        columns=["Unnamed: 0"],
        errors="ignore"
    )

except Exception as e:

    st.error("❌ Unable to load dataset.")

    st.code(str(e))

    st.stop()


# ==========================================================
# LOAD SAVED MODEL
# ==========================================================

if not os.path.exists(MODEL_PATH):

    st.error("❌ Model file not found!")

    st.code(MODEL_PATH)

    st.stop()


try:

    model = joblib.load(MODEL_PATH)

except Exception as e:

    st.error("❌ Unable to load the trained model.")

    st.code(str(e))

    st.stop()


# ==========================================================
# CREATE OPTIONS
# ==========================================================

crop_options = sorted(
    df["Item"]
    .dropna()
    .unique()
    .tolist()
)

area_options = sorted(
    df["Area"]
    .dropna()
    .unique()
    .tolist()
)


# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("🌾 Crop Yield Prediction")

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "🏠 Home",
        "🔮 Predict Yield",
        "📊 Data Analysis",
        "🤖 Model Performance",
        "ℹ️ About"
    ]
)


# ==========================================================
# HOME PAGE
# ==========================================================

if page == "🏠 Home":

    st.title("🌾 Crop Yield Prediction")

    st.subheader(
        "Machine Learning Based Crop Yield Prediction System"
    )

    st.write(
        """
        This application uses Machine Learning to predict
        crop yield based on agricultural and environmental
        conditions.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "🌱 Crops",
            df["Item"].nunique()
        )

    with col2:

        st.metric(
            "🌍 Areas",
            df["Area"].nunique()
        )

    with col3:

        st.metric(
            "📊 Records",
            len(df)
        )

    st.divider()

    st.subheader("📌 Project Features")

    st.write(
        """
        • Crop yield prediction

        • Crop and area selection

        • Rainfall analysis

        • Temperature analysis

        • Crop yield comparison

        • Model performance evaluation

        • Interactive data visualization
        """
    )


# ==========================================================
# PREDICT YIELD PAGE
# ==========================================================

elif page == "🔮 Predict Yield":

    st.title("🔮 Predict Crop Yield")

    st.write(
        "Enter the agricultural information below "
        "to predict crop yield."
    )

    st.divider()

    # ======================================================
    # PHASE 24 — USER INPUT
    # ======================================================

    st.subheader("🌱 Crop Information")

    crop = st.selectbox(
        "Select Crop",
        crop_options
    )

    area = st.selectbox(
        "Select Area / Country",
        area_options
    )

    year = st.number_input(
        "Year",
        min_value=1961,
        max_value=2030,
        value=2025,
        step=1
    )

    rainfall = st.number_input(
        "Rainfall (mm/year)",
        min_value=0.0,
        value=1200.0,
        step=10.0
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-10.0,
        max_value=60.0,
        value=28.0,
        step=0.5
    )

    pesticides = st.number_input(
        "Pesticides (tonnes)",
        min_value=0.0,
        value=1000.0,
        step=100.0
    )

    st.divider()

    # ======================================================
    # PHASE 25 — MAKE PREDICTION
    # ======================================================

    if st.button(
        "🌾 Predict Yield",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "Area": [area],

            "Item": [crop],

            "Year": [year],

            "average_rain_fall_mm_per_year": [
                rainfall
            ],

            "pesticides_tonnes": [
                pesticides
            ],

            "avg_temp": [
                temperature
            ]
        })

        try:

            prediction = model.predict(
                input_data
            )

            st.success("Prediction completed!")

            st.subheader("🌾 Predicted Yield")

            predicted_yield = float(prediction[0])

            st.metric(
                label="Predicted Yield",
                value=f"{predicted_yield:,.2f} hg/ha"
            )

            st.info(
                "This prediction is based on the selected crop, area, year, "
                "rainfall, pesticide usage, and average temperature."
            )

            predicted_yield = float(
                prediction[0]
            )

            st.success(
                f"🌾 Predicted Yield: "
                f"{predicted_yield:,.2f} hg/ha"
            )

            st.divider()

            st.subheader(
                "📋 Prediction Details"
            )

            display_data = pd.DataFrame({

                "Parameter": [

                    "Crop",

                    "Area / Country",

                    "Year",

                    "Rainfall",

                    "Temperature",

                    "Pesticides"

                ],

                "Value": [

                    crop,

                    area,

                    year,

                    f"{rainfall:,.2f} mm/year",

                    f"{temperature:.2f} °C",

                    f"{pesticides:,.2f} tonnes"

                ]
            })

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

        except Exception as e:

            st.error(
                "❌ Prediction failed."
            )

            st.code(str(e))


# ==========================================================
# PHASE 26 — DATA ANALYSIS PAGE
# ==========================================================

elif page == "📊 Data Analysis":

    st.title("📊 Data Analysis")

    st.write(
        "Explore crop yield patterns and relationships "
        "in the dataset."
    )

    st.divider()

    # ======================================================
    # YIELD DISTRIBUTION
    # ======================================================

    st.subheader(
        "📊 Yield Distribution"
    )

    yield_distribution = (
        df["hg/ha_yield"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(
        yield_distribution
    )

    st.divider()

    # ======================================================
    # AVERAGE YIELD BY CROP
    # ======================================================

    st.subheader(
        "🌾 Average Yield by Crop"
    )

    average_yield_by_crop = (

        df.groupby("Item")["hg/ha_yield"]

        .mean()

        .sort_values(
            ascending=False
        )
    )

    st.bar_chart(
        average_yield_by_crop
    )

    st.divider()

    # ======================================================
    # RAINFALL VS YIELD
    # ======================================================

    st.subheader(
        "🌧️ Rainfall vs Yield"
    )

    rainfall_yield = df[
        [
            "average_rain_fall_mm_per_year",
            "hg/ha_yield"
        ]
    ].dropna()

    st.scatter_chart(
        rainfall_yield
    )

    st.divider()

    # ======================================================
    # TEMPERATURE VS YIELD
    # ======================================================

    st.subheader(
        "🌡️ Temperature vs Yield"
    )

    temperature_yield = df[
        [
            "avg_temp",
            "hg/ha_yield"
        ]
    ].dropna()

    st.scatter_chart(
        temperature_yield
    )


# ==========================================================
# PHASE 27 — MODEL PERFORMANCE PAGE
# ==========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Compare the performance of the Machine Learning "
        "models used for crop yield prediction."
    )

    st.divider()

    # ======================================================
    # FEATURES AND TARGET
    # ======================================================

    X = df.drop(
        "hg/ha_yield",
        axis=1
    )

    y = df["hg/ha_yield"]

    categorical_features = [
        "Area",
        "Item"
    ]

    numerical_features = [
        "Year",
        "average_rain_fall_mm_per_year",
        "pesticides_tonnes",
        "avg_temp"
    ]

    # ======================================================
    # PREPROCESSOR
    # ======================================================

    preprocessor = ColumnTransformer(

        transformers=[

            (
                "categorical",

                OneHotEncoder(
                    handle_unknown="ignore"
                ),

                categorical_features
            )

        ],

        remainder="passthrough"
    )

    # ======================================================
    # TRAIN TEST SPLIT
    # ======================================================

    X_train, X_test, y_train, y_test = train_test_split(

        X,

        y,

        test_size=0.2,

        random_state=42
    )

    # ======================================================
    # CREATE MODELS
    # ======================================================

    models = {

        "Linear Regression":
            LinearRegression(),

        "Decision Tree":
            DecisionTreeRegressor(
                random_state=42
            ),

        "Random Forest":
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            ),

        "Gradient Boosting":
            GradientBoostingRegressor(
                random_state=42
            )
    }

    # ======================================================
    # CALCULATE METRICS
    # ======================================================

    results = []

    trained_models = {}

    for name, algorithm in models.items():

        pipeline = Pipeline([

            (
                "preprocessor",
                preprocessor
            ),

            (
                "model",
                algorithm
            )
        ])

        pipeline.fit(
            X_train,
            y_train
        )

        predictions = pipeline.predict(
            X_test
        )

        r2 = r2_score(
            y_test,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_test,
                predictions
            )
        )

        mae = mean_absolute_error(
            y_test,
            predictions
        )

        results.append({

            "Model": name,

            "R²": r2,

            "RMSE": rmse,

            "MAE": mae
        })

        trained_models[name] = pipeline

    # ======================================================
    # RESULTS DATAFRAME
    # ======================================================

    results_df = pd.DataFrame(
        results
    )

    # ======================================================
    # FIND BEST MODEL
    # ======================================================

    best_index = results_df[
        "R²"
    ].idxmax()

    best_model_name = results_df.loc[
        best_index,
        "Model"
    ]

    best_r2 = results_df.loc[
        best_index,
        "R²"
    ]

    best_rmse = results_df.loc[
        best_index,
        "RMSE"
    ]

    best_mae = results_df.loc[
        best_index,
        "MAE"
    ]

    # ======================================================
    # BEST MODEL
    # ======================================================

    st.subheader(
        "🏆 Best Model"
    )

    st.success(
        f"Best Model: {best_model_name}"
    )

    # ======================================================
    # METRIC CARDS
    # ======================================================

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "🏆 Best Model",
            best_model_name
        )

    with col2:

        st.metric(
            "R²",
            f"{best_r2:.4f}"
        )

    with col3:

        st.metric(
            "RMSE",
            f"{best_rmse:.2f}"
        )

    with col4:

        st.metric(
            "MAE",
            f"{best_mae:.2f}"
        )

    st.divider()

    # ======================================================
    # MODEL COMPARISON
    # ======================================================

    st.subheader(
        "📊 Model Comparison"
    )

    formatted_results = results_df.copy()

    formatted_results["R²"] = (
        formatted_results["R²"]
        .round(4)
    )

    formatted_results["RMSE"] = (
        formatted_results["RMSE"]
        .round(2)
    )

    formatted_results["MAE"] = (
        formatted_results["MAE"]
        .round(2)
    )

    st.dataframe(
        formatted_results,
        use_container_width=True,
        hide_index=True
    )

    # ======================================================
    # R² COMPARISON
    # ======================================================

    st.subheader(
        "📈 R² Score Comparison"
    )

    r2_chart = results_df.set_index(
        "Model"
    )[["R²"]]

    st.bar_chart(
        r2_chart
    )

    st.info(
        f"ℹ️ {best_model_name} achieved the highest "
        f"R² score on the test dataset."
    )


# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title(
        "ℹ️ About the Project"
    )

    st.subheader(
        "🌾 Crop Yield Prediction using Machine Learning"
    )

    st.write(
        """
        This project predicts agricultural crop yield
        using Machine Learning techniques.
        """
    )

    st.divider()

    st.subheader(
        "📌 Input Features"
    )

    st.write(
        """
        • Area / Country

        • Crop

        • Year

        • Average Rainfall

        • Pesticides

        • Average Temperature
        """
    )

    st.subheader(
        "🤖 Machine Learning Models"
    )

    st.write(
        """
        • Linear Regression

        • Decision Tree Regressor

        • Random Forest Regressor

        • Gradient Boosting Regressor
        """
    )

    st.subheader(
        "📊 Evaluation Metrics"
    )

    st.write(
        """
        R² Score:
        Measures how well the model explains the variation
        in crop yield.

        RMSE:
        Measures the average size of prediction errors,
        giving more weight to large errors.

        MAE:
        Measures the average absolute prediction error.
        """
    )

    st.divider()

    st.caption(
        "🌾 Crop Yield Prediction System | "
        "Machine Learning Project"
    )