
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# ==========================================================
# 1. PAGE CONFIGURATION
# ==========================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌾",
    layout="wide"
)


# ==========================================================
# 2. PROJECT PATHS
# ==========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR, "models", "crop_yield_model.pkl"
)

DATA_DIR = os.path.join(
    BASE_DIR, "data", "raw"
)


# ==========================================================
# 3. LOAD DATASET
# ==========================================================

if not os.path.exists(DATA_DIR):
    st.error("Data folder not found!")
    st.code(DATA_DIR)
    st.stop()

csv_files = sorted([
    file for file in os.listdir(DATA_DIR)
    if file.lower().endswith(".csv")
])

if not csv_files:
    st.error("No CSV file found inside data/raw!")
    st.stop()

data_filename = (
    "yield_df.csv"
    if "yield_df.csv" in csv_files
    else csv_files[0]
)

DATA_PATH = os.path.join(DATA_DIR, data_filename)

try:
    df = pd.read_csv(DATA_PATH)
    df.drop(
        columns=["Unnamed: 0"],
        errors="ignore",
        inplace=True
    )

except Exception as error:
    st.error("Unable to load dataset.")
    st.exception(error)
    st.stop()


# ==========================================================
# 4. VALIDATE DATASET
# ==========================================================

required_columns = [
    "Area",
    "Item",
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield"
]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    st.error("Required columns are missing!")
    st.write(missing_columns)
    st.stop()


# ==========================================================
# 5. DATA CLEANING
# ==========================================================

numeric_columns = [
    "Year",
    "average_rain_fall_mm_per_year",
    "pesticides_tonnes",
    "avg_temp",
    "hg/ha_yield"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df.dropna(
    subset=required_columns,
    inplace=True
)

df["Area"] = df["Area"].astype(str)
df["Item"] = df["Item"].astype(str)

df = df[df["hg/ha_yield"] >= 0].copy()

if df.empty:
    st.error("No valid records found in the dataset.")
    st.stop()


# ==========================================================
# 6. LOAD TRAINED MODEL
# ==========================================================

if not os.path.exists(MODEL_PATH):
    st.error("Trained model file not found!")
    st.code(MODEL_PATH)
    st.stop()

try:
    model = joblib.load(MODEL_PATH)

except Exception as error:
    st.error("Unable to load trained model.")
    st.exception(error)
    st.stop()


# ==========================================================
# 7. CREATE DROPDOWN OPTIONS
# ==========================================================

crop_options = sorted(
    df["Item"].unique().tolist()
)

area_options = sorted(
    df["Area"].unique().tolist()
)


# ==========================================================
# 8. CROP-SPECIFIC YIELD THRESHOLDS
# ==========================================================

def get_crop_thresholds(crop_name):

    crop_data = df.loc[
        df["Item"] == crop_name,
        "hg/ha_yield"
    ].dropna()

    if crop_data.empty:
        crop_data = df["hg/ha_yield"]

    return {
        "q25": float(crop_data.quantile(0.25)),
        "q50": float(crop_data.quantile(0.50)),
        "q75": float(crop_data.quantile(0.75)),
        "average": float(crop_data.mean()),
        "count": len(crop_data)
    }


def classify_yield(value, thresholds):

    if value < thresholds["q25"]:
        return (
            "🔴 Poor",
            "The predicted yield is in the lower "
            "quarter of this crop's historical results."
        )

    elif value < thresholds["q50"]:
        return (
            "🟠 Average",
            "The predicted yield is below the "
            "historical middle value for this crop."
        )

    elif value < thresholds["q75"]:
        return (
            "🟡 Good",
            "The predicted yield is above the "
            "historical middle value for this crop."
        )

    else:
        return (
            "🟢 Excellent",
            "The predicted yield is in the upper "
            "quarter of this crop's historical results."
        )


# ==========================================================
# 9. SELECT HISTORICAL COMPARISON DATA
# ==========================================================

def get_reference_data(crop_name, area_name):

    crop_area_data = df[
        (df["Item"] == crop_name) &
        (df["Area"] == area_name)
    ].copy()

    crop_data = df[
        df["Item"] == crop_name
    ].copy()

    # Prefer the selected crop and area.

    if len(crop_area_data) >= 8:
        return crop_area_data, "Selected crop and area"

    # If insufficient, compare all areas for the crop.

    if len(crop_data) >= 8:
        return crop_data, "Selected crop across all areas"

    # Last fallback.

    return df.copy(), "Entire dataset (limited crop history)"


# ==========================================================
# 10. GENERATE CROP INSIGHTS
# ==========================================================

def generate_crop_insights(
    crop_name,
    area_name,
    rainfall,
    temperature,
    pesticides,
    predicted_yield
):

    reference, reference_label = get_reference_data(
        crop_name,
        area_name
    )

    high_yield_cutoff = reference[
        "hg/ha_yield"
    ].quantile(0.75)

    high_yield_records = reference[
        reference["hg/ha_yield"] >= high_yield_cutoff
    ]

    if high_yield_records.empty:
        high_yield_records = reference

    input_values = {
        "Rainfall": (
            rainfall,
            "average_rain_fall_mm_per_year",
            "mm/year"
        ),

        "Temperature": (
            temperature,
            "avg_temp",
            "°C"
        ),

        "Pesticide usage": (
            pesticides,
            "pesticides_tonnes",
            "tonnes"
        )
    }

    possible_factors = []
    recommendations = []
    comparison_rows = []

    for name, (user_value, column, unit) in input_values.items():

        historical = pd.to_numeric(
            reference[column],
            errors="coerce"
        ).dropna()

        high_group = pd.to_numeric(
            high_yield_records[column],
            errors="coerce"
        ).dropna()

        if historical.empty:
            continue

        q25 = historical.quantile(0.25)
        median = historical.median()
        q75 = historical.quantile(0.75)

        high_median = (
            high_group.median()
            if not high_group.empty
            else median
        )

        if user_value < q25:

            comparison = "Below usual range"

            possible_factors.append(
                f"{name} is lower than the usual "
                "historical range for this comparison group."
            )

            if name == "Rainfall":

                recommendations.append(
                    "Check whether the crop is receiving "
                    "enough water. Review irrigation "
                    "availability and the crop's water needs."
                )

            elif name == "Temperature":

                recommendations.append(
                    "Check whether the temperature is "
                    "suitable for this crop's growing period."
                )

            else:

                recommendations.append(
                    "Monitor pests and diseases. Check "
                    "whether pest control is needed before "
                    "making pesticide-use decisions."
                )

        elif user_value > q75:

            comparison = "Above usual range"

            possible_factors.append(
                f"{name} is higher than the usual "
                "historical range for this comparison group."
            )

            if name == "Rainfall":

                recommendations.append(
                    "Check field drainage and watch for "
                    "excess water or waterlogging."
                )

            elif name == "Temperature":

                recommendations.append(
                    "Check for signs of heat stress and "
                    "whether the crop is suitable for "
                    "these temperature conditions."
                )

            else:

                recommendations.append(
                    "Check whether the pesticide and its "
                    "quantity are suitable for this crop. "
                    "Avoid unnecessary pesticide use."
                )

        else:

            comparison = "Within usual range"

        comparison_rows.append({

            "Parameter": name,

            "Your Input": (
                f"{user_value:,.2f} {unit}"
            ),

            "Historical Lower Level": (
                f"{q25:,.2f} {unit}"
            ),

            "Historical Middle Value": (
                f"{median:,.2f} {unit}"
            ),

            "Historical Upper Level": (
                f"{q75:,.2f} {unit}"
            ),

            "Result": comparison

        })

    if not possible_factors:

        possible_factors.append(
            "Rainfall, temperature and pesticide usage "
            "are within their usual historical ranges. "
            "Other farming conditions may still affect yield."
        )

    if predicted_yield >= high_yield_cutoff:

        yield_context = (
            "The predicted yield is at or above the "
            "upper quarter of historical yields in "
            "the comparison group."
        )

        recommendations.append(
            "Continue monitoring crop health and field "
            "conditions. Compare the actual harvest "
            "with this prediction."
        )

    else:

        yield_context = (
            "The predicted yield is below the upper "
            "quarter of historical yields in the "
            "comparison group."
        )

        recommendations.append(
            "Check soil quality, seed variety, irrigation, "
            "sowing time, pest problems and farming methods."
        )

    recommendations = list(
        dict.fromkeys(recommendations)
    )

    return {

        "reference_label": reference_label,

        "reference_count": len(reference),

        "high_yield_cutoff": float(high_yield_cutoff),

        "yield_context": yield_context,

        "possible_factors": possible_factors,

        "recommendations": recommendations,

        "comparison_table": pd.DataFrame(comparison_rows)

    }


# ==========================================================
# 11. SIDEBAR NAVIGATION
# ==========================================================

st.sidebar.title("🌾 Crop Yield Prediction")

st.sidebar.caption(
    "Machine Learning Based Agricultural Analysis"
)

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

st.sidebar.divider()

st.sidebar.caption("Yield unit: hg/ha")


# ==========================================================
# 12. HOME PAGE
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

        It also compares user inputs with historical records
        to identify possible factors worth investigating.
        """
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "🌱 Crops",
        df["Item"].nunique()
    )

    col2.metric(
        "🌍 Areas",
        df["Area"].nunique()
    )

    col3.metric(
        "📊 Records",
        f"{len(df):,}"
    )

    st.divider()

    st.subheader("📌 Project Features")

    st.markdown(
        """
        - Crop yield prediction
        - Crop and area selection
        - Crop-specific yield classification
        - Historical yield comparison
        - Rainfall analysis
        - Temperature analysis
        - Model performance evaluation
        - Prediction-based data analysis
        - Possible reasons for low or high yield
        - Suggestions for further investigation
        """
    )

    st.info(
        "Insights are based on historical patterns. "
        "They do not prove cause and effect."
    )


# ==========================================================
# 13. PREDICT YIELD PAGE
# ==========================================================

elif page == "🔮 Predict Yield":

    st.title("🔮 Predict Crop Yield")

    st.write(
        "Enter agricultural information to predict crop yield."
    )

    st.divider()

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

            # MODEL PREDICTION

            prediction = model.predict(input_data)

            predicted_yield = float(prediction[0])

            if not np.isfinite(predicted_yield):
                raise ValueError(
                    "The model returned an invalid prediction."
                )

            # CROP-SPECIFIC CLASSIFICATION

            thresholds = get_crop_thresholds(crop)

            status, status_message = classify_yield(
                predicted_yield,
                thresholds
            )

            # GENERATE INSIGHTS

            insights = generate_crop_insights(
                crop,
                area,
                rainfall,
                temperature,
                pesticides,
                predicted_yield
            )

            # SAVE LATEST PREDICTION FOR DATA ANALYSIS PAGE

            st.session_state["last_prediction"] = {

                "crop": crop,

                "area": area,

                "year": int(year),

                "rainfall": rainfall,

                "temperature": temperature,

                "pesticides": pesticides,

                "predicted_yield": predicted_yield,

                "status": status,

                "status_message": status_message,

                "thresholds": thresholds,

                "insights": insights

            }

            st.success(
                "✅ Prediction completed successfully!"
            )

            # --------------------------------------------------
            # PREDICTED YIELD
            # --------------------------------------------------

            st.subheader("🌾 Predicted Crop Yield")

            st.metric(
                "Predicted Yield",
                f"{predicted_yield:,.2f} hg/ha"
            )

            st.divider()

            # --------------------------------------------------
            # YIELD STATUS
            # --------------------------------------------------

            st.subheader("📊 Yield Status")

            col1, col2 = st.columns(2)

            col1.metric(
                "Yield Category",
                status
            )

            col2.metric(
                f"{crop} Historical Average",
                f"{thresholds['average']:,.2f} hg/ha"
            )

            if "Poor" in status:

                st.warning(status_message)

            elif "Average" in status:

                st.info(status_message)

            else:

                st.success(status_message)

            st.caption(
                f"Classification uses {thresholds['count']:,} "
                f"historical records for {crop}."
            )

            st.divider()

            # --------------------------------------------------
            # PREDICTION INTERPRETATION
            # --------------------------------------------------

            st.subheader("📌 Prediction Interpretation")

            st.write(
                f"""
                The model predicts **{predicted_yield:,.2f} hg/ha**
                for **{crop}** in **{area}** during **{int(year)}**.

                The predicted result is classified as **{status}**
                based on the historical yield distribution of
                the selected crop.
                """
            )

            st.divider()

            # --------------------------------------------------
            # CROP YIELD INSIGHTS
            # --------------------------------------------------

            st.header(
                "🔍 Crop Yield Insights & Recommendations"
            )

            st.write(
                """
                This section compares your input values with
                historical records to identify conditions
                that may deserve attention.
                """
            )

            st.caption(
                f"Comparison basis: {insights['reference_label']} "
                f"({insights['reference_count']:,} records)."
            )

            st.subheader(
                "Possible Factors to Investigate"
            )

            for factor in insights["possible_factors"]:

                st.markdown(f"- {factor}")

            st.subheader(
                "📈 Historical Input Comparison"
            )

            if not insights["comparison_table"].empty:

                st.dataframe(
                    insights["comparison_table"],
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info(
                    "Not enough historical information for comparison."
                )

            st.write(insights["yield_context"])

            st.subheader(
                "💡 Recommendations"
            )

            for recommendation in insights["recommendations"]:

                st.markdown(f"- {recommendation}")

            st.caption(
                "These suggestions are based on historical data, "
                "not confirmed causes. Soil quality, seeds, irrigation, "
                "sowing time, pests and farming methods can also "
                "affect crop yield. Pesticide usage alone cannot "
                "show whether pest control was effective."
            )

            st.divider()

            # --------------------------------------------------
            # PREDICTION DETAILS
            # --------------------------------------------------

            st.subheader("📋 Prediction Details")

            display_data = pd.DataFrame({

                "Parameter": [
                    "Crop",
                    "Area / Country",
                    "Year",
                    "Rainfall",
                    "Temperature",
                    "Pesticides",
                    "Predicted Yield",
                    "Yield Status"
                ],

                "Value": [
                    crop,
                    area,
                    int(year),
                    f"{rainfall:,.2f} mm/year",
                    f"{temperature:.2f} °C",
                    f"{pesticides:,.2f} tonnes",
                    f"{predicted_yield:,.2f} hg/ha",
                    status
                ]

            })

            st.dataframe(
                display_data,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # --------------------------------------------------
            # CROP-SPECIFIC CLASSIFICATION RANGE
            # --------------------------------------------------

            st.subheader(
                "📈 Crop Yield Classification Range"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Poor (below 25th percentile)",
                f"< {thresholds['q25']:,.2f} hg/ha"
            )

            col2.metric(
                "Median (50th percentile)",
                f"{thresholds['q50']:,.2f} hg/ha"
            )

            col3.metric(
                "Excellent (75th percentile and above)",
                f"≥ {thresholds['q75']:,.2f} hg/ha"
            )

        except Exception as error:

            st.error("Prediction failed.")

            st.exception(error)


# ==========================================================
# 14. DATA ANALYSIS PAGE
#     BASED ONLY ON THE LATEST PREDICTION
# ==========================================================

elif page == "📊 Data Analysis":

    st.title("📊 Prediction Analysis")

    st.write(
        """
        Understand your predicted crop yield through
        historical comparisons and simple explanations.
        """
    )

    st.divider()

    # CHECK WHETHER A PREDICTION EXISTS

    if "last_prediction" not in st.session_state:

        st.info(
            """
            No prediction is available yet.

            Please visit the Predict Yield page and generate
            a crop yield prediction first. Your prediction
            analysis will appear here.
            """
        )

    else:

        result = st.session_state["last_prediction"]

        crop = result["crop"]
        area = result["area"]
        year = result["year"]

        predicted_yield = result["predicted_yield"]

        status = result["status"]

        thresholds = result["thresholds"]

        insights = result["insights"]

        # --------------------------------------------------
        # 1. PREDICTION SUMMARY
        # --------------------------------------------------

        st.subheader("🌾 Your Prediction Summary")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Selected Crop",
            crop
        )

        col2.metric(
            "Selected Area",
            area
        )

        col3.metric(
            "Predicted Yield",
            f"{predicted_yield:,.2f} hg/ha"
        )

        st.write(
            f"**Prediction Year:** {year}"
        )

        if "Poor" in status:

            st.warning(
                f"Predicted Result: {status}"
            )

        elif "Average" in status:

            st.info(
                f"Predicted Result: {status}"
            )

        else:

            st.success(
                f"Predicted Result: {status}"
            )

        st.divider()

        # --------------------------------------------------
        # 2. HISTORICAL YIELD COMPARISON
        # --------------------------------------------------

        st.subheader(
            "📈 How Does Your Prediction Compare?"
        )

        st.write(
            f"""
            The chart below compares your predicted yield
            for **{crop}** with the historical yield levels
            of the same crop.
            """
        )

        comparison = pd.DataFrame({

            "Yield Level": [
                "Lower historical level",
                "Historical median",
                "Historical average",
                "Upper historical level",
                "Your prediction"
            ],

            "Yield (hg/ha)": [
                thresholds["q25"],
                thresholds["q50"],
                thresholds["average"],
                thresholds["q75"],
                predicted_yield
            ]

        })

        st.bar_chart(
            comparison.set_index("Yield Level")
        )

        average = thresholds["average"]

        difference = predicted_yield - average

        if difference > 0:

            st.success(
                f"Your predicted yield is {difference:,.2f} hg/ha "
                f"higher than the historical average for {crop}."
            )

        elif difference < 0:

            st.warning(
                f"Your predicted yield is {abs(difference):,.2f} "
                f"hg/ha lower than the historical average for {crop}."
            )

        else:

            st.info(
                "Your predicted yield is equal to the historical average."
            )

        st.caption(
            f"Historical comparison uses "
            f"{thresholds['count']:,} records for {crop}."
        )

        st.divider()

        # --------------------------------------------------
        # 3. INPUT CONDITION ANALYSIS
        # --------------------------------------------------

        st.subheader(
            "🌦️ Analysis of Your Input Conditions"
        )

        st.write(
            """
            The following section checks whether your rainfall,
            temperature and pesticide values are within the
            usual historical ranges for the selected crop
            and area, where enough records are available.
            """
        )

        reference, reference_label = get_reference_data(
            crop,
            area
        )

        input_details = [

            (
                "🌧️ Rainfall",
                result["rainfall"],
                "average_rain_fall_mm_per_year",
                "mm/year"
            ),

            (
                "🌡️ Temperature",
                result["temperature"],
                "avg_temp",
                "°C"
            ),

            (
                "🧪 Pesticide Usage",
                result["pesticides"],
                "pesticides_tonnes",
                "tonnes"
            )

        ]

        for name, user_value, column, unit in input_details:

            historical = pd.to_numeric(
                reference[column],
                errors="coerce"
            ).dropna()

            if historical.empty:
                continue

            q25 = historical.quantile(0.25)
            q75 = historical.quantile(0.75)

            st.markdown(f"**{name}**")

            col1, col2 = st.columns(2)

            col1.metric(
                "Your Input",
                f"{user_value:,.2f} {unit}"
            )

            col2.metric(
                "Usual Historical Range",
                f"{q25:,.2f} – {q75:,.2f} {unit}"
            )

            if user_value < q25:

                st.warning(
                    f"{name} is lower than the usual "
                    "historical range."
                )

            elif user_value > q75:

                st.warning(
                    f"{name} is higher than the usual "
                    "historical range."
                )

            else:

                st.success(
                    f"{name} is within the usual "
                    "historical range."
                )

            st.divider()

        st.caption(
            f"Historical comparison: {reference_label} "
            f"({len(reference):,} records)."
        )

        # --------------------------------------------------
        # 4. POSSIBLE FACTORS
        # --------------------------------------------------

        st.subheader(
            "🔍 What May Need Attention?"
        )

        for factor in insights["possible_factors"]:

            st.markdown(f"- {factor}")

        st.divider()

        # --------------------------------------------------
        # 5. SUGGESTIONS
        # --------------------------------------------------

        st.subheader(
            "💡 Suggestions"
        )

        for recommendation in insights["recommendations"]:

            st.markdown(f"- {recommendation}")

        st.divider()

        # --------------------------------------------------
        # 6. OVERALL PREDICTION EXPLANATION
        # --------------------------------------------------

        st.subheader(
            "📝 Overall Prediction Explanation"
        )

        st.write(
            f"""
            For **{crop}** in **{area}** during **{year}**,
            the model predicts a yield of
            **{predicted_yield:,.2f} hg/ha**.

            The predicted result is classified as **{status}**
            when compared with the historical yield levels
            of the selected crop.

            The input analysis identifies conditions that
            may deserve further attention.
            """
        )

        st.caption(
            "These comparisons show historical patterns, "
            "not confirmed causes of crop yield. Actual yield "
            "can also depend on soil quality, seeds, irrigation, "
            "sowing time, pests and farming methods. "
            "The model does not identify the exact cause "
            "of a predicted result."
        )

        st.divider()

        # --------------------------------------------------
        # 7. VIEW PREDICTION INPUTS
        # --------------------------------------------------

        with st.expander("📋 View Your Prediction Details"):

            prediction_details = pd.DataFrame({

                "Parameter": [
                    "Crop",
                    "Area",
                    "Year",
                    "Rainfall",
                    "Temperature",
                    "Pesticides",
                    "Predicted Yield",
                    "Yield Category"
                ],

                "Value": [
                    crop,
                    area,
                    year,
                    f"{result['rainfall']:,.2f} mm/year",
                    f"{result['temperature']:.2f} °C",
                    f"{result['pesticides']:,.2f} tonnes",
                    f"{predicted_yield:,.2f} hg/ha",
                    status
                ]

            })

            st.dataframe(
                prediction_details,
                use_container_width=True,
                hide_index=True
            )


# ==========================================================
# 15. MODEL PERFORMANCE PAGE
# ==========================================================

elif page == "🤖 Model Performance":

    st.title("🤖 Model Performance")

    st.write(
        "Compare the performance of different Machine Learning "
        "models used for crop yield prediction."
    )

    st.info(
        "This page trains the comparison models when opened. "
        "It may take some time."
    )

    st.divider()

    # FEATURES AND TARGET

    X = df[
        [
            "Area",
            "Item",
            "Year",
            "average_rain_fall_mm_per_year",
            "pesticides_tonnes",
            "avg_temp"
        ]
    ]

    y = df["hg/ha_yield"]

    categorical_features = [
        "Area",
        "Item"
    ]

    # TRAIN TEST SPLIT

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # MODELS

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

    results = []

    with st.spinner("Training and evaluating models..."):

        for name, algorithm in models.items():

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

            results.append({

                "Model": name,

                "R²": r2_score(
                    y_test,
                    predictions
                ),

                "RMSE": np.sqrt(
                    mean_squared_error(
                        y_test,
                        predictions
                    )
                ),

                "MAE": mean_absolute_error(
                    y_test,
                    predictions
                )

            })

    results_df = pd.DataFrame(results)

    # BEST MODEL

    best_row = results_df.loc[
        results_df["R²"].idxmax()
    ]

    st.subheader(
        "🏆 Best Model on This Test Split"
    )

    st.success(
        f"Best Model: {best_row['Model']}"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Best Model",
        best_row["Model"]
    )

    col2.metric(
        "R² Score",
        f"{best_row['R²']:.4f}"
    )

    col3.metric(
        "RMSE",
        f"{best_row['RMSE']:,.2f}"
    )

    col4.metric(
        "MAE",
        f"{best_row['MAE']:,.2f}"
    )

    st.divider()

    # MODEL COMPARISON TABLE

    st.subheader("📊 Model Comparison")

    formatted_results = results_df.copy()

    formatted_results["R²"] = (
        formatted_results["R²"].round(4)
    )

    formatted_results["RMSE"] = (
        formatted_results["RMSE"].round(2)
    )

    formatted_results["MAE"] = (
        formatted_results["MAE"].round(2)
    )

    st.dataframe(
        formatted_results,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # R2 CHART

    st.subheader("📈 R² Score Comparison")

    st.bar_chart(
        results_df.set_index("Model")[["R²"]]
    )

    # ERROR CHART

    st.subheader("📉 Error Comparison")

    st.bar_chart(
        results_df.set_index("Model")[["RMSE", "MAE"]]
    )

    st.caption(
        "These metrics describe this particular random test split. "
        "They do not guarantee identical performance on future data."
    )


# ==========================================================
# 16. ABOUT PAGE
# ==========================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About the Project")

    st.write(
        """
        Crop Yield Prediction is a Machine Learning application
        that predicts crop yield using historical agricultural
        and environmental information.
        """
    )

    st.divider()

    st.subheader("🌱 Input Features")

    st.markdown(
        """
        - **Crop:** Selected crop type.
        - **Area / Country:** Selected geographical area.
        - **Year:** Year of prediction.
        - **Rainfall:** Average annual rainfall in mm/year.
        - **Temperature:** Average temperature in °C.
        - **Pesticides:** Pesticide usage in tonnes.
        """
    )

    st.divider()

    st.subheader("🤖 Machine Learning")

    st.write(
        """
        The prediction page uses the trained model saved in
        `models/crop_yield_model.pkl`.

        The Model Performance page compares:

        - Linear Regression
        - Decision Tree
        - Random Forest
        - Gradient Boosting
        """
    )

    st.divider()

    st.subheader("🔍 Prediction Analysis")

    st.write(
        """
        The Data Analysis page is based on the latest prediction.

        It compares the predicted yield and the entered
        agricultural conditions with historical records.

        It also provides simple explanations and suggestions
        for further investigation.
        """
    )

    st.caption(
        "The analysis identifies historical patterns, not proven "
        "causes. The dataset does not include every factor that "
        "affects crop yield, such as soil quality, seed variety, "
        "irrigation, sowing date, pest pressure and farming practices."
    )

    st.divider()

    st.subheader("📏 Evaluation Metrics")

    st.markdown(
        """
        - **R²:** Indicates how much variation in yield is
          explained by the model on evaluation data.
        - **RMSE:** Measures prediction error and gives
          greater weight to large errors.
        - **MAE:** Measures the average absolute prediction error.
        """
    )

    st.divider()

    st.caption(
        "Dataset: CSV file stored in the project's data/raw folder."
    )
