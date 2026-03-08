import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from io import BytesIO

class EnsembleModel(BaseEstimator, RegressorMixin):
    def __init__(self, models, weights):
        """
        Initialize the ensemble regression model.

        :param models: A dictionary containing base learner models
        :param weights: Array of model weights
        """
        self.models = models
        self.weights = np.array(weights)

        if len(self.models) != len(self.weights):
            raise ValueError("The number of models does not match the number of weights!")

        self.model_names = list(self.models.keys())
        self.weights_dict = dict(zip(self.model_names, self.weights))

    def fit(self, X, y):
        for model in self.models.values():
            model.fit(X, y)
        return self

    def predict(self, X):
        predictions = np.zeros((X.shape[0]))
        for model_name, model in self.models.items():
            model_pred = model.predict(X)
            model_weight = self.weights_dict[model_name]
            predictions += model_pred * model_weight
        return predictions

    def get_model_weights(self):
        return self.weights_dict


st.set_page_config(page_title="Online Predictor", page_icon="📈", layout="wide")
st.title("📈 Predictor SR")
st.write("Supports single prediction and batch prediction (CSV / XLSX). ")

# =========================
# Load scaler and model
# =========================
@st.cache_resource
def load_objects():
    scaler = joblib.load("standard_scaler.pkl")
    model = joblib.load("ensemble.pkl")
    return scaler, model

try:
    scaler, ensemble_model = load_objects()
except Exception as e:
    st.error(f"Failed to load model or scaler: {e}")
    st.stop()

# =========================
# Feature order (must match training)
# =========================
feature_names = [
    "infP", "infC", "infAc", "infpro", "infS",
    "MLSS", "MLVSS", "VSS/TSS", "volumn",
    "ana-time", "pH", "T", "salinity"
]

default_values = [
    10.39, 74.03, 49.60, 24.43, 148.05,
    7.50, 5.30, 0.71, 14.0,
    3.0, 7.6, 22.0, 0.70
]

# =========================
# Utility functions
# =========================
def predict_dataframe(df, scaler, model, feature_names):
    """
    Perform batch prediction on a DataFrame.
    The DataFrame must contain all required feature columns.
    """
    missing_cols = [col for col in feature_names if col not in df.columns]
    if missing_cols:
        raise ValueError(f"The uploaded file is missing the following required columns: {missing_cols}")

    # Select features in training order
    X = df[feature_names].copy()

    # Standardize
    X_scaled = scaler.transform(X)

    # Predict
    preds = model.predict(X_scaled)

    # Return original features with prediction column
    result_df = df.copy()
    result_df["prediction"] = preds
    return result_df, X_scaled

def dataframe_to_excel_bytes(df, sheet_name="prediction_result"):
    """
    Convert a DataFrame to Excel binary stream for download
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    output.seek(0)
    return output.getvalue()

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["Single Prediction", "Batch File Prediction"])

# =========================
# Single prediction
# =========================
with tab1:
    st.subheader("Please enter the feature values for a single sample")

    cols = st.columns(3)
    input_values = {}

    for i, (feat, default) in enumerate(zip(feature_names, default_values)):
        with cols[i % 3]:
            input_values[feat] = st.number_input(
                label=feat,
                value=float(default),
                format="%.4f"
            )

    if st.button("Start Single Prediction", type="primary"):
        try:
            # Build input in fixed feature order
            X_input = pd.DataFrame(
                [[input_values[feat] for feat in feature_names]],
                columns=feature_names
            )

            st.write("### Original Input Data")
            st.dataframe(X_input)

            # Standardize
            X_scaled = scaler.transform(X_input)

            st.write("### Standardized Data")
            st.dataframe(pd.DataFrame(X_scaled, columns=feature_names))

            # Predict
            prediction = ensemble_model.predict(X_scaled)

            result_single = X_input.copy()
            result_single["prediction"] = prediction

            st.success(f"Prediction Result: {prediction[0]:.6f}")

            st.write("### Single Prediction Result Table")
            st.dataframe(result_single)

            # Download single prediction Excel
            single_excel = dataframe_to_excel_bytes(result_single, sheet_name="single_prediction")
            st.download_button(
                label="Download Single Prediction Result Excel",
                data=single_excel,
                file_name="single_prediction_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

            # Show model weights if available
            if hasattr(ensemble_model, "get_model_weights"):
                with st.expander("View Ensemble Model Weights"):
                    st.json(ensemble_model.get_model_weights())

        except Exception as e:
            st.error(f"Single prediction failed: {e}")

# =========================
# Batch file prediction
# =========================
with tab2:
    st.subheader("Upload a CSV or XLSX file for batch prediction")
    st.write("The uploaded file must contain the following column names exactly:")
    st.code(", ".join(feature_names))

    uploaded_file = st.file_uploader(
        "Select a CSV or XLSX file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:
        try:
            # Detect file type and read
            if uploaded_file.name.endswith(".csv"):
                df_uploaded = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".xlsx"):
                df_uploaded = pd.read_excel(uploaded_file)
            else:
                st.error("Only csv and xlsx files are supported.")
                st.stop()

            st.write("### Uploaded Data Preview")
            st.dataframe(df_uploaded.head())

            # Check required columns
            missing_cols = [col for col in feature_names if col not in df_uploaded.columns]
            if missing_cols:
                st.error(f"The file is missing required columns: {missing_cols}")
            else:
                if st.button("Start Batch Prediction", type="primary"):
                    result_df, X_scaled = predict_dataframe(
                        df_uploaded, scaler, ensemble_model, feature_names
                    )

                    st.success(f"Batch prediction completed successfully. Total records predicted: {len(result_df)}")

                    st.write("### Prediction Result Preview")
                    st.dataframe(result_df.head())

                    # Export as Excel
                    excel_data = dataframe_to_excel_bytes(
                        result_df,
                        sheet_name="batch_prediction"
                    )

                    st.download_button(
                        label="Download Batch Prediction Result Excel",
                        data=excel_data,
                        file_name="batch_prediction_result.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

                    st.info("After clicking download, your browser will allow you to choose the destination folder on your computer.")

        except Exception as e:
            st.error(f"Batch prediction failed: {e}")

