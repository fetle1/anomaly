from IPython import get_ipython
from IPython.display import display
# %%
# app.py

import streamlit as st
import pandas as pd
import plotly.express as px

# ---------------------------------------------
# Anomaly detection functions
# ---------------------------------------------

def detect_rule_based_anomalies(df):
    """
    Applies simple rules to detect anomalies in health data.
    Returns the dataframe with a new column 'Rule-Based Result' and a list of anomaly locations.
    """
    anomalies = []
    anomaly_locations = []

    for index, row in df.iterrows():
        issues = []

        # Rule 1: Heart Rate Check
        if 'heart_rate' in df.columns:
            if pd.notnull(row['heart_rate']) and (row['heart_rate'] < 50 or row['heart_rate'] > 120):
                issues.append("Abnormal heart rate")
                anomaly_locations.append({'row': index, 'column': 'heart_rate', 'issue': 'Abnormal heart rate'})

        # Rule 2: Systolic Blood Pressure Check
        if 'bp_sys' in df.columns:
            if pd.notnull(row['bp_sys']) and (row['bp_sys'] < 90 or row['bp_sys'] > 180):
                issues.append("Abnormal systolic BP")
                anomaly_locations.append({'row': index, 'column': 'bp_sys', 'issue': 'Abnormal systolic BP'})

        # Rule 3: Missing Values Check
        if row.isnull().sum() > 0:
            issues.append("Missing values")
            for col in df.columns:
                if pd.isnull(row[col]):
                    anomaly_locations.append({'row': index, 'column': col, 'issue': 'Missing value'})

        # Append the result for this row
        anomalies.append(", ".join(issues) if issues else "Normal")

    # Add result column to original dataframe
    df["Rule-Based Result"] = anomalies
    return df, anomaly_locations

def detect_time_series_anomalies(df):
    """
    Placeholder for time-series anomaly detection.
    You would implement a time-series specific method here, e.g., using Isolation Forest or other time-series models.
    This example uses a simple rule for demonstration.
    """
    st.warning("Time-series anomaly detection is a placeholder. Implement your specific logic here.")
    # Example: detect points significantly different from the previous one in a 'value' column
    anomalies = []
    anomaly_locations = []
    if 'value' in df.columns:
        df['value_diff'] = df['value'].diff().abs()
        threshold = df['value_diff'].quantile(0.95) # Example threshold
        for index, row in df.iterrows():
            if pd.notnull(row['value_diff']) and row['value_diff'] > threshold:
                anomalies.append("Potential time-series anomaly")
                anomaly_locations.append({'row': index, 'column': 'value', 'issue': 'Potential time-series anomaly'})
            else:
                anomalies.append("Normal")
    else:
         anomalies = ["N/A"] * len(df)


    df["Time-Series Result"] = anomalies
    return df, anomaly_locations

def detect_longitudinal_anomalies(df):
    """
    Placeholder for longitudinal anomaly detection.
    You would implement a longitudinal specific method here, e.g., analyzing trends within individuals over time.
    This example uses a simple rule for demonstration.
    """
    st.warning("Longitudinal anomaly detection is a placeholder. Implement your specific logic here.")
    # Example: detect significant changes within an individual's measurements over time (assuming 'patient_id' and 'time')
    anomalies = []
    anomaly_locations = []
    if 'patient_id' in df.columns and 'time' in df.columns and 'value' in df.columns:
        df_sorted = df.sort_values(by=['patient_id', 'time'])
        df_sorted['value_change'] = df_sorted.groupby('patient_id')['value'].diff().abs()
        threshold = df_sorted['value_change'].quantile(0.95) # Example threshold
        for index, row in df_sorted.iterrows():
            if pd.notnull(row['value_change']) and row['value_change'] > threshold:
                 anomalies.append("Potential longitudinal anomaly")
                 anomaly_locations.append({'row': index, 'column': 'value', 'issue': 'Potential longitudinal anomaly for patient_id ' + str(row['patient_id'])})
            else:
                 anomalies.append("Normal")
        # Need to merge back to the original index to keep consistent with the original df
        df = df.merge(df_sorted[['value_change']], left_index=True, right_index=True, how='left')
        df['Longitudinal Result'] = anomalies # This mapping might need adjustment based on your actual logic

    else:
        anomalies = ["N/A"] * len(df)

    df["Longitudinal Result"] = anomalies
    return df, anomaly_locations


# ---------------------------------------------
# Streamlit App UI
# ---------------------------------------------
# Page configuration
st.set_page_config(page_title="Anomaly Detection App", layout="wide")

# App Title
st.title("🧠 Anomaly Detection in Health Data")

# App Description
st.markdown("Upload your health-related CSV file and detect anomalies.")

# Upload CSV file with type selection
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

data_type = st.selectbox(
    "Select data type:",
    ("Cross-Sectional", "Longitudinal", "Time-Series")
)

if uploaded_file:
    try:
        # Read uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Show preview of data
        st.subheader("🔍 Preview of Uploaded Data")
        st.dataframe(df.head())

        # 1. Missing Value Analysis
        st.subheader("📊 Missing Value Analysis")
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': missing_percent})
        st.dataframe(missing_info[missing_info['Missing Values'] > 0])

        # 3. Recommendations for Missing Values (without links)
        if missing_info['Missing Values'].sum() > 0:
            st.subheader("💡 Recommendations for Missing Values")
            st.markdown("""
            Based on the percentage of missing values, consider these general approaches:

            *   **Low percentage (e.g., < 5%):** Consider simple imputation methods like mean, median, or mode. You might also consider dropping rows with missing values if the dataset is large enough.
            *   **Moderate percentage (e.g., 5% - 20%):** More sophisticated imputation methods like K-Nearest Neighbors (KNN) imputation or multiple imputation might be suitable. Analyze the nature of missingness (e.g., completely at random, at random, not at random).
            *   **High percentage (e.g., > 20%):** Imputation becomes less reliable. Consider if the feature is essential. You might need to collect more data or use models that can handle missing values.
            *   **Before applying any method:** Understand the reason for missingness if possible. Visualization of missing data patterns can be helpful.
            """)

        # 2. Visualize Distribution (for numerical columns)
        st.subheader("📈 Data Distribution")
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 0:
            selected_col = st.selectbox("Select column to visualize distribution:", numerical_cols)
            fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}')
            st.plotly_chart(fig)
        else:
            st.info("No numerical columns found for distribution visualization.")

        # Anomaly Detection based on data type
        st.subheader(f"🕵️ Running Anomaly Detection ({data_type})")
        result_df = df.copy()
        anomaly_locations = []

        if st.button(f"Run Anomaly Detection for {data_type}"):
            if data_type == "Cross-Sectional":
                result_df, anomaly_locations = detect_rule_based_anomalies(df.copy()) # Pass a copy to avoid modifying original df
            elif data_type == "Time-Series":
                result_df, anomaly_locations = detect_time_series_anomalies(df.copy()) # Pass a copy
            elif data_type == "Longitudinal":
                result_df, anomaly_locations = detect_longitudinal_anomalies(df.copy()) # Pass a copy

            # Display result table
            st.success(f"✅ {data_type} Anomaly Detection Completed")
            st.subheader("📋 Detected Anomalies Summary")
            st.dataframe(result_df)

            # 4. Display Specific Anomaly Locations
            if anomaly_locations:
                st.subheader("Specific Anomaly Locations (Row, Column, Issue)")
                anomaly_locations_df = pd.DataFrame(anomaly_locations)
                st.dataframe(anomaly_locations_df)
            else:
                st.info("No specific anomalies detected based on the applied methods.")


            # Allow result download
            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label=" Download Results as CSV",
                data=csv,
                file_name=f"{data_type.lower()}_anomalies.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f" Error processing file: {e}")
        st.exception(e) # Display full error traceback for debugging
else:
    st.info("👈 Please upload a CSV file to get started.")