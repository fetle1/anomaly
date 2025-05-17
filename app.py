# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
        
        # Append the result for this row
        anomalies.append(", ".join(issues) if issues else "Normal")

    # Add result column to original dataframe
    df["Rule-Based Result"] = anomalies
    return df, anomaly_locations
def build_and_train_imputation_network(input_cols, output_cols, complete_data):
  """
  Builds and trains a neural network for a specific missing value pattern.

  Args:
    input_cols (list): List of column names for the input features (non-missing).
    output_cols (list): List of column names for the output features (missing).
    complete_data (pd.DataFrame): DataFrame containing only complete cases.

  Returns:
    tensorflow.keras.models.Sequential: The trained neural network.
    sklearn.preprocessing.MinMaxScaler: Scaler used for input features.
    sklearn.preprocessing.MinMaxScaler: Scaler used for output features.
  """
  # Ensure columns exist in the complete data
  if not all(col in complete_data.columns for col in input_cols + output_cols):
    raise ValueError("Input or output columns not found in the complete dataset.")

  X_train = complete_data[input_cols].values
  y_train = complete_data[output_cols].values

  # Scale data to 0-1 range
  input_scaler = MinMaxScaler()
  output_scaler = MinMaxScaler()
  X_train_scaled = input_scaler.fit_transform(X_train)
  y_train_scaled = output_scaler.fit_transform(y_train)

  # Build the neural network
  model = Sequential()
  model.add(Dense(64, activation='relu', input_shape=(len(input_cols),)))
  model.add(Dense(32, activation='relu'))
  model.add(Dense(len(output_cols), activation='sigmoid')) # Sigmoid for output between 0 and 1

  # Compile and train the model
  model.compile(optimizer='adam', loss='mse')
  model.fit(X_train_scaled, y_train_scaled, epochs=100, batch_size=32, verbose=0) # Reduced verbosity

  return model, input_scaler, output_scaler

def impute_missing_values_with_nn(df):
  """
  Imputes missing values using trained neural networks for different patterns.

  Args:
    df (pd.DataFrame): DataFrame with missing values to impute.

  Returns:
    pd.DataFrame: DataFrame with missing values imputed.
  """
  # Step 1: Collect complete cases
  complete_set = df.dropna().copy()

  # Step 2: Collect incomplete cases
  incomplete_set = df[df.isnull().any(axis=1)].copy()

  if incomplete_set.empty:
    print("No missing values to impute.")
    return df

  # Step 3 & 4: Construct and train networks for each missing pattern
  trained_networks = {}
  for index, row in incomplete_set.iterrows():
    missing_cols = row.index[row.isnull()].tolist()
    non_missing_cols = row.index[row.notnull()].tolist()

    # Define a pattern key (e.g., sorted tuple of missing columns)
    pattern_key = tuple(sorted(missing_cols))

    if pattern_key not in trained_networks:
      print(f"Training network for pattern: missing {pattern_key}, non-missing {tuple(sorted(non_missing_cols))}")
      try:
        model, input_scaler, output_scaler = build_and_train_imputation_network(non_missing_cols, missing_cols, complete_set)
        trained_networks[pattern_key] = {
            'model': model,
            'input_scaler': input_scaler,
            'output_scaler': output_scaler,
            'input_cols': non_missing_cols,
            'output_cols': missing_cols
        }
      except ValueError as e:
          print(f"Skipping pattern {pattern_key} due to error: {e}")
          continue # Skip this pattern if columns are not in complete data

  # Step 5: Use trained networks to impute missing values
  imputed_df = df.copy()
  for index, row in incomplete_set.iterrows():
    missing_cols = row.index[row.isnull()].tolist()
    non_missing_cols = row.index[row.notnull()].tolist()
    pattern_key = tuple(sorted(missing_cols))

    if pattern_key in trained_networks:
      network_info = trained_networks[pattern_key]
      model = network_info['model']
      input_scaler = network_info['input_scaler']
      output_scaler = network_info['output_scaler']
      network_input_cols = network_info['input_cols']
      network_output_cols = network_info['output_cols']

      # Ensure input columns match the network's input columns
      if set(non_missing_cols) == set(network_input_cols):
          input_data = row[non_missing_cols].values.reshape(1, -1)
          input_data_scaled = input_scaler.transform(input_data)

          predicted_scaled = model.predict(input_data_scaled)
          predicted_original_scale = output_scaler.inverse_transform(predicted_scaled)

          # Impute the missing values in the original dataframe copy
          for i, col in enumerate(missing_cols):
               # Ensure the column is one of the network's output columns and handle order
               if col in network_output_cols:
                    # Find the index of the column in the network's output columns
                    col_index_in_network_output = network_output_cols.index(col)
                    imputed_df.loc[index, col] = predicted_original_scale[0, col_index_in_network_output]
      else:
          print(f"Input columns for row {index} do not match the trained network's input columns for pattern {pattern_key}. Skipping imputation for this row.")


  return imputed_df
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
st.title(" Anomaly Detection in Health Data")

# App Description
st.markdown("Upload your health-related CSV file and detect anomalies.")

# Upload CSV file with type selection
uploaded_file = st.file_uploader(" Upload a CSV file", type=["csv"])

data_type = st.selectbox(
    "Select data type:",
    ("Cross-Sectional", "Longitudinal", "Time-Series")
)

if uploaded_file:
    try:
        # Read uploaded CSV file
        df = pd.read_csv(uploaded_file)

        # Show preview of data
        st.subheader(" Preview of Uploaded Data")
        st.dataframe(df.head())

        # 1. Missing Value Analysis
        st.subheader(" Missing Value Analysis")
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': missing_percent})
        st.dataframe(missing_info[missing_info['Missing Values'] > 0])

        # 3. Recommendations for Missing Values (without links)
        if st.button("Impute Missing Values with Neural Networks"):
            if 'df' in locals(): # Check if df exists from file upload
                try:
                    df_imputed = impute_missing_values_with_nn(df.copy())
                    st.subheader("Data after Imputation")
                    st.dataframe(df_imputed.head())
                    # You can then proceed with anomaly detection on df_imputed
#                   # Update the df variable to the imputed one for subsequent steps
                    df = df_imputed
                except Exception as e:
                    st.error(f"Error during imputation: {e}")
                    st.exception(e)
            else:
                st.warning("Please upload a CSV file first.")


        # 2. Visualize Distribution (for numerical columns)
        st.subheader(" Data Distribution")
        numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_cols) > 0:
            selected_col = st.selectbox("Select column to visualize distribution:", numerical_cols)
            fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}')
            st.plotly_chart(fig)
        else:
            st.info("No numerical columns found for distribution visualization.")

        # Anomaly Detection based on data type
        st.subheader(f" Running Anomaly Detection ({data_type})")
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
            st.success(f" {data_type} Anomaly Detection Completed")
            st.subheader(" Detected Anomalies Summary")
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