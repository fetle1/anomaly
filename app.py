# app.py

import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.impute import SimpleImputer # For categorical imputation

# ---------------------------------------------
# Anomaly detection functions (Keep your existing functions)
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

        # Rule 3: Missing Values Check (This rule might be less relevant after imputation, but keeping for context)
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
# Neural Network Imputation Functions (Modified)
# ---------------------------------------------

@st.experimental_singleton # Cache the model training process
def build_and_train_imputation_network(input_cols, output_cols, complete_data):
  """
  Builds and trains a neural network for a specific missing value pattern.

  Args:
    input_cols (list): List of column names for the input features (non-missing).
    output_cols (list): List of column names for the output features (missing).
    complete_data (pd.DataFrame): DataFrame containing only complete cases.

  Returns:
    tuple: A tuple containing:
      - tensorflow.keras.models.Sequential: The trained neural network.
      - sklearn.preprocessing.MinMaxScaler: Scaler used for input features.
      - sklearn.preprocessing.MinMaxScaler: Scaler used for output features.
  """
  # Ensure columns exist and are numerical in the complete data
  numerical_complete_data = complete_data.select_dtypes(include=np.number)
  input_cols_numerical = [col for col in input_cols if col in numerical_complete_data.columns]
  output_cols_numerical = [col for col in output_cols if col in numerical_complete_data.columns]

  if not input_cols_numerical or not output_cols_numerical:
      return None, None, None # Cannot train if no numerical input or output columns

  try:
      X_train = numerical_complete_data[input_cols_numerical].values
      y_train = numerical_complete_data[output_cols_numerical].values

      # Scale data to 0-1 range
      input_scaler = MinMaxScaler()
      output_scaler = MinMaxScaler()
      X_train_scaled = input_scaler.fit_transform(X_train)
      y_train_scaled = output_scaler.fit_transform(y_train)

      # Build the neural network
      model = Sequential()
      model.add(Dense(64, activation='relu', input_shape=(len(input_cols_numerical),)))
      model.add(Dense(32, activation='relu'))
      model.add(Dense(len(output_cols_numerical), activation='sigmoid')) # Sigmoid for output between 0 and 1

      # Compile and train the model
      model.compile(optimizer='adam', loss='mse')
      model.fit(X_train_scaled, y_train_scaled, epochs=50, batch_size=32, verbose=0) # Reduced epochs for faster demo

      return model, input_scaler, output_scaler

  except Exception as e:
      st.error(f"Error building or training network for pattern (missing: {output_cols}, non-missing: {input_cols}): {e}")
      return None, None, None


def impute_missing_values_with_nn(df):
  """
  Imputes missing values using trained neural networks for different patterns (numerical)
  and mode imputation for categorical values.

  Args:
    df (pd.DataFrame): DataFrame with missing values to impute.

  Returns:
    pd.DataFrame: DataFrame with missing values imputed.
  """
  imputed_df = df.copy()

  # Identify numerical and categorical columns
  numerical_cols = imputed_df.select_dtypes(include=np.number).columns.tolist()
  categorical_cols = imputed_df.select_dtypes(include=['object', 'category']).columns.tolist()

  # Step 1: Collect complete cases (only considering numerical columns for NN training)
  complete_set_numerical = imputed_df[numerical_cols].dropna().copy()

  # Step 2: Collect incomplete cases (for both numerical and categorical)
  incomplete_set = imputed_df[imputed_df.isnull().any(axis=1)].copy()

  if incomplete_set.empty:
    st.info("No missing values to impute.")
    return df

  st.subheader("🧠 Imputing Missing Values with Neural Networks and Mode")

  # Train networks for each missing pattern in numerical columns
  trained_networks = {}
  missing_patterns_to_train = {}

  for index, row in incomplete_set.iterrows():
      missing_numerical_cols = [col for col in numerical_cols if pd.isnull(row[col])]
      non_missing_numerical_cols = [col for col in numerical_cols if pd.notnull(row[col])]

      if missing_numerical_cols: # Only consider patterns with missing numerical values
          pattern_key = tuple(sorted(missing_numerical_cols))
          if pattern_key not in missing_patterns_to_train:
              missing_patterns_to_train[pattern_key] = {
                  'missing_cols': missing_numerical_cols,
                  'non_missing_cols': non_missing_numerical_cols
              }

  # Step 3 & 4: Construct and train networks
  for pattern_key, cols_info in missing_patterns_to_train.items():
        st.write(f"Training network for pattern: missing {pattern_key}")
        model, input_scaler, output_scaler = build_and_train_imputation_network(
            cols_info['non_missing_cols'],
            cols_info['missing_cols'],
            complete_set_numerical
        )
        if model:
            trained_networks[pattern_key] = {
                'model': model,
                'input_scaler': input_scaler,
                'output_scaler': output_scaler,
                'input_cols': cols_info['non_missing_cols'],
                'output_cols': cols_info['missing_cols']
            }
        else:
            st.warning(f"Could not train network for pattern {pattern_key}. Skipping imputation for this pattern.")


  # Step 5: Use trained networks to impute missing numerical values and mode for categorical
  for index, row in incomplete_set.iterrows():
      # Impute Numerical Missing Values
      missing_numerical_cols = [col for col in numerical_cols if pd.isnull(row[col])]
      non_missing_numerical_cols = [col for col in numerical_cols if pd.notnull(row[col])]
      pattern_key = tuple(sorted(missing_numerical_cols))

      if pattern_key in trained_networks:
          network_info = trained_networks[pattern_key]
          model = network_info['model']
          input_scaler = network_info['input_scaler']
          output_scaler = network_info['output_scaler']
          network_input_cols = network_info['input_cols']
          network_output_cols = network_info['output_cols']

          # Ensure input columns match the network's input columns
          if set(non_missing_numerical_cols) == set(network_input_cols):
              try:
                  input_data = row[non_missing_numerical_cols].values.reshape(1, -1)
                  input_data_scaled = input_scaler.transform(input_data)

                  predicted_scaled = model.predict(input_data_scaled)
                  predicted_original_scale = output_scaler.inverse_transform(predicted_scaled)

                  # Impute the missing numerical values
                  for i, col in enumerate(missing_numerical_cols):
                       if col in network_output_cols:
                            col_index_in_network_output = network_output_cols.index(col)
                            imputed_df.loc[index, col] = predicted_original_scale[0, col_index_in_network_output]
              except Exception as e:
                  st.warning(f"Error during numerical imputation for row {index} with pattern {pattern_key}: {e}")
          else:
               st.warning(f"Input columns for row {index} do not match the trained network's input columns for pattern {pattern_key}. Skipping numerical imputation for this row.")


      # Impute Categorical Missing Values with Mode
      missing_categorical_cols = [col for col in categorical_cols if pd.isnull(row[col])]
      if missing_categorical_cols:
          for col in missing_categorical_cols:
              # Calculate mode excluding NaN values
              mode_value = imputed_df[col].mode()
              if not mode_value.empty:
                  imputed_df.loc[index, col] = mode_value[0]
              else:
                  st.warning(f"Could not find mode for categorical column '{col}'. Leaving missing value as NaN for row {index}.")

  return imputed_df

# ---------------------------------------------
# Streamlit App UI (Modified)
# ---------------------------------------------
# Page configuration
st.set_page_config(page_title="Anomaly Detection App", layout="wide")

# App Title
st.title(" Anomaly Detection in Health Data")

# App Description
st.markdown("Upload your health-related CSV file and detect anomalies.")

# Upload CSV file with type selection
uploaded_file = st.file_uploader(" Upload a CSV file", type=["csv"])

# Initialize df in session state if not present
if 'df' not in st.session_state:
    st.session_state['df'] = None

data_type = st.selectbox(
    "Select data type:",
    ("Cross-Sectional", "Longitudinal", "Time-Series")
)

if uploaded_file:
    # Read uploaded CSV file and store in session state
    try:
        # Only read if a new file is uploaded or if df is None
        if st.session_state['df'] is None or uploaded_file.getvalue() != st.session_state['uploaded_file_content']:
             st.session_state['df'] = pd.read_csv(uploaded_file)
             st.session_state['uploaded_file_content'] = uploaded_file.getvalue()
             st.success("File uploaded successfully!")

        df = st.session_state['df'] # Use df from session state

        # Show preview of data
        st.subheader(" Preview of Uploaded Data")
        st.dataframe(df.head())

        # 1. Missing Value Analysis
        st.subheader(" Missing Value Analysis")
        missing_values = df.isnull().sum()
        missing_percent = (missing_values / len(df)) * 100
        missing_info = pd.DataFrame({'Missing Values': missing_values, 'Percentage (%)': missing_percent})

        missing_info_display = missing_info[missing_info['Missing Values'] > 0]
        if not missing_info_display.empty:
             st.dataframe(missing_info_display)

             # 3. Recommendations for Missing Values
             st.subheader(" Recommendations for Missing Values")
             st.markdown("""
             Based on the percentage of missing values, consider these general approaches:

             *   **Low percentage (e.g., < 5%):** Consider simple imputation methods like mean, median, or mode. You might also consider dropping rows with missing values if the dataset is large enough.
             *   **Moderate percentage (e.g., 5% - 20%):** More sophisticated imputation methods like K-Nearest Neighbors (KNN) imputation or multiple imputation might be suitable. Analyze the nature of missingness (e.g., completely at random, at random, not at random).
             *   **High percentage (e.g., > 20%):** Imputation becomes less reliable. Consider if the feature is essential. You might need to collect more data or use models that can handle missing values.
             *   **Before applying any method:** Understand the reason for missingness if possible. Visualization of missing data patterns can be helpful.
             """)

             if st.button("Impute Missing Values with Neural Networks and Mode"):
                 with st.spinner("Imputing missing values..."):
                     try:
                         # Update the df in session state after imputation
                         st.session_state['df'] = impute_missing_values_with_nn(df.copy())
                         df = st.session_state['df'] # Update local df variable
                         st.success("Missing values imputed successfully!")
                         st.subheader("Data after Imputation")
                         st.dataframe(df.head())
                         # Re-display missing value analysis after imputation
                         st.subheader(" Missing Value Analysis After Imputation")
                         missing_values_after = df.isnull().sum()
                         missing_percent_after = (missing_values_after / len(df)) * 100
                         missing_info_after = pd.DataFrame({'Missing Values': missing_values_after, 'Percentage (%)': missing_percent_after})
                         missing_info_after_display = missing_info_after[missing_info_after['Missing Values'] > 0]
                         if not missing_info_after_display.empty:
                              st.dataframe(missing_info_after_display)
                         else:
                              st.info("No missing values remaining after imputation.")

                     except Exception as e:
                         st.error(f"Error during imputation: {e}")
                         st.exception(e)
         else:
              st.info("No missing values found in the uploaded data.")


        # 2. Visualize Distribution (for numerical columns - using the potentially imputed df)
        st.subheader(" Data Distribution")
        numerical_cols = df.select_dtypes(include=np.number).columns
        if len(numerical_cols) > 0:
            selected_col = st.selectbox("Select column to visualize distribution:", numerical_cols)
            fig = px.histogram(df, x=selected_col, title=f'Distribution of {selected_col}')
            st.plotly_chart(fig)
        else:
            st.info("No numerical columns found for distribution visualization.")

        # Anomaly Detection based on data type (using the potentially imputed df)
        st.subheader(f" Running Anomaly Detection ({data_type})")
        result_df = df.copy() # Start with the current state of the dataframe (potentially imputed)
        anomaly_locations = []

        if st.button(f"Run Anomaly Detection for {data_type}"):
            with st.spinner(f"Running {data_type} anomaly detection..."):
                try:
                    if data_type == "Cross-Sectional":
                        result_df, anomaly_locations = detect_rule_based_anomalies(df.copy()) # Pass a copy of the potentially imputed df
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
                        st.subheader(" Specific Anomaly Locations (Row, Column, Issue)")
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
                    st.error(f"Error during anomaly detection: {e}")
                    st.exception(e)


    except Exception as e:
        st.error(f" Error processing file: {e}")
        st.exception(e) # Display full error traceback for debugging
else:
    st.info(" Please upload a CSV file to get started.")