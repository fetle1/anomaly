import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import random
import os
import re # Import regular expressions for cleaning

# Set Streamlit page config
st.set_page_config(page_title="Health Data Analyzer", layout="wide")

# Custom UI Styling (optional, keep it simple for now)
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)


st.title("Health Data Analyzer")

# Use tabs for navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs([" Upload Data", " Data Overview", " Data Preprocessing", " Missing Data Analysis", " Data Imputation"])


with tab1:
    st.subheader("Upload your data")
    uploaded_file = st.file_uploader("Upload your dataset (CSV, XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.session_state["df"] = df.copy() # Store a copy of the original data
            st.session_state["df_processed"] = None # Reset processed data on new upload
            st.session_state["df_imputed"] = None # Reset imputed data on new upload
            st.success(" Data uploaded successfully!")
            st.write("First 5 rows of the uploaded data:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error uploading file: {e}")

with tab2:
    st.subheader("Data Overview")

    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)

            st.session_state["df"] = df.copy()
            st.session_state["df_processed"] = None
            st.session_state["df_imputed"] = None
            st.success("Data uploaded successfully!")

        except Exception as e:
            st.error(f"Error uploading file: {e}")

    # Only proceed if a DataFrame is available in session
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state.get("df_processed") or st.session_state.get("df")

        if df is not None:
            st.subheader("Data Preview")
            st.dataframe(df.head())

            st.subheader("Data Types")
            st.write(df.dtypes)

            st.subheader("Summary Statistics")
            st.write(df.describe(include='all'))

            st.subheader("Column Distribution Visualization")
            column = st.selectbox("Select a column to visualize", df.columns, key='overview_column_select')

            if column:
                fig, ax = plt.subplots()
                if pd.api.types.is_numeric_dtype(df[column]):
                    sns.histplot(df[column].dropna(), kde=True, ax=ax)
                    ax.set_title(f'Distribution of {column}')
                elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
                    series = df[column].dropna().astype(str)
                    if not series.empty:
                        sns.countplot(y=series, ax=ax)
                        ax.set_title(f'Count of {column}')
                    else:
                        st.warning(f"No non-null values in column '{column}' to plot.")
                else:
                    st.info(f"Column '{column}' has a non-visualizable dtype.")
                st.pyplot(fig)
            else:
                st.info("Please select a column to visualize.")
        else:
            st.warning("No valid data to show.")
    else:
        st.warning("Please upload data first in the 'Upload Data' section.")

with tab3:  # Preprocessing
    st.sidebar.subheader("Preprocessing Options")

    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"].copy() # Start with a fresh copy of the original data

        st.write("Current Data Head:")
        st.dataframe(df.head())
        st.write("Current Data Types:")
        st.write(df.dtypes)

        # Preprocessing Steps Configuration
        st.sidebar.subheader("Preprocessing Options")
        process_bp = st.sidebar.checkbox("Split 'BP' column (e.g., '120/80')", value=True)
        clean_strings = st.sidebar.checkbox("Clean String Columns (remove spaces, empty strings, etc.)", value=True)
        convert_to_category = st.sidebar.checkbox("Convert suitable object columns to 'category'", value=True)
        identifier_cols_input = st.sidebar.text_input("Enter identifier columns (comma-separated)", "clinical_id,first_name,last_name,patient_id")
        identifier_cols = [col.strip() for col in identifier_cols_input.split(',')]
        columns_to_drop_input = st.sidebar.text_input("Enter columns to drop (comma-separated)", "")
        columns_to_drop = [col.strip() for col in columns_to_drop_input.split(',') if col.strip()]


        if st.button("Apply Preprocessing"):

            # Step 1: Handle 'BP' column if it exists and is object type
            if process_bp and 'BP' in df.columns and (pd.api.types.is_object_dtype(df['BP']) or pd.api.types.is_string_dtype(df['BP'])):
                st.info("Processing 'BP' column...")
                try:
                    # Ensure the 'BP' column is treated as string type first
                    df['BP'] = df['BP'].astype(str).str.strip() # Add strip to remove leading/trailing spaces

                    # Identify rows that can be split
                    splittable_rows = df['BP'].str.contains('/', na=False)

                    # Split the 'BP' column into two new columns: 'Systolic_BP' and 'Diastolic_BP'
                    # Apply split only to rows that contain '/'
                    split_bp = df.loc[splittable_rows, 'BP'].str.split('/', expand=True)

                    # Initialize new columns with NaN
                    df['Systolic_BP'] = np.nan
                    df['Diastolic_BP'] = np.nan

                    # Assign split values
                    if 0 in split_bp.columns:
                        df.loc[splittable_rows, 'Systolic_BP'] = pd.to_numeric(split_bp[0], errors='coerce')
                    if 1 in split_bp.columns:
                         df.loc[splittable_rows, 'Diastolic_BP'] = pd.to_numeric(split_bp[1], errors='coerce')


                    # Drop the original 'BP' column
                    df = df.drop('BP', axis=1)
                    st.success(" 'BP' column split into 'Systolic_BP' and 'Diastolic_BP'")
                except Exception as e:
                    st.error(f"Error processing 'BP' column: {e}")
                    st.warning("Skipping 'BP' processing due to error.")

            # Step 2: Clean String Columns
            if clean_strings:
                st.info("Cleaning string columns...")
                for col in df.columns:
                    # Check if the column is of object or category dtype
                    if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                         try:
                            # Convert to string first to handle mixed types gracefully
                            df[col] = df[col].astype(str)
                            # Remove leading/trailing whitespace
                            df[col] = df[col].str.strip()
                            # Replace empty strings with NaN
                            df[col] = df[col].replace('', np.nan)
                            # Replace string 'nan' with actual NaN (if it exists as a string)
                            df[col] = df[col].replace('nan', np.nan)
                            # Example: Convert to lowercase (optional)
                            # df[col] = df[col].str.lower()
                         except Exception as e:
                            st.warning(f"Could not clean string column '{col}': {e}")
                st.success("String columns cleaned.")


            # Step 3: Convert suitable object columns to 'category' type
            if convert_to_category:
                st.info("Converting suitable object columns to 'category' type...")
                 # Iterate through all columns
                for col in df.columns:
                    # Check if the column's data type is 'object' and not in the identifier list
                    if pd.api.types.is_object_dtype(df[col]) and col not in identifier_cols:
                        try:
                             # Check the number of unique values. If not too many, convert to category.
                             # Threshold can be adjusted (e.g., < 50 or < 100)
                            if df[col].nunique() < 50:
                                df[col] = df[col].astype('category')
                            else:
                                st.info(f"Column '{col}' has too many unique values ({df[col].nunique()}), skipping conversion to category.")
                        except Exception as e:
                             st.warning(f"Could not convert column '{col}' to category: {e}")
                st.success("Object columns (excluding identifiers and those with too many unique values) converted to category dtype.")


            # Step 4: Drop specified columns
            if columns_to_drop:
                 st.info(f"Dropping columns: {', '.join(columns_to_drop)}")
                 existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
                 if existing_columns_to_drop:
                    df = df.drop(columns=existing_columns_to_drop, errors='ignore')
                    st.success(f"Dropped columns: {', '.join(existing_columns_to_drop)}")
                 else:
                    st.warning("None of the specified columns to drop were found in the DataFrame.")


            st.write("Preprocessing complete. Updated DataFrame head:")
            st.dataframe(df.head())
            st.write("Updated Data Types:")
            st.write(df.dtypes)
            st.write("Updated Missing Value Count:")
            st.write(df.isnull().sum())

            # Store the processed DataFrame in session state
            st.session_state["df_processed"] = processed_df  # after preprocessing is applied


    else:
        st.warning("Please upload data first in the 'Upload Data' tab.")


with tab4:
    st.subheader("Missing Data Analysis")
     # Use df from session state, prioritize processed if available
    if "df_processed" in st.session_state and st.session_state["df_processed"] is not None:
        df = st.session_state["df_processed"]
        st.info("Analyzing missing data for the processed dataset.")
    elif "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.warning("Data has not been preprocessed yet. Displaying missing data analysis for the raw data.")
    else:
        st.warning("Please upload data first in the 'Upload Data' tab.")
        df = None # Ensure df is None if no data is loaded
    if "df_processed" not in st.session_state or st.session_state["df_processed"] is None:
        st.warning("Please preprocess the data first in the 'Data Preprocessing' tab before imputation.")
        st.stop()

    if df is not None:
        st.subheader("📊 Missing Data Summary")
        missing_pct = df.isnull().mean() * 100
        missing_summary = missing_pct[missing_pct > 0].sort_values(ascending=False)

        if not missing_summary.empty:
            st.write("Percentage of missing values per column:")
            st.dataframe(missing_summary.reset_index().rename(columns={'index': 'Column', 0: 'Missing Percentage (%)'}))

            st.subheader(" Missingness Mechanism (Heuristic)")
            st.info("Note: This is a heuristic analysis and requires domain knowledge for definitive determination.")
            for col in missing_summary.index:
                st.write(f"Column: **{col}**")
                if pd.api.types.is_numeric_dtype(df[col]):
                    numeric_df = df.select_dtypes(include=np.number)
                    # Ensure the column exists in the numeric subset before calculating correlation
                    if col in numeric_df.columns and len(numeric_df.columns) > 1: # Need at least two columns for correlation
                        # Calculate absolute correlation with other numeric columns
                        correlations = numeric_df.corr()[col].drop(col).abs()
                        if not correlations.empty:
                             corr = correlations.max()
                             if corr > 0.3:
                                 st.write("Likely MAR (Missing At Random) - High correlation with other numeric features.")
                             else:
                                 st.write("Possibly MCAR (Missing Completely At Random) - Low correlation with other numeric features.")
                        else:
                             st.write("Cannot determine mechanism for numeric column with no other numeric columns for correlation analysis.")

                    else:
                        st.write("Cannot determine mechanism for numeric column (no other numeric columns or only one numeric column).")

                elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                    st.write("MNAR suspected (Missing Not At Random) - Categorical column, mechanism is often related to the missing value itself.")
                else:
                    st.write("Mechanism determination not implemented for this data type.")
        else:
            st.success("No missing data found in the dataset.")


   
with tab5:  # Imputation
    st.sidebar.subheader("Imputation Options")
    # ... only imputation sidebar widgets here ...
imputation_strategy = st.sidebar.radio(
    "Choose Imputation Strategy",
    (
        "Default Strategy (rules by missing %)",
        "Mean/Mode for all",
        "Median/Mode for all",
        "KNN for all",
        "Drop rows with missing values"
    )
)
if imputation_strategy == "Default Strategy (rules by missing %)":
    # Use df from session state, prioritize processed if available
    if "df_processed" in st.session_state and st.session_state["df_processed"] is not None:
        df = st.session_state["df_processed"].copy() # Work on a copy for imputation
        st.info("Applying imputation to the processed dataset.")

        st.sidebar.subheader("Imputation Options")
        drop_high_missing_cols = st.sidebar.checkbox("Drop columns with > 50% missing", value=True)
        drop_low_missing_rows = st.sidebar.checkbox("Drop rows with < 1% missing in a column (if that column has < 5% total missing)", value=True)
        impute_low_missing = st.sidebar.checkbox("Impute missing < 5% (Mean/Mode)", value=True)
        impute_moderate_missing = st.sidebar.checkbox("Impute missing 5%-50% (KNN)", value=True)
        knn_neighbors = st.sidebar.slider("KNN n_neighbors", 2, 10, 5)


        if st.button("Apply Imputation"):
            st.write("Starting imputation process...")
            initial_cols = df.columns.tolist() # Track columns before dropping

            for col in initial_cols: # Iterate over initial columns to handle dropped ones
                 if col not in df.columns: # Skip if column was dropped in a previous step
                     continue

                 missing_pct = df[col].isnull().mean() * 100
                 if missing_pct == 0:
                     continue

                 st.write(f" Processing column: `{col}` ({missing_pct:.1f}% missing)")

                 # Strategy 1: Drop column/rows if too much missing data
                 if drop_high_missing_cols and missing_pct > 50:
                      st.warning(f"Column `{col}` has {missing_pct:.1f}% missing values. Dropping the column.")
                      df = df.drop(col, axis=1)
                      st.success(f"Dropped column: `{col}`")
                      continue # Move to the next column

                 # Strategy 2: Drop rows if a small percentage of rows have missing data in a column
                 if drop_low_missing_rows and missing_pct < 5:
                      missing_rows_count = df[col].isnull().sum()
                      # Check if dropping these rows is a significant loss of data (e.g., less than 1% of total rows)
                      if missing_rows_count / len(df) < 0.01:
                           initial_row_count = len(df)
                           df = df.dropna(subset=[col])
                           rows_dropped = initial_row_count - len(df)
                           if rows_dropped > 0:
                                st.warning(f"Dropped {rows_dropped} rows with missing values in `{col}` (less than 1% of data).")
                           else:
                                st.info(f"No rows dropped for `{col}` despite low missing percentage.")
                           continue # Move to the next column
                      else:
                            st.info(f"Column `{col}` has {missing_pct:.1f}% missing. Proceeding with imputation.")


                 # Strategy 3: Impute for small missing percentage (if not dropped rows)
                 if impute_low_missing and missing_pct < 5 and drop_low_missing_rows == False:
                      if pd.api.types.is_numeric_dtype(df[col]):
                          df[col].fillna(df[col].mean(), inplace=True)
                          st.success(f"Filled missing values in `{col}` with mean.")
                      elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                          mode_val = df[col].mode()
                          if not mode_val.empty:
                              df[col].fillna(mode_val[0], inplace=True)
                              st.success(f"Filled missing values in `{col}` with mode.")
                          else:
                              st.warning(f"Could not find mode for column `{col}`. Skipping imputation.")
                      else:
                           st.warning(f"Imputation strategy not defined for data type of column `{col}` (missing < 5%).")

                 # Strategy 4: KNN Imputation for moderate missing percentage
                 elif impute_moderate_missing and missing_pct >= 5 and missing_pct <= 50:
                      st.info(f"Column `{col}` has {missing_pct:.1f}% missing. Using KNN Imputation with k={knn_neighbors}.")
                      if pd.api.types.is_numeric_dtype(df[col]):
                          try:
                              imputer = KNNImputer(n_neighbors=knn_neighbors)
                              df[[col]] = imputer.fit_transform(df[[col]])
                              st.success(f"Imputed missing values in `{col}` using KNN.")
                          except Exception as e:
                              st.error(f"Error during KNN imputation for column `{col}`: {e}")
                              st.warning(f"Skipping KNN imputation for `{col}`.")

                      elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                          try:
                              # Convert to string to handle potential NaN and other types
                              cat_data = df[col].astype(str)

                              # Handle unseen categories by adding a placeholder before fitting LabelEncoder
                              unique_values = cat_data.dropna().unique().tolist()
                              unique_values_for_encoding = unique_values + ['_placeholder_for_knn'] # Add a temporary placeholder

                              le = LabelEncoder()
                              le.fit(unique_values_for_encoding)

                              encoded_data = le.transform(cat_data)

                              imputer = KNNImputer(n_neighbors=knn_neighbors)
                              imputed_encoded_data = imputer.fit_transform(encoded_data.reshape(-1, 1))

                              imputed_encoded_data_int = np.round(imputed_encoded_data).flatten().astype(int)

                              # Ensure imputed values are within the range of fitted categories
                              max_encoded_val = len(le.classes_) - 1
                              imputed_encoded_data_int = np.clip(imputed_encoded_data_int, 0, max_encoded_val)

                              # Inverse transform
                              df[col] = le.inverse_transform(imputed_encoded_data_int)

                              # Replace the placeholder back to NaN if it was imputed to that value
                              df[col] = df[col].replace('_placeholder_for_knn', np.nan)

                              st.success(f"Imputed missing values in categorical column `{col}` using Label Encoding and KNN.")

                          except Exception as e:
                              st.error(f"Error during categorical imputation for column `{col}`: {e}")
                              st.warning(f"Skipping categorical imputation for `{col}`.")
                      else:
                           st.warning(f"Imputation strategy not defined for data type of column `{col}` (missing >= 5% and <= 50%).")
                 else:
                     st.info(f"Column `{col}` ({missing_pct:.1f}% missing) was not imputed based on current settings.")
elif imputation_strategy == "Mean/Mode for all":
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
elif imputation_strategy == "Median/Mode for all":
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
elif imputation_strategy == "KNN for all":
    imputer = KNNImputer(n_neighbors=knn_neighbors)
    df = pd.DataFrame(imputer.fit_transform(df.select_dtypes(include=[np.number])), columns=df.select_dtypes(include=[np.number]).columns)
elif imputation_strategy == "Drop rows with missing values":
    df = df.dropna()
# UI/UX: Add a “Next” Button to Navigate Tabs
#To avoid scrolling to the top, add a navigation aid at the bottom:

##python
#Copy code
col1, col2 = st.columns([8, 2])
with col2:
    if st.button("➡️ Next: Modeling", key="next_button"):
        st.experimental_set_query_params(tab="modeling") 

        st.write("Imputation process complete. Checking for remaining missing values:")
        remaining_missing = df.isnull().sum()
        if remaining_missing.sum() == 0:
            st.success("No missing values remaining after imputation.")
        else:
            st.warning("Some missing values may remain depending on imputation strategies applied.")
            st.write(remaining_missing[remaining_missing > 0])


            # Store the imputed DataFrame in session state
            st.session_state["df_imputed"] = df

            # Provide a download link for the imputed data
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_string = csv_buffer.getvalue()
            st.download_button(
                label="Download Imputed Data (CSV)",
                data=csv_string,
                file_name="imputed_data.csv",
                mime="text/csv"
            )

    elif "df" in st.session_state and st.session_state["df"] is not None:
         st.warning("Please preprocess the data first in the 'Data Preprocessing' tab before imputation.")
    else:
        st.warning("Please upload data first in the 'Upload Data' tab.")
