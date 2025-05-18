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

# Set Streamlit page config
st.set_page_config(page_title="Health Data Analyzer", layout="wide")

# Custom UI Styling with gradient background
# Sidebar menu
menu = st.sidebar.selectbox(
    " Hello! what can I do for you?",
    ["Upload Data", "Data Overview", "Data Preprocessing", "Missing Data Analysis", "Data Imputation"]
)

# File uploader
if menu == "Upload Data":
    st.subheader("Upload your data")
    uploaded_file = st.file_uploader("Upload your dataset (CSV, XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.session_state["df"] = df
            st.success(" Data uploaded successfully!")
            st.write("First 5 rows of the uploaded data:")
            st.dataframe(df.head())
        except Exception as e:
            st.error(f"Error uploading file: {e}")


# Data Overview
if menu == "Data Overview":
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.subheader(" Data Preview")
        st.dataframe(df.head())

        st.subheader(" Data Types")
        st.write(df.dtypes)

        st.subheader(" Summary Statistics")
        st.write(df.describe(include='all'))

        st.subheader(" Column Distribution Visualization")
        column = st.selectbox("Select a column to visualize", df.columns)

        color_palette = random.choice(sns.color_palette("Set2", 10))
        fig, ax = plt.subplots()

        # Check column data type more robustly
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.histplot(df[column].dropna(), kde=True, ax=ax, color=color_palette)
            ax.set_title(f'Distribution of {column}')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
        elif pd.api.types.is_object_dtype(df[column]) or pd.api.types.is_categorical_dtype(df[column]):
            if df[column].nunique() < 15: # Increased limit for readability
                # Ensure there are non-null values before plotting pie
                if not df[column].dropna().empty:
                    df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette("Pastel1"))
                    ax.set_ylabel('')
                    ax.set_title(f'Distribution of {column}')
                else:
                    st.warning(f"Column '{column}' has no non-null values to display pie chart.")
            else:
                 # Ensure there are non-null values before plotting countplot
                if not df[column].dropna().empty:
                    sns.countplot(y=df[column], ax=ax, palette=random.choice(["Set1", "Set2", "Set3", "Pastel1", "Pastel2", "viridis", "plasma"])) # Added more palettes
                    ax.set_title(f'Count of {column}')
                    ax.set_xlabel('Count')
                    ax.set_ylabel(column)
                else:
                     st.warning(f"Column '{column}' has no non-null values to display countplot.")
        else:
            st.info(f"Column '{column}' has a data type that is not easily visualized with standard plots.")

        st.pyplot(fig)
    else:
        st.warning("Please upload data first in the 'Upload Data' section.")


# Data Preprocessing
if menu == "Data Preprocessing":
    if "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"].copy() # Work on a copy to avoid modifying the original

        st.subheader("Data Preprocessing")

        # Handle 'BP' column if it exists and is object type
        if 'BP' in df.columns and (pd.api.types.is_object_dtype(df['BP']) or pd.api.types.is_string_dtype(df['BP'])):
            st.write("Processing 'BP' column...")
            try:
                # Ensure the 'BP' column is treated as string type first
                df['BP'] = df['BP'].astype(str)

                # Split the 'BP' column into two new columns: 'Systolic_BP' and 'Diastolic_BP'
                split_bp = df['BP'].str.split('/', expand=True)

                # Use pd.to_numeric to convert the resulting strings to floating-point numbers
                df['Systolic_BP'] = pd.to_numeric(split_bp[0], errors='coerce')
                df['Diastolic_BP'] = pd.to_numeric(split_bp[1], errors='coerce')

                # Drop the original 'BP' column
                df = df.drop('BP', axis=1)
                st.success(" 'BP' column split into 'Systolic_BP' and 'Diastolic_BP'")
            except Exception as e:
                st.error(f"Error processing 'BP' column: {e}")
                st.warning("Skipping 'BP' processing due to error.")

        # Identify identifier columns to exclude from categorical conversion
        identifier_cols = ['clinical_id', 'first_name', 'last_name', 'patient_id'] # Added patient_id as a common identifier

        st.write("Converting suitable object columns to 'category' type...")
        # Create a new DataFrame without the identifier columns for categorical conversion
        # .copy() is important
        df_categorical_subset = df.drop(columns=identifier_cols, errors='ignore').copy()

        # Iterate through the columns of the subset DataFrame
        for col in df_categorical_subset.columns:
            # Check if the column's data type is 'object' and not an identifier
            if pd.api.types.is_object_dtype(df_categorical_subset[col]):
                # Convert the column to 'category' dtype
                df[col] = df[col].astype('category') # Apply conversion to the original df
        st.success("Object columns (excluding identifiers) converted to category dtype.")

        # Optional: Define a list of columns to drop (can be made interactive later)
        columns_to_drop = st.multiselect("Select columns to drop", df.columns)
        if columns_to_drop:
             st.write(f"Dropping columns: {', '.join(columns_to_drop)}")
             # Drop the specified columns
             existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
             df = df.drop(columns=existing_columns_to_drop, errors='ignore')
             st.success(f"Dropped selected columns.")


        st.write("Preprocessing complete. Updated DataFrame head:")
        st.dataframe(df.head())
        st.write("Updated Data Types:")
        st.write(df.dtypes)

        # Store the processed DataFrame in session state
        st.session_state["df_processed"] = df

    else:
        st.warning("Please upload data first in the 'Upload Data' section.")

# Missing Data Analysis
if menu == "Missing Data Analysis":
     # Use df from session state, prioritize processed if available
    if "df_processed" in st.session_state and st.session_state["df_processed"] is not None:
        df = st.session_state["df_processed"]
        st.subheader(" Missing Data Summary")
        missing_pct = df.isnull().mean() * 100
        missing_summary = missing_pct[missing_pct > 0].sort_values(ascending=False)

        if not missing_summary.empty:
            st.write("Percentage of missing values per column:")
            st.write(missing_summary)

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

    elif "df" in st.session_state and st.session_state["df"] is not None:
        df = st.session_state["df"]
        st.warning("Data has not been preprocessed yet. Displaying missing data analysis for the raw data.")
        st.subheader(" Missing Data Summary (Raw Data)")
        missing_pct = df.isnull().mean() * 100
        missing_summary = missing_pct[missing_pct > 0].sort_values(ascending=False)
        if not missing_summary.empty:
            st.write("Percentage of missing values per column:")
            st.write(missing_summary)
        else:
            st.success("No missing data found in the raw dataset.")
    else:
        st.warning("Please upload data first in the 'Upload Data' section.")


# Imputation process
if menu == "Data Imputation":
    # Use df from session state, prioritize processed if available
    if "df_processed" in st.session_state and st.session_state["df_processed"] is not None:
        df = st.session_state["df_processed"].copy() # Work on a copy for imputation
        st.subheader(" Data Imputation")

        st.write("Starting imputation process...")

        for col in df.columns:
            missing_pct = df[col].isnull().mean() * 100
            if missing_pct == 0:
                continue

            st.write(f" Imputing column: `{col}` ({missing_pct:.1f}% missing)")

            # Strategy 1: Drop column/rows if too much missing data (configurable threshold)
            if missing_pct > 50: # Example threshold: drop column if > 50% missing
                 st.warning(f"Column `{col}` has {missing_pct:.1f}% missing values. Dropping the column.")
                 df = df.drop(col, axis=1)
                 st.success(f"Dropped column: `{col}`")
                 continue # Move to the next column

            # Strategy 2: Drop rows if a small percentage of rows have missing data in a column
            if missing_pct < 5: # Example threshold: drop rows if < 5% missing in this column
                 missing_rows_count = df[col].isnull().sum()
                 # Check if dropping these rows is a significant loss of data (e.g., less than 1% of total rows)
                 if missing_rows_count / len(df) < 0.01: # Example threshold: drop rows if < 1% of total dataset size
                      df = df.dropna(subset=[col])
                      st.warning(f"Dropped {missing_rows_count} rows with missing values in `{col}` (less than 1% of data).")
                 else:
                      st.info(f"Column `{col}` has {missing_pct:.1f}% missing. Using imputation method for small missing percentage.")
                      # Apply imputation for small missing percentage (mean/mode)
                      if pd.api.types.is_numeric_dtype(df[col]):
                          df[col].fillna(df[col].mean(), inplace=True)
                          st.success(f"Filled missing values in `{col}` with mean.")
                      elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                          # Ensure mode exists before filling
                          mode_val = df[col].mode()
                          if not mode_val.empty:
                              df[col].fillna(mode_val[0], inplace=True)
                              st.success(f"Filled missing values in `{col}` with mode.")
                          else:
                              st.warning(f"Could not find mode for column `{col}`. Skipping imputation.")
                      else:
                           st.warning(f"Imputation strategy not defined for data type of column `{col}` (missing < 5%).")

            # Strategy 3: KNN Imputation for moderate missing percentage
            else: # Missing percentage is >= 5% and <= 50%
                 st.info(f"Column `{col}` has {missing_pct:.1f}% missing. Using KNN Imputation.")
                 if pd.api.types.is_numeric_dtype(df[col]):
                     try:
                         imputer = KNNImputer(n_neighbors=5)
                         # KNNImputer expects a 2D array
                         df[[col]] = imputer.fit_transform(df[[col]])
                         st.success(f"Imputed missing values in `{col}` using KNN.")
                     except Exception as e:
                         st.error(f"Error during KNN imputation for column `{col}`: {e}")
                         st.warning(f"Skipping KNN imputation for `{col}`.")

                 elif pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
                     try:
                         # Convert to string to handle potential NaN and other types during Label Encoding
                         cat_data = df[col].astype(str)

                         # Handle unseen categories by adding a placeholder before fitting LabelEncoder
                         # This prevents errors if NaN was the only "unseen" category and it gets imputed to something else
                         unique_values = cat_data.unique().tolist()
                         if 'nan' in unique_values: unique_values.remove('nan')
                         unique_values_for_encoding = unique_values + ['_placeholder_for_knn'] # Add a temporary placeholder

                         le = LabelEncoder()
                         le.fit(unique_values_for_encoding) # Fit on unique values plus placeholder

                         # Transform the column (NaN becomes an integer category)
                         encoded_data = le.transform(cat_data)

                         imputer = KNNImputer(n_neighbors=5)
                         # Reshape for imputer and perform imputation
                         imputed_encoded_data = imputer.fit_transform(encoded_data.reshape(-1, 1))

                         # Round imputed values to the nearest integer to match encoded categories
                         imputed_encoded_data_int = np.round(imputed_encoded_data).flatten().astype(int)

                         # Ensure imputed values are within the range of fitted categories
                         max_encoded_val = len(le.classes_) - 1
                         imputed_encoded_data_int = np.clip(imputed_encoded_data_int, 0, max_encoded_val)

                         # Inverse transform to get back the original categories
                         df[col] = le.inverse_transform(imputed_encoded_data_int)

                         # Replace the placeholder back to NaN if it was imputed to that value (unlikely but safe)
                         df[col] = df[col].replace('_placeholder_for_knn', np.nan)

                         st.success(f"Imputed missing values in categorical column `{col}` using Label Encoding and KNN.")

                     except Exception as e:
                         st.error(f"Error during categorical imputation for column `{col}`: {e}")
                         st.warning(f"Skipping categorical imputation for `{col}`.")
                 else:
                     st.warning(f"Imputation strategy not defined for data type of column `{col}` (missing >= 5% and <= 50%).")


        st.write("Imputation process complete. Checking for remaining missing values:")
        st.write(df.isnull().sum())

        # Store the imputed DataFrame in session state
        st.session_state["df_imputed"] = df

        # Optional: Provide a download link for the imputed data
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
         st.warning("Please preprocess the data first in the 'Data Preprocessing' section before imputation.")
    else:
        st.warning("Please upload data first in the 'Upload Data' section.")

# empty block at the end for potential future additions or just as a separator
