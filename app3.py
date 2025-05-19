import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Utility Functions
# -----------------------------

def map_sex_values(series):
    mapping = {
        'male': 'male', 'm': 'male', 'man': 'male', 'boy': 'male',
        'female': 'female', 'f': 'female', 'woman': 'female', 'women': 'female', 'girl': 'female'
    }
    return series.str.lower().str.strip().map(mapping).fillna(series)

def default_preprocessing(df, id_cols):
    summary = []
    # Lowercase column names
    old_cols = df.columns.tolist()
    df.columns = [col.lower().strip() for col in df.columns]
    if old_cols != df.columns.tolist():
        summary.append(f"Lowercased and stripped column names.")

    # Strip spaces and lowercase string data, map sex values
    for col in df.columns:
        if col in id_cols:
            continue
        if df[col].dtype == object:
            old_vals = df[col].unique()
            df[col] = df[col].astype(str).str.strip().str.lower()
            # Map sex/gender values
            if 'sex' in col or 'gender' in col:
                df[col] = map_sex_values(df[col])
            new_vals = df[col].unique()
            if not np.array_equal(old_vals, new_vals):
                summary.append(f"Processed strings and mapped values in column `{col}`.")
    return df, summary

def convert_column_types(df, convert_dict):
    summary = []
    for col, new_type in convert_dict.items():
        if col in df.columns:
            try:
                if new_type == 'category':
                    df[col] = df[col].astype('category')
                elif new_type == 'int':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif new_type == 'float':
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype(float)
                elif new_type == 'string':
                    df[col] = df[col].astype(str)
                else:
                    summary.append(f"Unknown type {new_type} for column {col}. Skipped.")
                    continue
                summary.append(f"Converted column `{col}` to {new_type}.")
            except Exception as e:
                summary.append(f"Failed to convert column `{col}` to {new_type}: {e}")
    return df, summary

def drop_columns(df, drop_list):
    summary = []
    for col in drop_list:
        if col in df.columns:
            df = df.drop(columns=[col])
            summary.append(f"Dropped column `{col}`.")
    return df, summary

def impute_mean_mode(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
    return df

def impute_median_mode(df):
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
    return df

def impute_knn_all(df, k):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = df.drop(columns=numeric_cols)
    imputer = KNNImputer(n_neighbors=k)
    df_numeric = pd.DataFrame(imputer.fit_transform(df[numeric_cols]), columns=numeric_cols)
    return pd.concat([df_numeric, non_numeric_cols], axis=1)

def apply_default_strategy(df, options):
    knn_neighbors = options.get("knn_neighbors", 5)
    for col in df.columns:
        if col not in df.columns:
            continue

        missing_pct = df[col].isnull().mean() * 100
        if missing_pct == 0:
            continue

        st.write(f"Processing `{col}` ({missing_pct:.1f}% missing)")

        if options["drop_high_missing_cols"] and missing_pct > 50:
            st.warning(f"Dropping column `{col}` (>50% missing)")
            df = df.drop(columns=[col])
            continue

        if options["drop_low_missing_rows"] and missing_pct < 5:
            missing_rows = df[col].isnull().sum()
            if missing_rows / len(df) < 0.01:
                df = df.dropna(subset=[col])
                st.warning(f"Dropped rows for `{col}` (<1% of data)")
                continue

        if options["impute_low_missing"] and missing_pct < 5 and not options["drop_low_missing_rows"]:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
                st.success(f"Imputed `{col}` with mean.")
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
                    st.success(f"Imputed `{col}` with mode.")

        elif options["impute_moderate_missing"] and 5 <= missing_pct <= 50:
            if pd.api.types.is_numeric_dtype(df[col]):
                try:
                    imputer = KNNImputer(n_neighbors=knn_neighbors)
                    df[[col]] = imputer.fit_transform(df[[col]])
                    st.success(f"KNN-imputed `{col}`.")
                except Exception as e:
                    st.error(f"KNN error on `{col}`: {e}")
            else:
                try:
                    cat_data = df[col].astype(str)
                    le = LabelEncoder()
                    encoded = le.fit_transform(cat_data.fillna("_missing"))
                    imputer = KNNImputer(n_neighbors=knn_neighbors)
                    imputed = imputer.fit_transform(encoded.reshape(-1, 1))
                    imputed_int = np.clip(np.round(imputed).astype(int), 0, len(le.classes_)-1)
                    df[col] = le.inverse_transform(imputed_int.flatten())
                    df[col] = df[col].replace("_missing", np.nan)
                    st.success(f"KNN-imputed categorical `{col}`.")
                except Exception as e:
                    st.error(f"KNN categorical error on `{col}`: {e}")
        else:
            st.info(f"No action taken for `{col}` ({missing_pct:.1f}% missing).")
    return df

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Data Imputation App", layout="wide")
st.title("Data Imputation App")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "upload"

if "id_columns" not in st.session_state:
    st.session_state.id_columns = []

if "preprocess_summary" not in st.session_state:
    st.session_state.preprocess_summary = []

# Sidebar Navigation
tabs = ["Upload Data", "Data Preprocessing", "Imputation", "Modeling"]
tab_selection = st.sidebar.radio("Navigation", tabs)
st.session_state.active_tab = tab_selection

# Tab: Upload Data
if st.session_state.active_tab == "Upload Data":
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("Data uploaded successfully.")
        st.dataframe(df.head())

# Tab: Data Preprocessing
elif st.session_state.active_tab == "Data Preprocessing":
    st.header("Data Preprocessing")

    if "df" not in st.session_state:
        st.warning("Please upload data first in 'Upload Data' tab.")
        st.stop()

    df = st.session_state["df"].copy()

    # Input identifier columns (comma separated)
    id_cols_input = st.text_input("Enter identifier columns (comma separated) to keep aside (will NOT be processed):",
                                  value=",".join(st.session_state.id_columns))
    id_columns = [c.strip().lower() for c in id_cols_input.split(",") if c.strip()]
    st.session_state.id_columns = id_columns

    # Show columns and datatypes for reference
    st.write("### Columns and Data Types")
    col_info = pd.DataFrame({"Column": df.columns, "Data Type": df.dtypes})
    st.dataframe(col_info)

    # Section: Default Preprocessing (apply to all non-id cols)
    if st.button("Run Default Preprocessing"):
        # Apply default preprocessing to all columns except id_columns
        df_non_id = df.drop(columns=[col for col in id_columns if col in df.columns], errors='ignore')
        df_processed, summary = default_preprocessing(df_non_id, id_columns)
        st.session_state["df_processed"] = pd.concat([df[id_columns], df_processed], axis=1)
        st.session_state.preprocess_summary = summary
        st.success("Default preprocessing completed.")
        for line in summary:
            st.write(f"- {line}")

    # Section: Interactive Preprocessing
    st.write("---")
    st.subheader("Interactive Preprocessing")

    # Columns to drop
    drop_cols = st.multiselect("Select columns to drop", options=df.columns.tolist())
    # Variable type conversion
    st.write("Convert variable types:")
    type_options = ["int", "float", "category", "string"]
    convert_cols = st.multiselect("Select columns to convert", options=df.columns.tolist())
    convert_dict = {}
    for col in convert_cols:
        new_type = st.selectbox(f"Convert `{col}` to:", options=type_options, key=f"type_{col}")
        convert_dict[col] = new_type

    if st.button("Apply Interactive Preprocessing"):
        df_processed = st.session_state.get("df_processed", df.copy())
        summary = []

        # Drop columns
        df_processed, drop_summary = drop_columns(df_processed, drop_cols)
        summary.extend(drop_summary)

        # Convert types
        df_processed, convert_summary = convert_column_types(df_processed, convert_dict)
        summary.extend(convert_summary)

        st.session_state["df_processed"] = df_processed
        st.session_state.preprocess_summary.extend(summary)

        st.success("Interactive preprocessing applied.")
        for line in summary:
            st.write(f"- {line}")

    # Show processed data if available
    if "df_processed" in st.session_state:
        st.write("### Processed Data Preview")
        st.dataframe(st.session_state["df_processed"].head())

    # Show summary
    if st.session_state.preprocess_summary:
        st.write("### Preprocessing Summary")
        for line in st.session_state.preprocess_summary:
            st.write(f"- {line}")

# Tab: Imputation
elif st.session_state.active_tab == "Imputation":
    st.header("Missing Value Imputation")

    if "df_processed" not in st.session_state:
        st.warning("Please preprocess data first.")
        st.stop()

    # Exclude identifier columns before imputation
    df = st.session_state["df_processed"].copy()
    id_cols = st.session_state.id_columns
    id_df = pd.DataFrame()
    if id_cols:
        id_df = df[id_cols].copy()
        df = df.drop(columns=id_cols, errors='ignore')

    strategy = st.sidebar.radio(
        "Choose Imputation Strategy",
        (
            "Default Strategy (rules by missing %)",
            "Mean/Mode for all",
            "Median/Mode for all",
            "KNN for all",
            "Drop rows with missing values"
        )
    )

    if strategy == "Default Strategy (rules by missing %)":
        drop_high_missing_cols = st.sidebar.checkbox("Drop columns with >50% missing", True)
        drop_low_missing_rows = st.sidebar.checkbox("Drop rows with <1% missing per column", True)
        impute_low_missing = st.sidebar.checkbox("Impute missing <5% (Mean/Mode)", True)
        impute_moderate_missing = st.sidebar.checkbox("Impute 5%-50% missing (KNN)", True)
        knn_neighbors = st.sidebar.slider("KNN neighbors", 2, 10, 5)

        if st.button("Apply Imputation"):
            options = {
                "drop_high_missing_cols": drop_high_missing_cols,
                "drop_low_missing_rows": drop_low_missing_rows,
                "impute_low_missing": impute_low_missing,
                "impute_moderate_missing": impute_moderate_missing,
                "knn_neighbors": knn_neighbors
            }
            df = apply_default_strategy(df, options)

    elif strategy == "Mean/Mode for all":
        if st.button("Apply Imputation"):
            df = impute_mean_mode(df)
            st.success("Imputed missing values with mean/mode.")

    elif strategy == "Median/Mode for all":
        if st.button("Apply Imputation"):
            df = impute_median_mode(df)
            st.success("Imputed missing values with median/mode.")

    elif strategy == "KNN for all":
        knn_neighbors = st.sidebar.slider("KNN neighbors", 2, 10, 5)
        if st.button("Apply Imputation"):
            df = impute_knn_all(df, knn_neighbors)
            st.success("Imputed missing values using KNN.")

    elif strategy == "Drop rows with missing values":
        if st.button("Apply Imputation"):
            df = df.dropna()
            st.warning("Dropped all rows with missing values.")

    # Merge id columns back
    if not id_df.empty:
        df = pd.concat([id_df.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    st.session_state["df_imputed"] = df

    st.write("Imputation complete. Checking for remaining missing values:")
    remaining = df.isnull().sum()
    if remaining.sum() == 0:
        st.success("No missing values remain.")
    else:
        st.warning("Some missing values remain:")
        st.dataframe(remaining[remaining > 0])

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Imputed Data (CSV)",
        data=csv_buffer.getvalue(),
        file_name="imputed_data.csv",
        mime="text/csv"
    )

    col1, col2 = st.columns([8, 2])
    with col2:
        if st.button("Next: Modeling"):
            st.session_state.active_tab = "Modeling"

# Tab: Modeling
elif st.session_state.active_tab == "Modeling":
    st.header("Modeling (Coming Soon)")
    st.info("This section will allow you to build models on imputed data.")
    if "df_imputed" in st.session_state:
        st.dataframe(st.session_state["df_imputed"].head())
    else:
        st.warning("No imputed data available.")
