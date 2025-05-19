import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Utility Functions
# -----------------------------
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
st.title(" Data Imputation App")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "upload"

# Sidebar Navigation
tabs = ["Upload Data", "Data Preprocessing", "Imputation", "Modeling"]
tab_selection = st.sidebar.radio("Navigation", tabs)
st.session_state.active_tab = tab_selection

# Tab: Upload Data
if st.session_state.active_tab == "Upload Data":
    st.header(" Upload Data")
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("Data uploaded successfully.")
        st.dataframe(df.head())

# Tab: Preprocessing
elif st.session_state.active_tab == "Data Preprocessing":
    st.header(" Data Preprocessing")
    if "df" not in st.session_state:
        st.warning("Please upload data first in 'Upload Data' tab.")
    else:
        df = st.session_state["df"].copy()
        st.session_state["df_processed"] = df  # Placeholder for actual processing
        st.success("Data passed to 'df_processed'.")
        st.dataframe(df.head())

# Tab: Imputation
elif st.session_state.active_tab == "Imputation":
    st.header(" Missing Value Imputation")
    if "df_processed" not in st.session_state:
        st.warning("Please preprocess data first.")
        st.stop()

    df = st.session_state["df_processed"].copy()

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
        df = impute_mean_mode(df)
    elif strategy == "Median/Mode for all":
        df = impute_median_mode(df)
    elif strategy == "KNN for all":
        knn_neighbors = st.sidebar.slider("KNN neighbors", 2, 10, 5)
        df = impute_knn_all(df, knn_neighbors)
    elif strategy == "Drop rows with missing values":
        df = df.dropna()
        st.warning("Dropped all rows with missing values.")

    st.session_state["df_imputed"] = df
    st.write(" Imputation complete. Checking for remaining missing values:")
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
        if st.button(" Next: Modeling"):
            st.session_state.active_tab = "Modeling"

# Tab: Modeling
elif st.session_state.active_tab == "Modeling":
    st.header(" Modeling (Coming Soon)")
    st.info("This section will allow you to build models on imputed data.")
    if "df_imputed" in st.session_state:
        st.dataframe(st.session_state["df_imputed"].head())
    else:
        st.warning("No imputed data available.")
