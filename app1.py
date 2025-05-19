import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Custom CSS Styling
# -----------------------------

def local_css():
    st.markdown(
        """
        <style>
        /* Background gradient */
        .stApp {
            background: linear-gradient(135deg, #f0f4f8, #d9e2ec);
            color: #0f172a;
        }
        /* Sidebar background */
        [data-testid="stSidebar"] {
            background-color: #334e68;
            color: white;
        }
        /* Sidebar radio button text */
        [data-testid="stSidebar"] label, [data-testid="stSidebar"] .stRadio > label {
            color: white;
        }
        /* Buttons */
        div.stButton > button {
            background-color: #1e40af;
            color: white;
            border-radius: 8px;
            padding: 0.5em 1.2em;
            font-weight: 600;
            border: none;
            transition: background-color 0.3s ease;
        }
        div.stButton > button:hover {
            background-color: #3b82f6;
            color: white;
        }
        /* Container for bottom buttons */
        .bottom-button-container {
            display: flex;
            justify-content: flex-end;
            gap: 0.5em;
            margin-top: 2rem;
            margin-bottom: 2rem;
        }
        /* Breadcrumb style */
        .breadcrumb {
            font-size: 0.9rem;
            margin-bottom: 1rem;
            color: #64748b;
        }
        .breadcrumb span {
            font-weight: 600;
            color: #1e40af;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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

def map_sex_values(df, col):
    if col not in df.columns:
        st.warning(f"Column `{col}` not found for sex mapping.")
        return df
    mapping = {
        "male": "Male", "m": "Male", "man": "Male", "boy": "Male",
        "female": "Female", "f": "Female", "woman": "Female", "women": "Female", "girl": "Female"
    }
    df[col] = df[col].astype(str).str.lower().map(lambda x: mapping.get(x, x)).replace({"male":"Male","female":"Female"})
    return df

def preprocess_default(df):
    df.columns = df.columns.str.lower()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.strip()
    sex_cols = [c for c in df.columns if "sex" in c]
    for c in sex_cols:
        df = map_sex_values(df, c)
    return df

def convert_column_types(df, to_convert):
    for col, new_type in to_convert.items():
        if col in df.columns:
            try:
                if new_type == "category":
                    df[col] = df[col].astype("category")
                elif new_type == "int":
                    df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
                elif new_type == "float":
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                elif new_type == "string":
                    df[col] = df[col].astype(str)
                st.success(f"Converted `{col}` to {new_type}.")
            except Exception as e:
                st.error(f"Failed to convert `{col}` to {new_type}: {e}")
        else:
            st.warning(f"Column `{col}` not found to convert.")
    return df

def drop_columns(df, cols_to_drop):
    missing_cols = [c for c in cols_to_drop if c not in df.columns]
    if missing_cols:
        st.warning(f"Columns not found and can't be dropped: {missing_cols}")
    return df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# -----------------------------
# Navigation and Breadcrumbs
# -----------------------------

tabs = ["Upload Data", "Data Preprocessing", "Imputation", "Modeling"]

def show_breadcrumb(active_tab):
    crumbs = []
    for t in tabs:
        if t == active_tab:
            crumbs.append(f"<span>{t}</span>")
        else:
            crumbs.append(t)
    st.markdown(f'<div class="breadcrumb"> > '.join(crumbs) + "</div>", unsafe_allow_html=True)

def go_next_tab():
    current_idx = tabs.index(st.session_state.active_tab)
    if current_idx < len(tabs) - 1:
        st.session_state.active_tab = tabs[current_idx + 1]

def go_prev_tab():
    current_idx = tabs.index(st.session_state.active_tab)
    if current_idx > 0:
        st.session_state.active_tab = tabs[current_idx - 1]

# -----------------------------
# Streamlit App Setup
# -----------------------------

st.set_page_config(page_title="Data Imputation App", layout="wide")
local_css()
st.title("Data Imputation App")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload Data"

if "df" not in st.session_state:
    st.session_state.df = None

if "df_processed" not in st.session_state:
    st.session_state.df_processed = None

if "df_imputed" not in st.session_state:
    st.session_state.df_imputed = None

if "identifiers" not in st.session_state:
    st.session_state.identifiers = []

# Sidebar navigation
tab_selection = st.sidebar.radio("Navigation", tabs, index=tabs.index(st.session_state.active_tab))
st.session_state.active_tab = tab_selection

# Show breadcrumb
show_breadcrumb(st.session_state.active_tab)

# -----------------------------
# Tab: Upload Data
# -----------------------------
if st.session_state.active_tab == "Upload Data":
    st.header("Upload Data")

    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df
        st.success("Data uploaded successfully.")
        st.dataframe(df.head())

    # Bottom buttons container
    with st.container():
        st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
        # Previous button disabled here (first tab)
        st.markdown('<div style="flex-grow: 1"></div>', unsafe_allow_html=True)
        if st.button("Next: Data Preprocessing"):
            if st.session_state.df is not None:
                st.session_state.active_tab = "Data Preprocessing"
            else:
                st.warning("Please upload a CSV file first.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Tab: Data Preprocessing
# -----------------------------
elif st.session_state.active_tab == "Data Preprocessing":
    st.header("Data Preprocessing")

    if st.session_state.df is None:
        st.warning("Please upload data first in 'Upload Data' tab.")
        st.stop()

    df = st.session_state.df.copy()

    id_cols_input = st.text_input(
        "Enter identifier columns (comma separated) to keep aside (e.g. ID, patient_id):",
        value=",".join(st.session_state.identifiers) if st.session_state.identifiers else ""
    )
    identifiers = [x.strip() for x in id_cols_input.split(",") if x.strip()]
    st.session_state.identifiers = identifiers

    df_proc = df.drop(columns=identifiers, errors='ignore')

    if st.button("Apply Default Preprocessing"):
        df_proc = preprocess_default(df_proc)
        st.session_state.df_processed = df_proc
        st.success("Default preprocessing applied.")
        st.write("Summary of preprocessing:")
        st.write(f"- Lowercased all column names")
        st.write(f"- Stripped leading/trailing spaces from string columns")
        if identifiers:
            st.write(f"- Removed identifier columns from data during processing: {identifiers}")
        if any(["sex" in c for c in df_proc.columns]):
            st.write("- Mapped sex-related values to standard Male/Female")

    st.subheader("Advanced Preprocessing")
    st.markdown("Convert column types and drop unwanted columns:")

    cols_for_conversion = st.multiselect("Select columns to convert type:", options=df_proc.columns)
    new_types = {}
    if cols_for_conversion:
        for col in cols_for_conversion:
            new_type = st.selectbox(f"Select new type for '{col}':", options=["int", "float", "category", "string"], key=f"type_{col}")
            new_types[col] = new_type
    if st.button("Apply Type Conversion"):
        if st.session_state.df_processed is None:
            st.warning("Please apply default preprocessing first.")
        else:
            df_proc = st.session_state.df_processed.copy()
            df_proc = convert_column_types(df_proc, new_types)
            st.session_state.df_processed = df_proc

    cols_to_drop = st.multiselect("Select columns to drop:", options=df_proc.columns)
    if st.button("Drop Selected Columns"):
        if st.session_state.df_processed is None:
            st.warning("Please apply default preprocessing first.")
        else:
            df_proc = st.session_state.df_processed.copy()
            df_proc = drop_columns(df_proc, cols_to_drop)
            st.session_state.df_processed = df_proc
            st.success(f"Dropped columns: {cols_to_drop}")

    if st.session_state.df_processed is not None:
        st.write("Preprocessed Data Preview:")
        st.dataframe(st.session_state.df_processed.head())

    # Bottom buttons container
    with st.container():
        st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
        if st.button("Previous: Upload Data"):
            st.session_state.active_tab = "Upload Data"
        if st.button("Next: Imputation"):
            if st.session_state.df_processed is not None:
                st.session_state.active_tab = "Imputation"
            else:
                st.warning("Please preprocess data first.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Tab: Imputation
# -----------------------------
elif st.session_state.active_tab == "Imputation":
    st.header("Imputation Strategies")

    if st.session_state.df_processed is None:
        st.warning("Please preprocess data first.")
        st.stop()

    df = st.session_state.df_processed.copy()

    st.write("Choose an imputation strategy:")

    impute_strategy = st.radio(
        "Select imputation strategy:",
        ["Default (Rule-Based)", "Mean/Mode", "Median/Mode", "KNN Imputation", "Drop Rows with Missing Values"]
    )

    knn_neighbors = st.number_input("KNN neighbors (if applicable)", min_value=1, max_value=15, value=5, step=1)

    # Default strategy options
    if impute_strategy == "Default (Rule-Based)":
        options = {
            "drop_high_missing_cols": st.checkbox("Drop columns with >50% missing", value=True),
            "drop_low_missing_rows": st.checkbox("Drop rows with <1% missing", value=False),
            "impute_low_missing": st.checkbox("Impute columns with <5% missing (mean/mode)", value=True),
            "impute_moderate_missing": st.checkbox("Impute columns with 5-50% missing (KNN)", value=True),
            "knn_neighbors": knn_neighbors,
        }

    imputed_df = None
    if st.button("Run Imputation"):
        if impute_strategy == "Default (Rule-Based)":
            imputed_df = apply_default_strategy(df.copy(), options)
        elif impute_strategy == "Mean/Mode":
            imputed_df = impute_mean_mode(df.copy())
        elif impute_strategy == "Median/Mode":
            imputed_df = impute_median_mode(df.copy())
        elif impute_strategy == "KNN Imputation":
            imputed_df = impute_knn_all(df.copy(), knn_neighbors)
        elif impute_strategy == "Drop Rows with Missing Values":
            imputed_df = df.dropna()
        else:
            st.warning("Please select a valid strategy.")

        if imputed_df is not None:
            st.session_state.df_imputed = imputed_df
            st.success("Imputation done.")
            st.dataframe(imputed_df.head())

    # Bottom buttons container
    with st.container():
        st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
        if st.button("Previous: Data Preprocessing"):
            st.session_state.active_tab = "Data Preprocessing"
        if st.button("Next: Modeling"):
            if st.session_state.df_imputed is not None:
                st.session_state.active_tab = "Modeling"
            else:
                st.warning("Please run imputation first.")
        st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# Tab: Modeling (Placeholder)
# -----------------------------
elif st.session_state.active_tab == "Modeling":
    st.header("Modeling / Analysis")

    if st.session_state.df_imputed is None:
        st.warning("Please run imputation first.")
        st.stop()

    df = st.session_state.df_imputed.copy()

    st.write("You can now perform modeling or further analysis on the imputed data.")

    st.dataframe(df.head())

    # Bottom buttons container
    with st.container():
        st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
        if st.button("Previous: Imputation"):
            st.session_state.active_tab = "Imputation"
        # No next button here since last tab
        st.markdown('</div>', unsafe_allow_html=True)
