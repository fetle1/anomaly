import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# Set Page Configuration First
# -----------------------------
st.set_page_config(page_title="TafitiX", layout="wide")

# -----------------------------
# Inject CSS for Custom Styling
# -----------------------------
def local_css():
    st.markdown(
        """
        <style>
        .stApp {
            background-image: url('https://images.unsplash.com/photo-1581093588401-245d91841d61?ixlib=rb-4.0.3&auto=format&fit=crop&w=1050&q=80');
            background-size: cover;
            background-position: center;
            color: #0f172a;
        }
        .block-container {
            background-color: rgba(255, 255, 255, 0.85);
            padding: 2rem;
            border-radius: 1rem;
        }
        [data-testid="stSidebar"] {
            background-color: #cfe8fc;
        }
        [data-testid="stSidebar"] .stRadio > label, [data-testid="stSidebar"] label, [data-testid="stSidebar"] div {
            color: white !important;
            font-weight: bold;
        }
        div.stButton > button {
            background-color: #003366;
            color: white;
            font-size: 1rem;
            padding: 0.6em 1.5em;
            border-radius: 10px;
            font-weight: bold;
        }
        div.stButton > button:hover {
            background-color: #0055aa;
            color: #ffffff;
        }
        .bottom-button-container {
            display: flex;
            justify-content: space-between;
            margin-top: 2rem;
        }
        .breadcrumb {
            font-size: 0.9rem;
            color: #003366;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

local_css()

# -----------------------------
# Utility Functions
# -----------------------------
def clean_and_preprocess(df):
    actions = []
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]
    actions.append("Standardized column names (lowercase, underscores, stripped spaces)")

    for col in df.columns:
        if any(keyword in col for keyword in ["sex", "gender"]):
            df[col] = df[col].astype(str).str.lower().str.strip()
            df[col] = df[col].replace({
                "m": "male", "man": "male", "boy": "male",
                "f": "female", "woman": "female", "girl": "female"
            })
            actions.append(f"Standardized gender labels in column '{col}'")

    return df, actions

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

        if options["drop_high_missing_cols"] and missing_pct > 50:
            df.drop(columns=[col], inplace=True)
            continue

        if options["drop_low_missing_rows"] and missing_pct < 5:
            df.dropna(subset=[col], inplace=True)
            continue

        if options["impute_low_missing"] and missing_pct < 5:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                mode_val = df[col].mode()
                if not mode_val.empty:
                    df[col].fillna(mode_val[0], inplace=True)
        elif options["impute_moderate_missing"] and 5 <= missing_pct <= 50:
            if pd.api.types.is_numeric_dtype(df[col]):
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                df[[col]] = imputer.fit_transform(df[[col]])
            else:
                cat_data = df[col].astype(str)
                le = LabelEncoder()
                encoded = le.fit_transform(cat_data.fillna("_missing"))
                imputer = KNNImputer(n_neighbors=knn_neighbors)
                imputed = imputer.fit_transform(encoded.reshape(-1, 1))
                imputed_int = np.clip(np.round(imputed).astype(int), 0, len(le.classes_)-1)
                df[col] = le.inverse_transform(imputed_int.flatten())
                df[col] = df[col].replace("_missing", np.nan)
    return df

def detect_rule_based_anomalies(df):
    anomalies = pd.Series([False] * len(df), index=df.index)
    reasons = [[] for _ in range(len(df))]

    def add_reason(idx, reason):
        reasons[idx].append(reason)

    def col_exists(*cols):
        return all(col in df.columns for col in cols)

    for idx, row in df.iterrows():
        if col_exists('hgb') and pd.notna(row['hgb']) and row['hgb'] < 0:
            anomalies[idx] = True
            add_reason(idx, 'Negative hemoglobin (hgb)')
        if col_exists('glucose') and pd.notna(row['glucose']) and row['glucose'] < 0:
            anomalies[idx] = True
            add_reason(idx, 'Negative glucose')
        if col_exists('spo2') and pd.notna(row['spo2']) and row['spo2'] < 0:
            anomalies[idx] = True
            add_reason(idx, 'Negative oxygen saturation (spo2)')
        if col_exists('systolic', 'dystolic') and pd.notna(row['systolic']) and pd.notna(row['dystolic']) and row['dystolic'] > row['systolic']:
            anomalies[idx] = True
            add_reason(idx, 'Dystolic > Systolic')
        if col_exists('sex', 'pregnant') and pd.notna(row['sex']) and row['sex'].lower() == 'male' and row['pregnant']:
            anomalies[idx] = True
            add_reason(idx, 'Male marked as pregnant')
        if col_exists('sex', 'bph') and pd.notna(row['sex']) and row['sex'].lower() == 'female' and row['bph']:
            anomalies[idx] = True
            add_reason(idx, 'Female marked with BPH')
        if col_exists('age', 'pregnant') and pd.notna(row['age']) and row['pregnant']:
            if row['age'] < 5:
                anomalies[idx] = True
                add_reason(idx, 'Pregnant under age 5')
            if row['age'] > 70:
                anomalies[idx] = True
                add_reason(idx, 'Pregnant over age 70')
        if col_exists('dob', 'dod'):
            dob = pd.to_datetime(row['dob'], errors='coerce')
            dod = pd.to_datetime(row['dod'], errors='coerce')
            if pd.notna(dob) and pd.notna(dod) and dob > dod:
                anomalies[idx] = True
                add_reason(idx, 'DOB after DOD')
        if col_exists('admission_date', 'discharge_date'):
            adm = pd.to_datetime(row['admission_date'], errors='coerce')
            dis = pd.to_datetime(row['discharge_date'], errors='coerce')
            if pd.notna(adm) and pd.notna(dis) and dis < adm:
                anomalies[idx] = True
                add_reason(idx, 'Discharge before admission')

    return anomalies, reasons

# -----------------------------
# Streamlit UI Logic
# -----------------------------

st.title("TafitiX")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Upload"

breadcrumb = f"You are here: ➤ <span>{st.session_state.active_tab}</span>"
st.markdown(f'<div class="breadcrumb">{breadcrumb}</div>', unsafe_allow_html=True)

tabs = ["Upload", "Preprocessing", "Missing Data Analysis", "Anomaly Detection"]
tab_selection = st.sidebar.radio("Navigation", tabs)
st.session_state.active_tab = tab_selection

# Upload Tab
if st.session_state.active_tab == "Upload":
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("File uploaded.")
        st.dataframe(df.head())

    st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col2:
        if st.button("Next ➡"):
            st.session_state.active_tab = "Preprocessing"
    st.markdown('</div>', unsafe_allow_html=True)

# Preprocessing Tab
elif st.session_state.active_tab == "Preprocessing":
    if "df" not in st.session_state:
        st.warning("Please upload your data first.")
    else:
        df = st.session_state["df"].copy()
        df, changes = clean_and_preprocess(df)
        st.session_state["df_processed"] = df
        st.success("Preprocessing complete.")
        st.write("Changes made:")
        for change in changes:
            st.markdown(f"- {change}")
        st.dataframe(df.head())

        with st.expander(" Advanced: Drop or Convert Columns"):
            identifier_cols = st.text_input("Enter comma-separated identifier columns to exclude temporarily")
            drop_cols = st.multiselect("Select columns to drop", df.columns)
            convert_cols = st.multiselect("Select columns to convert", df.columns)
            convert_type = st.selectbox("Convert selected columns to:", ["numeric", "float", "int", "category"])

            for col in convert_cols:
                if convert_type == "numeric":
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                elif convert_type == "float":
                    df[col] = df[col].astype(float)
                elif convert_type == "int":
                    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')
                elif convert_type == "category":
                    df[col] = df[col].astype("category")

            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)

            st.session_state["df_processed"] = df
            st.success("Advanced changes applied.")

    st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            st.session_state.active_tab = "Upload"
    with col2:
        if st.button("Next ➡"):
            st.session_state.active_tab = "Imputation"
    st.markdown('</div>', unsafe_allow_html=True)

# Imputation Tab
elif st.session_state.active_tab == "Missing Data Analysis":
    if "df_processed" not in st.session_state:
        st.warning("Preprocess your data first.")
    else:
        df = st.session_state["df_processed"].copy()
        strategy = st.selectbox("Choose Imputation Strategy", [
            "Default Strategy (by missing %)", "Mean/Mode", "Median/Mode", "KNN", "Drop missing rows"])

        if strategy == "Default Strategy (by missing %)":
            opts = {
                "drop_high_missing_cols": st.checkbox("Drop cols >50% missing", True),
                "drop_low_missing_rows": st.checkbox("Drop rows <1% missing", True),
                "impute_low_missing": st.checkbox("Impute low missing (<5%)", True),
                "impute_moderate_missing": st.checkbox("KNN for 5%-50%", True),
                "knn_neighbors": st.slider("KNN Neighbors", 2, 10, 5)
            }
            df = apply_default_strategy(df, opts)
        elif strategy == "Mean/Mode":
            df = impute_mean_mode(df)
        elif strategy == "Median/Mode":
            df = impute_median_mode(df)
        elif strategy == "KNN":
            df = impute_knn_all(df, 5)
        elif strategy == "Drop missing rows":
            df = df.dropna()

        st.session_state["df_imputed"] = df
        st.success("Imputation complete")
        st.dataframe(df.head())

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button(" Download Imputed Data", data=csv_buffer.getvalue(), file_name="imputed_data.csv", mime="text/csv")

    st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back"):
            st.session_state.active_tab = "Preprocessing"
    with col2:
        if st.button("Next ➡"):
            st.session_state.active_tab = "Anomaly Detection"
    st.markdown('</div>', unsafe_allow_html=True)

# Modeling Tab
elif st.session_state.active_tab == "Anomaly Detection":
    st.info("Modeling features coming soon. You can export and use imputed data.")
    if "df_imputed" in st.session_state:
        df = st.session_state["df_imputed"]
        anomalies, reasons = detect_rule_based_anomalies(df)
        anomaly_df = df[anomalies].copy()
        anomaly_df['Anomaly Reason'] = ["; ".join(r) for r in reasons if r]

        st.markdown("##  Rule-Based Anomaly Detection")
        st.write(f"Total anomalies detected: {anomalies.sum()}")
        st.dataframe(anomaly_df)

        csv_anomalies = io.StringIO()
        anomaly_df.to_csv(csv_anomalies, index=False)
        st.download_button(" Download Anomalies", data=csv_anomalies.getvalue(), file_name="anomalies.csv", mime="text/csv")

    st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
    if st.button("⬅ Back"):
        st.session_state.active_tab = "Imputation"
    st.markdown('</div>', unsafe_allow_html=True)
