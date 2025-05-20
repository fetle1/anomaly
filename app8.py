import streamlit as st
import pandas as pd
import numpy as np
import io

# Optional visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# For imputation
from sklearn.impute import KNNImputer

# Translation dictionary
translations = {
    "en": {
        "Upload": "Upload",
        "Preprocessing": "Preprocessing",
        "Missing Data Analysis": "Missing Data Analysis",
        "Anomaly Detection": "Anomaly Detection",
        "Next ➡": "Next ➡",
        "⬅ Back": "⬅ Back",
        "Download": "📥 Download",
        "No anomalies": "No anomalies detected.",
        "Anomalies found": "{} anomalies found"
    },
    "sw": {
        "Upload": "Pakia",
        "Preprocessing": "Usafishaji",
        "Missing Data Analysis": "Uchanganuzi wa Upungufu",
        "Anomaly Detection": "Ugunduzi wa Shida",
        "Next ➡": "Ifuatayo ➡",
        "⬅ Back": "⬅ Nyuma",
        "Download": "📥 Pakua",
        "No anomalies": "Hakuna shida zilizogunduliwa.",
        "Anomalies found": "{} shida zimegunduliwa"
    },
    "fr": {
        "Upload": "Téléverser",
        "Preprocessing": "Prétraitement",
        "Missing Data Analysis": "Analyse des Données Manquantes",
        "Anomaly Detection": "Détection d'Anomalies",
        "Next ➡": "Suivant ➡",
        "⬅ Back": "⬅ Retour",
        "Download": "📥 Télécharger",
        "No anomalies": "Aucune anomalie détectée.",
        "Anomalies found": "{} anomalies détectées"
    }
}

def T(text):
    lang = st.session_state.get("language", "en")
    return translations.get(lang, translations["en"]).get(text, text)

# Set page config
st.set_page_config(page_title="TafitiX", layout="wide")

# Apply custom CSS
st.markdown("""
    <style>
        .breadcrumb { font-weight: bold; font-size: 16px; margin-bottom: 10px; color: #336699; }
        .bottom-button-container { margin-top: 30px; }
        .stButton>button { width: 100%; }
        .stSelectbox>div { width: 100%; }
    </style>
""", unsafe_allow_html=True)

# Language selector
lang = st.sidebar.selectbox("🌐 Language / Lugha / Langue", ["en", "sw", "fr"], format_func=lambda x: translations[x]["Upload"])
st.session_state["language"] = lang

# Utility: Cleaning
def clean_and_preprocess(df):
    changes = []
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        changes.append("Dropped column: Unnamed: 0")
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    changes.append("Normalized column names")
    return df, changes

# Utility: Default strategy
def apply_default_strategy(df, options):
    missing_percent = df.isnull().mean()
    if options["drop_high_missing_cols"]:
        cols_to_drop = missing_percent[missing_percent > 0.5].index.tolist()
        df.drop(columns=cols_to_drop, inplace=True)

    if options["drop_low_missing_rows"]:
        row_missing_percent = df.isnull().mean(axis=1)
        df = df[row_missing_percent < 0.01]

    if options["impute_low_missing"]:
        for col in df.columns[df.isnull().mean() < 0.05]:
            if df[col].dtype in [np.float64, np.int64]:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode().iloc[0], inplace=True)

    if options["impute_moderate_missing"]:
        cols = df.columns[df.isnull().mean().between(0.05, 0.5)]
        if not cols.empty:
            knn = KNNImputer(n_neighbors=options["knn_neighbors"])
            df[cols] = knn.fit_transform(df[cols])

    return df

# Utility: Mean/Mode
def impute_mean_mode(df):
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
    return df

# Utility: Median/Mode
def impute_median_mode(df):
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue
        if df[col].dtype in [np.float64, np.int64]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode().iloc[0], inplace=True)
    return df

# Utility: KNN
def impute_knn_all(df, n_neighbors=5):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    knn = KNNImputer(n_neighbors=n_neighbors)
    df[numeric_cols] = knn.fit_transform(df[numeric_cols])
    return df
# Rule-based anomaly detection function
def detect_rule_based_anomalies(df):
    anomalies = pd.Series([False] * len(df), index=df.index)

    def col_exists(*cols):
        return all(col in df.columns for col in cols)

    if col_exists('hemoglobin'):
        anomalies |= df['hemoglobin'] <= 0

    if col_exists('glucose'):
        anomalies |= df['glucose'] <= 0

    if col_exists('spo2'):
        anomalies |= df['spo2'] <= 0

    if col_exists('systolic', 'dystolic'):
        anomalies |= df['dystolic'] > df['systolic']

    if col_exists('sex', 'pregnant'):
        anomalies |= (df['sex'].astype(str).str.lower() == 'male') & (df['pregnant'] == True)

    if col_exists('sex', 'bph'):
        anomalies |= (df['sex'].astype(str).str.lower() == 'female') & (df['bph'] == True)

    if col_exists('age', 'pregnant'):
        anomalies |= (df['age'] < 5) & (df['pregnant'] == True)
        anomalies |= (df['age'] > 70) & (df['pregnant'] == True)

    if col_exists('dob', 'dod'):
        anomalies |= pd.to_datetime(df['dob'], errors='coerce') > pd.to_datetime(df['dod'], errors='coerce')

    if col_exists('admission_date', 'discharge_date'):
        anomalies |= pd.to_datetime(df['discharge_date'], errors='coerce') < pd.to_datetime(df['admission_date'], errors='coerce')

    return anomalies

# -----------------------------
# Streamlit UI Logic - Tabs
# -----------------------------
st.title("TafitiX")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = T("Upload")

breadcrumb = f"You are here: ➤ <span>{st.session_state.active_tab}</span>"
st.markdown(f'<div class="breadcrumb">{breadcrumb}</div>', unsafe_allow_html=True)

tabs = [T("Upload"), T("Preprocessing"), "Data Overview", T("Missing Data Analysis"), T("Anomaly Detection")]
tab_selection = st.sidebar.radio("Navigation", tabs)
st.session_state.active_tab = tab_selection

# Upload Tab
if st.session_state.active_tab == T("Upload"):
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("File uploaded.")
        st.dataframe(df.head())

    st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col2:
        if st.button("Next ➡", key="upload_next"):
            st.session_state.active_tab = T("Preprocessing")
    st.markdown('</div>', unsafe_allow_html=True)

# Preprocessing Tab
elif st.session_state.active_tab == T("Preprocessing"):
    if "df" not in st.session_state:
        st.warning("Please upload your data first.")
    else:
        df = st.session_state["df"].copy()
        if st.button("Run Preprocessing"):
            df, changes = clean_and_preprocess(df)
            st.session_state["df_processed"] = df
            st.success("Preprocessing complete.")
            st.write("Changes made:")
            for change in changes:
                st.markdown(f"- {change}")
            st.dataframe(df.head())

        with st.expander("Advanced: Drop or Convert Columns"):
            identifier_cols = st.text_input("Enter comma-separated identifier columns to exclude temporarily")
            convert_col = st.selectbox("Select a column to convert type", df.columns)
            current_type = str(df[convert_col].dtype)
            st.markdown(f"**Current type:** {current_type}")
            convert_to = st.selectbox("Convert to:", ["Keep as is", "float", "int", "category", "string"])
            if convert_to != "Keep as is":
                if convert_to == "float":
                    df[convert_col] = pd.to_numeric(df[convert_col], errors='coerce').astype(float)
                elif convert_to == "int":
                    df[convert_col] = pd.to_numeric(df[convert_col], errors='coerce').astype("Int64")
                elif convert_to == "category":
                    df[convert_col] = df[convert_col].astype("category")
                elif convert_to == "string":
                    df[convert_col] = df[convert_col].astype(str)
                st.success(f"Converted {convert_col} to {convert_to}")

            drop_cols = st.multiselect("Select columns to drop", df.columns)
            if st.button("Apply Drops"):
                if drop_cols:
                    df.drop(columns=drop_cols, inplace=True)
                    st.success("Dropped selected columns.")
                st.session_state["df_processed"] = df

    st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back", key="pre_back"):
            st.session_state.active_tab = T("Upload")
    with col2:
        if st.button("Next ➡", key="pre_next"):
            st.session_state.active_tab = "Data Overview"
    st.markdown('</div>', unsafe_allow_html=True)

# Data Overview Tab
elif st.session_state.active_tab == "Data Overview":
    if "df_processed" not in st.session_state:
        st.warning("Please preprocess your data first.")
    else:
        df = st.session_state["df_processed"]
        st.subheader("Summary Statistics for Numeric Variables")
        st.dataframe(df.describe())

        st.subheader("Visualize a Variable")
        selected_var = st.selectbox("Select a variable", df.columns)
        if pd.api.types.is_numeric_dtype(df[selected_var]):
            chart_type = st.selectbox("Choose chart", ["Histogram", "Line Chart"])
            if chart_type == "Histogram":
                st.bar_chart(df[selected_var].value_counts().sort_index())
            else:
                st.line_chart(df[selected_var])
        else:
            chart_type = st.selectbox("Choose chart", ["Bar Chart", "Pie Chart"])
            counts = df[selected_var].value_counts()
            if chart_type == "Bar Chart":
                st.bar_chart(counts)
            else:
                st.pyplot(counts.plot.pie(autopct='%1.1f%%', figsize=(5, 5)).figure)

        st.subheader("Outlier Detection (IQR Method)")
        outlier_col = st.selectbox("Select numeric column to check outliers", df.select_dtypes(include=[np.number]).columns)
        Q1 = df[outlier_col].quantile(0.25)
        Q3 = df[outlier_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[outlier_col] < lower_bound) | (df[outlier_col] > upper_bound)]
        st.markdown(f"**Outliers in {outlier_col}:** {len(outliers)}")
        st.dataframe(outliers)

    st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back", key="overview_back"):
            st.session_state.active_tab = T("Preprocessing")
    with col2:
        if st.button("Next ➡", key="overview_next"):
            st.session_state.active_tab = T("Missing Data Analysis")
    st.markdown('</div>', unsafe_allow_html=True)
# Missing Data Analysis Tab
elif st.session_state.active_tab == T("Missing Data Analysis"):
    if "df_processed" not in st.session_state:
        st.warning("Please preprocess your data first.")
    else:
        df = st.session_state["df_processed"].copy()

        st.subheader("Missing Value Summary")
        missing_percent = df.isnull().mean() * 100
        st.dataframe(missing_percent[missing_percent > 0].sort_values(ascending=False).reset_index().rename(columns={"index": "Column", 0: "Missing %"}))

        st.subheader("Missing Pattern Visualization (MICE)")
        try:
            import missingno as msno
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots()
            msno.matrix(df, ax=ax)
            st.pyplot(fig)
        except ImportError:
            st.warning("missingno not installed. Please install with `pip install missingno`.")

        st.subheader("Choose Imputation Strategy")
        strategy = st.selectbox("Imputation Method", [
            "Default Strategy (by missing %)", "Mean/Mode", "Median/Mode", "KNN", "Drop missing rows"
        ])

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
        st.success("✅ Imputation complete")

        if st.checkbox("Visualize data after imputation"):
            st.dataframe(df.head())

        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        st.download_button("📥 Download Imputed Data", data=csv_buffer.getvalue(), file_name="imputed_data.csv", mime="text/csv")

    st.markdown('<div class="bottom-button-container">', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back", key="impute_back"):
            st.session_state.active_tab = "Data Overview"
    with col2:
        if st.button("Next ➡", key="impute_next"):
            st.session_state.active_tab = T("Anomaly Detection")
    st.markdown('</div>', unsafe_allow_html=True)

# Anomaly Detection Tab
elif st.session_state.active_tab == T("Anomaly Detection"):
    st.subheader("Rule-Based Anomaly Detection")

    if "df_imputed" in st.session_state:
        df = st.session_state["df_imputed"].copy()

        # Run rule-based anomaly detection
        anomalies = detect_rule_based_anomalies(df)
        anomaly_df = df[anomalies]
        count = anomalies.sum()

        if count > 0:
            st.success(f"✅ {count} anomaly{'ies' if count > 1 else ''} found")
            st.dataframe(anomaly_df)

            csv = anomaly_df.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Anomalies", data=csv, file_name='anomalies.csv', mime='text/csv')
        else:
            st.info("No anomalies detected using rule-based logic.")
    else:
        st.warning("⚠ Please preprocess and impute data first.")
