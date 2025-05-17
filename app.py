import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Set Streamlit page config
st.set_page_config(page_title="Health Data Analyzer", layout="wide", page_icon="🔎")

# Custom style
st.markdown("""
    <style>
        .main {background-color: #f4f6f9;}
        h1, h2, h3 {color: #1f4e79;}
        .stButton button {
            background-color: #1f77b4;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🧠 Health Data Analyzer - Preprocessing Module")

# Sidebar menu
menu = st.sidebar.selectbox(
    "Choose a Task",
    ["Upload Data", "Data Overview", "Missing Data Analysis", "Data Imputation"]
)

# File uploader
if menu == "Upload Data":
    uploaded_file = st.file_uploader("📁 Upload your dataset (CSV, XLSX)", type=["csv", "xlsx"])
    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
        st.session_state["df"] = df
        st.success("✅ Data uploaded successfully!")

# Data Overview
if menu == "Data Overview" and "df" in st.session_state:
    df = st.session_state["df"]
    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    st.subheader("🧾 Data Types")
    st.write(df.dtypes)

    st.subheader("📈 Distribution Plot")
    column = st.selectbox("Select a column", df.columns)
    if df[column].dtype in [np.float64, np.int64]:
        fig, ax = plt.subplots()
        sns.histplot(df[column].dropna(), kde=True, ax=ax)
        st.pyplot(fig)
    else:
        st.bar_chart(df[column].value_counts())

# Missing data analysis
if menu == "Missing Data Analysis" and "df" in st.session_state:
    df = st.session_state["df"]
    st.subheader("🩻 Missing Data Summary")
    missing_pct = df.isnull().mean() * 100
    st.write(missing_pct[missing_pct > 0].sort_values(ascending=False))

    st.subheader("🔍 Missingness Mechanism (Heuristic)")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            st.write(f"Column: **{col}**")
            if df[col].dtype in [np.float64, np.int64]:
                corr = df.corr()[col].drop(col).abs().max()
                if corr > 0.3:
                    st.info("Likely MAR (Missing At Random)")
                else:
                    st.info("Possibly MCAR")
            else:
                st.warning("MNAR suspected (Categorical column, hard to infer without domain knowledge)")

# Imputation process
if menu == "Data Imputation" and "df" in st.session_state:
    df = st.session_state["df"]
    df_copy = df.copy()

    st.subheader("🧪 Imputing Missing Values...")

    for col in df.columns:
        missing_pct = df[col].isnull().mean() * 100
        if missing_pct == 0:
            continue

        st.write(f"🔧 Imputing column: `{col}` ({missing_pct:.1f}% missing)")
        
        # Drop rows if >50% and total missing rows < 30% of dataset
        if missing_pct > 50:
            missing_rows = df[col].isnull().sum()
            total_rows = len(df)
            if missing_rows / total_rows < 0.3:
                df_copy = df_copy[df_copy[col].notnull()]
                st.warning(f"Dropped {missing_rows} rows due to excessive missingness in `{col}`")
            continue

        # Simple imputation
        if missing_pct < 5:
            if df[col].dtype in [np.float64, np.int64]:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                st.success("Mean imputation applied")
            else:
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
                st.success("Mode imputation applied")
        
        # KNN Imputation
        else:
            st.info("Applying KNN Imputation (5-nearest neighbors)...")
            if df[col].dtype in [np.float64, np.int64]:
                imputer = KNNImputer(n_neighbors=5)
                df_copy[col] = imputer.fit_transform(df_copy[[col]])
            else:
                le = LabelEncoder()
                cat_data = df_copy[col].astype(str)
                df_copy[col] = le.fit_transform(cat_data)
                imputer = KNNImputer(n_neighbors=5)
                df_copy[[col]] = imputer.fit_transform(df_copy[[col]])
                df_copy[col] = le.inverse_transform(df_copy[col].astype(int))

    st.session_state["df_imputed"] = df_copy
    st.success("✅ Imputation complete! Proceed to the next module.")

