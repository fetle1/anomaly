import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import io
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
import random

# Set Streamlit page config
st.set_page_config(page_title="Health Data Analyzer", layout="wide", page_icon="🧠")

# Custom UI Styling
st.markdown("""
    <style>
        .main {background-color: #e6f2ff;}
        h1, h2, h3 {color: #003366; font-family: 'Segoe UI', sans-serif;}
        .stButton button {
            background-color: #0066cc;
            color: blue;
            font-weight: bold;
            border-radius: 10px;
            padding: 10px 20px;
            transition: 0.3s;
        }
        .stButton button:hover {
            background-color: #004d99;
            transform: scale(1.05);
        }
    </style>
""", unsafe_allow_html=True)

# Title with animation
st.markdown("""
    <h1 style='animation: fadeInDown 2s;'>🧠 Health Data Analyzer - Preprocessing Module</h1>
    <style>
    @keyframes fadeInDown {
        from {opacity: 0; transform: translateY(-20px);}
        to {opacity: 1; transform: translateY(0);}
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar menu
menu = st.sidebar.selectbox(
    "📌 Choose a Task",
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

    st.subheader("📈 Column Distribution Visualization")
    column = st.selectbox("Select a column", df.columns)

    color_palette = random.choice(sns.color_palette("Set2", 10))
    fig, ax = plt.subplots()

    if df[column].dtype in [np.float64, np.int64]:
        sns.histplot(df[column].dropna(), kde=True, ax=ax, color=color_palette)
    else:
        if df[column].nunique() < 10:
            df[column].value_counts().plot.pie(autopct='%1.1f%%', ax=ax, colors=sns.color_palette("Pastel1"))
            ax.set_ylabel('')
        else:
            sns.countplot(y=df[column], ax=ax, palette="Set2")

    st.pyplot(fig)

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

        if missing_pct > 50:
            missing_rows = df[col].isnull().sum()
            total_rows = len(df)
            if missing_rows / total_rows < 0.3:
                df_copy = df_copy[df_copy[col].notnull()]
                st.warning(f"Dropped {missing_rows} rows due to excessive missingness in `{col}`")
            continue

        if missing_pct < 5:
            if df[col].dtype in [np.float64, np.int64]:
                df_copy[col].fillna(df_copy[col].mean(), inplace=True)
                st.success("Mean imputation applied")
            else:
                df_copy[col].fillna(df_copy[col].mode()[0], inplace=True)
                st.success("Mode imputation applied")
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

    # Save cleaned dataset
    st.session_state["df_imputed"] = df_copy
    csv = df_copy.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Cleaned Data",
        data=csv,
        file_name="cleaned_health_data.csv",
        mime='text/csv',
        key="download_button"
    )

    st.success("✅ Imputation complete and dataset saved! Proceed to anomaly detection next.")
