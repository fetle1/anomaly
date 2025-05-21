import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import KNNImputer
from scipy import stats
from sklearn.ensemble import IsolationForest
from scipy.stats import shapiro
from sklearn.covariance import MinCovDet
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout

st.set_page_config(page_title="Anomaly Detection App", layout="wide")
# Translation dictionary
translations = {
    "en": {
        "Upload": "Upload",
        "Data Overview": "Data Overview",
        "Preprocessing": "Preprocessing",
        "Missing Data Analysis": "Missing Data Analysis",
        "Anomaly Detection": "Anomaly Detection",
        "Standard/Basic": "Standard/Basic",
        "Advanced": "Advanced",
        "Next ➡": "Next ➡",
        "⬅ Back": "⬅ Back",
        "Download": "📥 Download",
        "No anomalies": "No anomalies detected.",
        "Anomalies found": "{} anomalies found",
        "Select a column to visualize": "Select a column to visualize",
        "Select detection method": "Select detection method",
        "Encoding Dimension": "Encoding Dimension",
        "Dropout Rate": "Dropout Rate",
        "Epochs": "Epochs",
        "Threshold": "Threshold",
        "Train Autoencoder": "Train Autoencoder",
        "Autoencoder Reconstruction Error Histogram": "Autoencoder Reconstruction Error Histogram",
        "Rule-based Anomaly Detection": "Rule-based Anomaly Detection",
        "Detected anomalies:": "Detected anomalies:",
    },
    "sw": {
        "Upload": "Pakia",
        "Data Overview": "Muhtasari wa Data",
        "Preprocessing": "Usafishaji",
        "Missing Data Analysis": "Uchanganuzi wa Upungufu",
        "Anomaly Detection": "Ugunduzi wa Shida",
        "Standard/Basic": "Kawaida/Msingi",
        "Advanced": "Kisasa",
        "Next ➡": "Ifuatayo ➡",
        "⬅ Back": "⬅ Nyuma",
        "Download": "📥 Pakua",
        "No anomalies": "Hakuna shida zilizogunduliwa.",
        "Anomalies found": "{} shida zimegunduliwa",
        "Select a column to visualize": "Chagua safu ya kuonyesha",
        "Select detection method": "Chagua njia ya ugunduzi",
        "Encoding Dimension": "Kipimo cha Uthibitishaji",
        "Dropout Rate": "Kiwango cha Kukosa",
        "Epochs": "Vipindi",
        "Threshold": "Kiwango cha Kuingilia",
        "Train Autoencoder": "Fanya Mafunzo ya Autoencoder",
        "Autoencoder Reconstruction Error Histogram": "Histogramu ya Makosa ya Ujenzi wa Autoencoder",
        "Rule-based Anomaly Detection": "Ugunduzi wa Shida kwa Kanuni",
        "Detected anomalies:": "Shida zilizogunduliwa:",
    },
    "fr": {
        "Upload": "Téléverser",
        "Data Overview": "Vue d'ensemble des données",
        "Preprocessing": "Prétraitement",
        "Missing Data Analysis": "Analyse des données manquantes",
        "Anomaly Detection": "Détection d'anomalies",
        "Standard/Basic": "Standard/Basique",
        "Advanced": "Avancé",
        "Next ➡": "Suivant ➡",
        "⬅ Back": "⬅ Retour",
        "Download": "📥 Télécharger",
        "No anomalies": "Aucune anomalie détectée.",
        "Anomalies found": "{} anomalies détectées",
        "Select a column to visualize": "Sélectionnez une colonne à visualiser",
        "Select detection method": "Sélectionnez la méthode de détection",
        "Encoding Dimension": "Dimension de codage",
        "Dropout Rate": "Taux d'abandon",
        "Epochs": "Époques",
        "Threshold": "Seuil",
        "Train Autoencoder": "Entraîner l'Autoencodeur",
        "Autoencoder Reconstruction Error Histogram": "Histogramme des erreurs de reconstruction de l'autoencodeur",
        "Rule-based Anomaly Detection": "Détection d'anomalies basée sur des règles",
        "Detected anomalies:": "Anomalies détectées :",
    }
}

def T(text):
    lang = st.session_state.get("language", "en")
    return translations.get(lang, translations["en"]).get(text, text)

# Page config


# Sidebar language selector
lang = st.sidebar.selectbox("🌐 Language / Lugha / Langue", ["en", "sw", "fr"], format_func=lambda x: translations[x]["Upload"])
st.session_state["language"] = lang

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = {}

# --- Upload tab ---

def upload_data():
    st.header("Upload")
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)  # or pd.read_excel
        st.session_state.data = df
        st.session_state.upload_complete = True
        st.success("Data uploaded successfully!")

# --- Data overview tab ---
def data_overview():
    st.header(T("Data Overview"))
    df = st.session_state.data
    if df is None:
        st.warning(T("Upload") + " your dataset first.")
        return

    st.subheader(T("Column Types"))
    type_info = pd.DataFrame({"Column": df.columns, "Current Type": [df[col].dtype for col in df.columns]})
    st.dataframe(type_info)

    st.subheader(T("Summary Statistics (Numerical Variables)"))
    st.dataframe(df.describe())

    st.subheader(T("Select a column to visualize"))
    selected_column = st.selectbox(T("Select a column to visualize"), df.columns)
    if selected_column:
        if pd.api.types.is_numeric_dtype(df[selected_column]):
            fig, ax = plt.subplots()
            sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
            st.pyplot(fig)
        else:
            fig = px.histogram(df, x=selected_column)
            st.plotly_chart(fig)
# Clean column names

# --- Preprocessing tab ---
def preprocessing():
    st.header("Preprocessing")

    if not st.session_state.upload_complete:
        st.warning("Please upload data first.")
        return

    df = st.session_state.data

    # --- DATA CLEANING ---
    st.subheader(T("Data Cleaning"))
    changes = []

    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    changes.append("Stripped and standardized column names")

    gender_columns = [col for col in df.columns if 'gender' in col]
    for col in gender_columns:
        df.rename(columns={col: 'sex'}, inplace=True)
        changes.append(f"Renamed column '{col}' to 'sex'")

    if 'age' not in df.columns:
        age_columns = [col for col in df.columns if 'age' in col]
        if age_columns:
            df.rename(columns={age_columns[0]: 'age'}, inplace=True)
            changes.append(f"Renamed column '{age_columns[0]}' to 'age'")

    if changes:
        st.write("✅ Cleaning steps applied:")
        for change in changes:
            st.markdown(f"- {change}")
    else:
        st.info("No automatic cleaning changes were made.")

    st.session_state.data = df

    # --- VARIABLE TYPE CONVERSION ---
    st.subheader("🔄 Variable Type Conversion")
    selected_type_col = st.selectbox("Select a column to change its type", df.columns)
    current_dtype = df[selected_type_col].dtype

    target_dtype = st.selectbox(
        f"Convert column '{selected_type_col}' from {current_dtype} to:",
        ["int", "float", "str", "bool", "category"]
    )

    if st.button("Apply Type Conversion"):
        try:
            df[selected_type_col] = df[selected_type_col].astype(target_dtype)
            st.success(f"✅ Converted column '{selected_type_col}' to type '{target_dtype}'")
        except Exception as e:
            st.error(f"❌ Conversion failed: {e}")

    st.session_state.data = df

    # --- DROP COLUMNS ---
    st.subheader("🗑️ Drop Variables")
    columns_to_drop = st.multiselect("Select columns to drop from the dataset", df.columns)

    if columns_to_drop and st.button("Drop Selected Columns"):
        df.drop(columns=columns_to_drop, inplace=True)
        st.success(f"✅ Dropped columns: {', '.join(columns_to_drop)}")
        st.session_state.data = df

    # --- ENCODING CATEGORICAL VARIABLES ---
    st.subheader("🔁 Encode Categorical Variables")
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if categorical_cols:
        selected_col = st.selectbox("Choose a categorical column to encode", categorical_cols)
        encoding_type = st.radio(
            "Select encoding type",
            ["Label Encoding (Ordinal)", "One-Hot Encoding (Nominal)"]
        )

        if encoding_type and st.button("Apply Encoding"):
            if encoding_type.startswith("Label"):
                le = LabelEncoder()
                df[selected_col] = le.fit_transform(df[selected_col].astype(str))
                st.success(f"✅ Label encoding applied to '{selected_col}'")
            elif encoding_type.startswith("One-Hot"):
                df = pd.get_dummies(df, columns=[selected_col], drop_first=True)
                st.success(f"✅ One-hot encoding applied to '{selected_col}'")

        st.warning("💡 Use Label Encoding for **ordinal** variables and One-Hot Encoding for **nominal** ones.")
    else:
        st.info("No categorical columns found for encoding.")

    st.session_state.data = df

    # --- MISSING DATA ANALYSIS ---
    st.subheader(T("Missing Data Analysis"))
    st.write(df.isnull().sum())

    st.markdown("### Missing Data Pattern")
    msno.matrix(df)
    st.pyplot(plt.gcf())
    plt.clf()

    # --- MISSINGNESS MECHANISM TEST ---
    
    # --- GLOBAL MISSING DATA STRATEGY ---
    st.subheader(T("Global Missing Data Handling Strategy"))
    global_method = st.selectbox("Choose a default imputation method for all variables",
                                 ["None", "Mean", "Median", "Mode", "KNN", "Drop Row"])

    # --- COLUMN-WISE IMPUTATION ---
    st.subheader(T("Imputation per Column"))
    for col in numeric_cols_with_na:
        col_method = st.selectbox(f"{T('Imputation method for')} {col}",
                                  ["Default (Global)", "Mean", "Median", "Mode", "KNN", "Drop Row"],
                                  key=col)

        if st.button(f"Apply {col_method} to {col}", key="btn_" + col):
            method_to_apply = global_method if col_method == "Default (Global)" else col_method

            if method_to_apply == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif method_to_apply == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif method_to_apply == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method_to_apply == "KNN":
                knn = KNNImputer()
                df[df.columns] = knn.fit_transform(df)
            elif method_to_apply == "Drop Row":
                df.dropna(subset=[col], inplace=True)
            elif method_to_apply == "None":
                pass  # Do nothing

            st.success(f"{method_to_apply} imputation applied to {col}")

    st.session_state.data = df
    st.session_state.preprocessing_complete = True
def detect_rule_based_anomalies(df):
    df = df.copy()
    anomalies = pd.Series([False] * len(df), index=df.index)
    reasons = {i: [] for i in df.index}  # Dictionary to track reasons for each row

    def col_exists(*cols):
        return all(col in df.columns for col in cols)

    # Clean up types
    df["systolic_bp"] = pd.to_numeric(df.get("systolic_bp", pd.NA), errors="coerce")
    df["diastolic_bp"] = pd.to_numeric(df.get("diastolic_bp", pd.NA), errors="coerce")

    if col_exists('hemoglobin'):
        condition = df['hemoglobin'] <= 0
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("Hemoglobin ≤ 0")

    if col_exists('glucose'):
        condition = df['glucose'] <= 0
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("Glucose ≤ 0")

    if col_exists('spo2'):
        condition = df['spo2'] <= 0
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("SpO2 ≤ 0")

    if col_exists('systolic_bp', 'diastolic_bp'):
        condition = df['diastolic_bp'] > df['systolic_bp']
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("Diastolic BP > Systolic BP")

    if col_exists('sex', 'pregnant'):
        condition = (df['sex'].astype(str).str.lower() == 'male') & (df['pregnant'] == True)
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("Male marked as pregnant")

    if col_exists('sex', 'bph'):
        condition = (df['sex'].astype(str).str.lower() == 'female') & (df['bph'] == True)
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("Female marked as having BPH")

    if col_exists('age', 'pregnant'):
        condition_young = (df['age'] < 5) & (df['pregnant'] == True)
        condition_old = (df['age'] > 70) & (df['pregnant'] == True)
        anomalies |= condition_young | condition_old
        for i in df[condition_young].index:
            reasons[i].append("Pregnant but age < 5")
        for i in df[condition_old].index:
            reasons[i].append("Pregnant but age > 70")

    if col_exists('dob', 'dod'):
        dob = pd.to_datetime(df['dob'], errors='coerce')
        dod = pd.to_datetime(df['dod'], errors='coerce')
        condition = dob > dod
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("DOB after DOD")

    if col_exists('admission_date', 'discharge_date'):
        adm = pd.to_datetime(df['admission_date'], errors='coerce')
        dis = pd.to_datetime(df['discharge_date'], errors='coerce')
        condition = dis < adm
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("Discharge before admission")

    if col_exists('pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy', 'sex'):
        condition_f = (df['pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy'] == 3) & \
                      (df['sex'].astype(str).str.upper() == 'F')
        condition_m = df['pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy'].isin([1, 2]) & \
                      (df['sex'].astype(str).str.upper() == 'M')
        anomalies |= condition_f | condition_m
        for i in df[condition_f].index:
            reasons[i].append("Female marked as death due to pregnancy-related cause (code 3)")
        for i in df[condition_m].index:
            reasons[i].append("Male marked as pregnant/death related to pregnancy (code 1/2)")

    if col_exists('age', 'dob', 'dod'):
        try:
            yob = pd.to_datetime(df['dob'], errors='coerce').dt.year
            yod = pd.to_datetime(df['dod'], errors='coerce').dt.year
            age_calc = yod - yob
            age = pd.to_numeric(df['age'], errors='coerce')
            condition = age != age_calc
            anomalies |= condition
            for i in df[condition].index:
                reasons[i].append("Reported age does not match DOB-DOD")
        except Exception as e:
            print(f"Error in age calculation: {e}")

    reasons_final = {i: "; ".join(reasons[i]) for i in df[anomalies].index}
    return anomalies, reasons_final


# --- Basic anomaly detection ---
def basic_anomaly_detection():
    st.header(T("Standard/Basic"))

    if not st.session_state.preprocessing_complete:
        st.warning("Please complete preprocessing first.")
        return

    df = st.session_state.data
    numeric_df = df.select_dtypes(include=np.number)

    method = st.selectbox(T("Select detection method"), 
                          ["Z-score", "IQR", "Median Absolute Deviation", "Mahalanobis Distance", "Rule-based"])
    outliers = pd.Series(False, index=df.index)
    reasons = {}

    if method == "Z-score":
        threshold = st.slider("Z-score Threshold", 2.0, 5.0, 3.0)
        z_scores = np.abs(stats.zscore(numeric_df))
        outliers = (z_scores > threshold).any(axis=1)

    elif method == "IQR":
        q1 = numeric_df.quantile(0.25)
        q3 = numeric_df.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = ((numeric_df < lower_bound) | (numeric_df > upper_bound)).any(axis=1)

    elif method == "Median Absolute Deviation":
        med = numeric_df.median()
        mad = (np.abs(numeric_df - med)).median()
        threshold = st.slider("MAD Threshold", 2.0, 5.0, 3.0)
        mad_score = np.abs(numeric_df - med) / mad
        outliers = (mad_score > threshold).any(axis=1)

    elif method == "Mahalanobis Distance":
        robust_cov = MinCovDet().fit(numeric_df)
        mahal_dist = robust_cov.mahalanobis(numeric_df)
        threshold = st.slider("Mahalanobis Distance Threshold", float(np.percentile(mahal_dist, 90)),
                              float(np.max(mahal_dist)), float(np.percentile(mahal_dist, 95)))
        outliers = mahal_dist > threshold

    elif method == "Rule-based":
        outliers, reasons = detect_rule_based_anomalies(df)

    st.write(f"{T('Anomalies found')}: {outliers.sum()}")

    if method == "Rule-based" and reasons:
        df_outliers = df[outliers].copy()
        df_outliers["reason"] = df_outliers.index.map(reasons)
        st.dataframe(df_outliers)
    else:
        st.dataframe(df.loc[outliers])

    if st.button(T("Download") + " CSV - Basic Anomalies"):
        if method == "Rule-based" and reasons:
            csv = df_outliers.to_csv(index=False).encode()
        else:
            csv = df.loc[outliers].to_csv(index=False).encode()
        st.download_button(label=T("Download"), data=csv, file_name='basic_anomalies.csv', mime='text/csv')

# --- Autoencoder anomaly detection ---
def autoencoder_anomaly_detection():
    st.header(T("Advanced") + " - Autoencoder")
    df = st.session_state.data
    if df is None:
        st.warning(T("Upload") + " your dataset first.")
        return

    numeric_df = df.select_dtypes(include=[np.number]).dropna()
    if numeric_df.empty:
        st.warning("No numeric columns available for autoencoder.")
        return

    encoding_dim = st.slider(T("Encoding Dimension"), min_value=2, max_value=64, value=16)
    dropout_rate = st.slider(T("Dropout Rate"), 0.0, 0.5, 0.0)
    epochs = st.slider(T("Epochs"), 5, 100, 20)

    # Build autoencoder
    input_dim = numeric_df.shape[1]
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(input_layer)
    if dropout_rate > 0:
        encoded = Dropout(dropout_rate)(encoded)
    decoded = Dense(input_dim, activation="linear")(encoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    if st.button(T("Train Autoencoder")):
        history = autoencoder.fit(numeric_df, numeric_df,
                                  epochs=epochs,
                                  batch_size=32,
                                  shuffle=True,
                                  verbose=0)
        st.success("Training finished!")

        reconstructions = autoencoder.predict(numeric_df)
        mse = np.mean(np.square(numeric_df - reconstructions), axis=1)

        threshold = st.slider(T("Threshold"), float(np.min(mse)), float(np.max(mse)), float(np.percentile(mse, 95)))
        anomalies = mse > threshold

        st.write(T("Anomalies found").format(np.sum(anomalies)))
        st.dataframe(df.loc[anomalies])

        fig, ax = plt.subplots()
        ax.hist(mse, bins=50)
        ax.axvline(threshold, color='r', linestyle='--')
        ax.set_title(T("Autoencoder Reconstruction Error Histogram"))
        st.pyplot(fig)

        if st.button(T("Download") + " CSV - Autoencoder Anomalies"):
            csv = df.loc[anomalies].to_csv(index=False).encode()
            st.download_button(label=T("Download"), data=csv, file_name='autoencoder_anomalies.csv', mime='text/csv')






   
# --- Main tabs ---
tab_list = [
    T("Upload"),
    T("Data Overview"),
    T("Preprocessing"),
    T("Standard/Basic"),
    T("Advanced"),
    )
]

tabs = st.tabs(tab_list)

with tabs[0]:
    upload_data()

with tabs[1]:
    data_overview()

with tabs[2]:
    preprocessing()

with tabs[3]:
    basic_anomaly_detection()

with tabs[4]:
    autoencoder_anomaly_detection()


