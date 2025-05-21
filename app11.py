import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import missingno as msno
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from sklearn.ensemble import IsolationForest
from scipy.stats import normaltest, shapiro, anderson
from sklearn.covariance import MinCovDet
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(layout="wide")
import streamlit as st
import pandas as pd
import numpy as np
import io
import tensorflow

# Optional visualization libraries
import matplotlib.pyplot as plt
import seaborn as sns

# For imputation
from sklearn.impute import KNNImputer
import pygments
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
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "_", regex=False)
    changes.append("Normalized column names: replaced spaces and dots with underscores, lowercased")

    # Rename any column that includes 'gender' to 'sex'
    gender_columns = [col for col in df.columns if 'gender' in col]
    for col in gender_columns:
        df.rename(columns={col: 'sex'}, inplace=True)
        changes.append(f"Renamed column '{col}' to 'sex'")

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
    reasons = {i: [] for i in df.index}  # Dictionary to track reasons for each row

    def col_exists(*cols):
        return all(col in df.columns for col in cols)

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

    if col_exists('systolic', 'dystolic'):
        condition = df['dystolic'] > df['systolic']
        anomalies |= condition
        for i in df[condition].index:
            reasons[i].append("Dystolic > Systolic")

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
            reasons[i].append("Female marked as death due to pregnancy-related causes (code 3)")
        for i in df[condition_m].index:
            reasons[i].append("Male marked as pregnant/died due to pregnancy-related causes (code 1/2)")

    # Only keep reasons for rows where anomalies == True
    reasons_final = {i: "; ".join(reasons[i]) for i in df[anomalies].index}

    return anomalies, reasons_final

# Initialize session state for data
if 'data' not in st.session_state:
    st.session_state.data = None

if 'anomalies' not in st.session_state:
    st.session_state.anomalies = {}

def upload_data():
    st.subheader("Upload Your Dataset")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.session_state.data = df
        st.success("Data uploaded successfully!")

def data_overview():
    st.subheader("Data Overview")
    df = st.session_state.data
    if df is not None:
        st.markdown("### Column Types")
        type_info = pd.DataFrame({"Column": df.columns, "Current Type": [df[col].dtype for col in df.columns]})
        st.dataframe(type_info)

        st.markdown("### Summary Statistics (Numerical Variables)")
        st.dataframe(df.describe())

        st.markdown("### Variable Visualization")
        selected_column = st.selectbox("Select a column to visualize", df.columns)
        if selected_column:
            if pd.api.types.is_numeric_dtype(df[selected_column]):
                fig, ax = plt.subplots()
                sns.histplot(df[selected_column].dropna(), kde=True, ax=ax)
                st.pyplot(fig)
            else:
                fig = px.histogram(df, x=selected_column)
                st.plotly_chart(fig)
    else:
        st.warning("Please upload a dataset first.")

def preprocess_data():
    st.subheader("Preprocessing")
    df = st.session_state.data

    st.markdown("#### Missing Value Count")
    st.write(df.isnull().sum())

    st.markdown("#### Missing Value Pattern (MICE Visualization)")
    msno.matrix(df)
    st.pyplot(plt.gcf())

    st.markdown("#### Statistical Tests for Missingness Type")
    missing_tests = {}
    for col in df.columns:
        if df[col].isnull().sum() > 0 and pd.api.types.is_numeric_dtype(df[col]):
            group = df[col].isnull()
            for other_col in df.columns:
                if other_col != col and pd.api.types.is_numeric_dtype(df[other_col]):
                    t_stat, p_value = stats.ttest_ind(df[other_col][group], df[other_col][~group], nan_policy='omit')
                    if p_value < 0.05:
                        missing_tests[col] = "Potential MAR/NMAR"
                        break
            else:
                missing_tests[col] = "Potential MCAR"
    st.write(missing_tests)

    st.markdown("#### Missing Data Imputation")
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            st.markdown(f"**{col}**")
            method = st.selectbox(f"Imputation method for {col}", ["Mean", "Median", "Mode", "KNN", "Drop Row"], key=col)
            if method == "Mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "Median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "Mode":
                df[col].fillna(df[col].mode()[0], inplace=True)
            elif method == "KNN":
                knn = KNNImputer()
                df[df.columns] = knn.fit_transform(df[df.columns])
                break
            elif method == "Drop Row":
                df.dropna(subset=[col], inplace=True)
    st.success("Imputation complete.")
    st.session_state.data = df

def basic_anomaly_detection():
    st.subheader("Standard/Basic Anomaly Detection")
    df = st.session_state.data.select_dtypes(include=[np.number]).dropna()

    st.markdown("### Normality Tests")
    for col in df.columns:
        st.markdown(f"**{col}**")
        stat, p = shapiro(df[col])
        st.write(f"Shapiro-Wilk: p = {p:.4f} ({'Non-normal' if p < 0.05 else 'Normal'})")

    st.markdown("### Outlier Detection")
    method = st.selectbox("Select detection method", ["Z-score", "IQR", "Median Deviation", "Mahalanobis"])
    all_outliers = pd.DataFrame()
    for col in df.columns:
        st.markdown(f"**{col}**")
        if method == "Z-score":
            z = np.abs(stats.zscore(df[col]))
            outliers = df[z > 3]
        elif method == "IQR":
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5 * IQR) | (df[col] > Q3 + 1.5 * IQR)]
        elif method == "Median Deviation":
            med = np.median(df[col])
            mad = np.median(np.abs(df[col] - med))
            modified_z = 0.6745 * (df[col] - med) / mad
            outliers = df[np.abs(modified_z) > 3.5]
        elif method == "Mahalanobis":
            cov = MinCovDet().fit(df)
            mahal = cov.mahalanobis(df)
            outliers = df[mahal > np.percentile(mahal, 97.5)]

        all_outliers = pd.concat([all_outliers, outliers])
        st.write(f"Outliers detected: {len(outliers)}")
        fig = px.histogram(df, x=col)
        st.plotly_chart(fig)

    if not all_outliers.empty:
        st.download_button("Download Detected Anomalies", all_outliers.drop_duplicates().to_csv(index=False), "basic_anomalies.csv")
        st.session_state.anomalies['basic'] = all_outliers.drop_duplicates()

def autoencoder_anomaly_detection():
    st.subheader("Autoencoder-Based Detection")
    df = st.session_state.data.select_dtypes(include=[np.number]).dropna()

    encoding_dim = st.number_input("Encoding Dimension", value=16)
    dropout_rate = st.slider("Dropout Rate", 0.0, 0.5, 0.0)
    epochs = st.number_input("Epochs", value=20)
    batch_size = st.number_input("Batch Size", value=32)

    X = df.values
    input_dim = X.shape[1]

    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation='relu')(input_layer)
    if dropout_rate > 0:
        encoded = Dropout(dropout_rate)(encoded)
    decoded = Dense(input_dim, activation='linear')(encoded)

    autoencoder = Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.fit(X, X, epochs=int(epochs), batch_size=int(batch_size), shuffle=True, verbose=0)

    reconstructed = autoencoder.predict(X)
    mse = np.mean(np.power(X - reconstructed, 2), axis=1)

    st.markdown("### Thresholding Method")
    method = st.selectbox("Select thresholding method", ["Z-score", "IQR"], key="ae_thresh")
    if method == "Z-score":
        z_scores = stats.zscore(mse)
        threshold = 3
        anomalies = df[np.abs(z_scores) > threshold]
    else:
        q1 = np.percentile(mse, 25)
        q3 = np.percentile(mse, 75)
        iqr = q3 - q1
        threshold = q3 + 1.5 * iqr
        anomalies = df[mse > threshold]

    st.markdown(f"**Anomalies Detected: {len(anomalies)}**")
    fig, ax = plt.subplots()
    sns.histplot(mse, bins=50, kde=True, ax=ax)
    st.pyplot(fig)

    # PCA Visualization
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(X)
    df_viz = pd.DataFrame(pca_result, columns=["PC1", "PC2"])
    df_viz["Anomaly"] = mse > threshold
    fig = px.scatter(df_viz, x="PC1", y="PC2", color="Anomaly", title="Anomaly Visualization (PCA)")
    st.plotly_chart(fig)

    # Feature Importance via Random Forest
    y = (mse > threshold).astype(int)
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)
    importances = rf.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    feat_df = pd.DataFrame({"Feature": df.columns[sorted_idx], "Importance": importances[sorted_idx]})
    st.markdown("### Feature Importance")
    st.bar_chart(feat_df.set_index("Feature"))

    if not anomalies.empty:
        st.download_button("Download Detected Anomalies", anomalies.to_csv(index=False), "autoencoder_anomalies.csv")
        st.session_state.anomalies['autoencoder'] = anomalies

# ... (no changes to isolation_forest_detection or layout below)

# Define tab layout
main_tabs = st.tabs(["1. Upload", "2. Data Overview", "3. Data Cleaning and Preprocessing", "4. Anomaly Detection"])

with main_tabs[0]:
    upload_data()

with main_tabs[1]:
    if st.session_state.data is not None:
        data_overview()
    else:
        st.warning("Please upload data in the Upload tab first.")

with main_tabs[2]:
    if st.session_state.data is not None:
        subtab = st.radio("Choose a section:", ["Data Cleaning", "Preprocessing"])
        if subtab == "Data Cleaning":
            st.markdown("### Data Cleaning")
            st.info("This section will include all existing basic preprocessing functionality.")
        elif subtab == "Preprocessing":
            preprocess_data()
    else:
        st.warning("Please upload data in the Upload tab first.")

with main_tabs[3]:
    if st.session_state.data is not None:
        subtab = st.radio("Choose detection method:", ["Standard/Basic", "Advanced"])
        if subtab == "Standard/Basic":
            basic_anomaly_detection()
        else:
            st.markdown("### Advanced Anomaly Detection")
            autoencoder_anomaly_detection()
            st.markdown("---")
            isolation_forest_detection()
    else:
        st.warning("Please upload data in the Upload tab first.")
