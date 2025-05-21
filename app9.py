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


# -----------------------------
# Streamlit UI Logic - Tabs
# -----------------------------
st.title("TafitiX")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = T("Upload")

breadcrumb = f"You are here: ➤ <span>{st.session_state.active_tab}</span>"
st.markdown(f'<div class="breadcrumb">{breadcrumb}</div>', unsafe_allow_html=True)

tabs = [T("Upload"),T("Preprocessing"), T("Missing Data Analysis"),T("Data Overview"), T("Anomaly Detection")]
tab_selection = st.sidebar.radio("Navigation", tabs)
st.session_state.active_tab = tab_selection

# Upload Tab
if st.session_state.active_tab == T("Upload"):
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("File uploaded.")
        st.dataframe(df.head(10))

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
            st.dataframe(df.head(10))

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
# Anomaly Detection Tab

elif st.session_state.active_tab == T("Anomaly Detection"):
    st.subheader("Anomaly Detection")
    if "df_imputed" in st.session_state:
        df = st.session_state["df_imputed"]
    else:
        st.warning("Please complete the data imputation step before detecting anomalies.")
        st.stop()
    detection_method = st.radio("Select Anomaly Detection Method", ["Rule-Based", "Autoencoder", "Statistical"])
    if detection_method == "Rule-Based":
        anomalies, reasons = detect_rule_based_anomalies(df.copy())
        anomaly_df = df[anomalies].copy()
        anomaly_df["Reason"] = [reasons[i] for i in anomaly_df.index]
    
        st.markdown(f"**Detected {len(anomaly_df)} anomalies**")
        st.dataframe(anomaly_df)
    

        csv = anomaly_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Rule-Based Anomalies", data=csv, file_name="rule_based_anomalies.csv", mime="text/csv")

    elif detection_method == "Autoencoder":
        batch_size = 32
        dropout_rate = 0.0
        encoding_dim = 16
        epochs = 20
    
        # Scale data and train autoencoder
        from sklearn.preprocessing import StandardScaler
        from tensorflow.keras.models import Model
        from tensorflow.keras.layers import Input, Dense, Dropout
        import tensorflow as tf
        import plotly.express as px
    
        # Filter numeric columns only
        numeric_cols = df.select_dtypes(include='number').columns
        df_numeric = df[numeric_cols].dropna()
    
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_numeric)
    
        input_dim = X_scaled.shape[1]
        input_layer = Input(shape=(input_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_layer)
        if dropout_rate > 0:
            encoded = Dropout(dropout_rate)(encoded)
        decoded = Dense(input_dim, activation='linear')(encoded)
    
        autoencoder = Model(inputs=input_layer, outputs=decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(X_scaled, X_scaled, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)
    
        # Get reconstruction error
        X_pred = autoencoder.predict(X_scaled)
        mse = np.mean(np.square(X_scaled - X_pred), axis=1)
    
        st.markdown("### MSE Distribution (Reconstruction Error)")
        mse_df = pd.DataFrame({'Reconstruction Error': mse})
        fig_mse = px.histogram(mse_df, x='Reconstruction Error', nbins=50, title="Reconstruction Error Distribution")
        st.plotly_chart(fig_mse)

       # fig_mse = px.histogram(mse, nbins=50, title="Reconstruction Error Distribution")
        #st.plotly_chart(fig_mse)
    
        # Thresholding Method Selection
        st.markdown("### Thresholding")
        thershold_method = st.radio("Select method to determine threshold", ["Manual (slider)", "Z-score", "IQR"])
    
        if thershold_method == "Manual (slider)":
            threshold = st.slider("Set anomaly threshold (between 0 and 1)", 0.0, 1.0, 0.05)
        elif thershold_method == "Z-score":
            z_scores = (mse - np.mean(mse)) / np.std(mse)
            z_thresh = st.slider("Set Z-score threshold", 0.0, 5.0, 3.0)
            threshold = np.percentile(mse, 100 * (1 - np.mean(z_scores > z_thresh)))
        elif thershold_method == "IQR":
            q1, q3 = np.percentile(mse, [25, 75])
            iqr = q3 - q1
            iqr_thresh = st.slider("Set IQR multiplier", 1.0, 3.0, 1.5)
            threshold = q3 + iqr_thresh * iqr
    
        anomalies = mse > threshold
        anomaly_df = df_numeric[anomalies]
        st.markdown(f"**Anomalies Detected:** {anomalies.sum()} rows")
        st.dataframe(anomaly_df)
    
        # Allow download of anomalies
        csv = anomaly_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Anomalies", data=csv, file_name="anomalies.csv", mime="text/csv")
    
        # General Visualization
        st.markdown("### Data Distribution Explorer")
        selected_var = st.selectbox("Select a variable to visualize", df.columns)
        import plotly.express as px
        if pd.api.types.is_numeric_dtype(df[selected_var]):
            plot_type = st.radio("Plot type", ["Histogram", "Line"])
            if plot_type == "Histogram":
                fig = px.histogram(df, x=selected_var, title=f"{selected_var} Histogram")
            else:
                fig = px.line(df, y=selected_var, title=f"{selected_var} Line Plot")
        else:
            plot_type = st.radio("Plot type", ["Bar", "Pie"])
            value_counts = df[selected_var].value_counts().reset_index()
            value_counts.columns = [selected_var, 'Count']
            if plot_type == "Bar":
                fig = px.bar(value_counts, x=selected_var, y='Count', title=f"{selected_var} Distribution")
            else:
                fig = px.pie(value_counts, names=selected_var, values='Count', title=f"{selected_var} Pie Chart")
        st.plotly_chart(fig)
    elif detection_method == "Statistical":
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        selected_var = st.selectbox("Select numeric variable to analyze", numeric_cols)
    
        st.markdown("### Distribution Overview")
        fig_hist = px.histogram(df, x=selected_var, nbins=50, title=f"{selected_var} Distribution")
        st.plotly_chart(fig_hist)
    
        stat_method = st.radio("Select statistical method", ["Z-score", "Median", "IQR"])
    
    if stat_method == "Z-score":
        from scipy.stats import zscore
        z_scores = zscore(df[selected_var].dropna())
        threshold = st.slider("Z-score Threshold", 0.0, 5.0, 3.0)
        anomaly_mask = np.abs(z_scores) > threshold
        anomalies = df.loc[anomaly_mask]
        reason = f"Z-score > {threshold}"
    
        st.markdown(f"**Detected {len(anomalies)} anomalies**")
        st.dataframe(anomalies)
        csv = anomalies.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Statistical Anomalies", data=csv, file_name="statistical_anomalies.csv", mime="text/csv")
    
    elif stat_method == "Median":
        threshold = st.slider("Absolute Difference from Median", 0.0, float(df[selected_var].std()), 1.0)
        anomalies = abs(df[selected_var] - df[selected_var].median()) > threshold
        reason = f"Deviation > {threshold} from Median"
    
        anomaly_df = df[anomalies].copy()
        anomaly_df["Reason"] = reason
        st.markdown(f"**Detected {len(anomaly_df)} anomalies**")
        st.dataframe(anomaly_df)
        csv = anomaly_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Statistical Anomalies", data=csv, file_name="statistical_anomalies.csv", mime="text/csv")
    
    elif stat_method == "IQR":
        multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5)
        q1 = df[selected_var].quantile(0.25)
        q3 = df[selected_var].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr
        anomalies = (df[selected_var] < lower) | (df[selected_var] > upper)
        reason = f"Outside IQR x {multiplier}"
    
        anomaly_df = df[anomalies].copy()
        anomaly_df["Reason"] = reason
        st.markdown(f"**Detected {len(anomaly_df)} anomalies**")
        st.dataframe(anomaly_df)
        csv = anomaly_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Statistical Anomalies", data=csv, file_name="statistical_anomalies.csv", mime="text/csv")
    

    anomaly_df = df[anomalies].copy()
    anomaly_df["Reason"] = reason
    st.markdown(f"**Detected {len(anomaly_df)} anomalies**")
    st.dataframe(anomaly_df)

    csv = anomaly_df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Statistical Anomalies", data=csv, file_name="statistical_anomalies.csv", mime="text/csv")

