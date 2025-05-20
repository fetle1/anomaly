import streamlit as st
import pandas as pd
import numpy as np
import io
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow as tf
import plotly.express as px

# Replace with actual translation if needed
#T = lambda x: x  

# Placeholder anomaly detection functions
def clean_and_preprocess(df):
    changes = []

    # Drop unwanted column
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)
        changes.append("Dropped column: Unnamed: 0")

    # Normalize column names: strip spaces, lowercase, replace spaces and dots with underscores
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_").str.replace(".", "_", regex=False)
    changes.append("Normalized column names: replaced spaces and dots with underscores, lowercased")

    # Rename any column that includes 'gender' to 'sex'
    gender_columns = [col for col in df.columns if 'gender' in col]
    for col in gender_columns:
        df.rename(columns={col: 'sex'}, inplace=True)
        changes.append(f"Renamed column '{col}' to 'sex'")

    # Rename any column that includes 'age' to 'age' (if 'age' is not already in columns)
    if 'age' not in df.columns:
        age_columns = [col for col in df.columns if 'age' in col]
        if age_columns:
            df.rename(columns={age_columns[0]: 'age'}, inplace=True)
            changes.append(f"Renamed column '{age_columns[0]}' to 'age'")

    return df, changes

def merge_and_clean_birth_death_dates(df):
    # Merge year_of_birth, month_of_birth, day_of_birth into dob
    if {'year_of_birth', 'month_of_birth', 'day_of_birth'}.issubset(df.columns):
        birth_df = df[['year_of_birth', 'month_of_birth', 'day_of_birth']].rename(
            columns={
                'year_of_birth': 'year',
                'month_of_birth': 'month',
                'day_of_birth': 'day'
            }
        )
        df['dob'] = pd.to_datetime(birth_df, errors='coerce')
        df.drop(columns=['year_of_birth', 'month_of_birth', 'day_of_birth'], inplace=True)

    # Merge year_of_death, month_of_death, day_of_death into dod
    if {'year_of_death', 'month_of_death', 'day_of_death'}.issubset(df.columns):
        death_df = df[['year_of_death', 'month_of_death', 'day_of_death']].rename(
            columns={
                'year_of_death': 'year',
                'month_of_death': 'month',
                'day_of_death': 'day'
            }
        )
        df['dod'] = pd.to_datetime(death_df, errors='coerce')
        df.drop(columns=['year_of_death', 'month_of_death', 'day_of_death'], inplace=True)

    return df, changes

def detect_rule_based_anomalies(df):
    def col_exists(*cols):
    return all(col in df.columns for col in cols)

def rule_based_anomaly_detection(df):
    anomalies = pd.Series([False] * len(df))

    # Example Rule 1: If age < 10 and parity > 0
    if col_exists('age', 'parity'):
        anomalies |= (df['age'] < 10) & (df['parity'] > 0)

    # Example Rule 2: If female and parity > 20
    if col_exists('sex', 'parity'):
        anomalies |= (df['sex'].astype(str).str.upper() == 'F') & (df['parity'] > 20)

    # Example Rule 3: If male and parity > 0
    if col_exists('sex', 'parity'):
        anomalies |= (df['sex'].astype(str).str.upper() == 'M') & (df['parity'] > 0)

    # Example Rule 4: Female with prostate cancer
    if col_exists('sex', 'diagnosis'):
        anomalies |= (df['sex'].astype(str).str.upper() == 'F') & (df['diagnosis'].astype(str).str.contains('prostate', case=False, na=False))

    # Example Rule 5: Pregnancy related and male
    if col_exists('sex', 'pregnancy_status'):
        anomalies |= (df['sex'].astype(str).str.upper() == 'M') & (df['pregnancy_status'].notna())

    # Additional Rule 6: Check inconsistencies in pregnancy and death within 6 weeks
    if col_exists('pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy', 'sex'):
        anomalies |= (
            (df['pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy'] == 3) &
            (df['sex'].astype(str).str.upper() == 'F')
        )

        anomalies |= (
            (df['pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy'].isin([1, 2])) &
            (df['sex'].astype(str).str.upper() == 'M')
        )
    if col_exists('dob', 'dod'):
          anomalies |= pd.to_datetime(df['dob'], errors='coerce') > pd.to_datetime(df['dod'], errors='coerce')
  
    if col_exists('pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy', 'sex'):
          anomalies |= (
              (df['pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy'] == 3) &
              (df['sex'].astype(str).str.upper() == 'F')
          )
          anomalies |= (
              (df['pregenant_or_died_with_in_six_weeks_of_end_of_pregenancy'].isin([1, 2])) &
              (df['sex'].astype(str).str.upper() == 'M')
          )
    if col_exists('age', 'dob', 'dod'):
            try:
                yob = pd.to_numeric(df['dob'], errors='coerce')
                yod = pd.to_numeric(df['dod'], errors='coerce')
                age_calc = yod - yob
                age = pd.to_numeric(df['age'], errors='coerce')
                anomalies |= (age != age_calc)
            except Exception as e:
                print(f"Error calculating age anomaly: {e}")
    
    return anomalies


def apply_default_strategy(df, opts):
    if opts.get("drop_high_missing_cols"):
        df = df.loc[:, df.isnull().mean() < 0.5]
    if opts.get("drop_low_missing_rows"):
        df = df[df.isnull().mean(axis=1) < 0.01]
    if opts.get("impute_low_missing"):
        for col in df.columns[df.isnull().mean() < 0.05]:
            if df[col].dtype in ['float64', 'int64']:
                df[col].fillna(df[col].mean(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
    if opts.get("impute_moderate_missing"):
        from sklearn.impute import KNNImputer
        knn = KNNImputer(n_neighbors=opts.get("knn_neighbors", 5))
        numeric_cols = df.select_dtypes(include='number').columns
        df[numeric_cols] = knn.fit_transform(df[numeric_cols])
    return df

def impute_mean_mode(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].mean(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def impute_median_mode(df):
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df

def impute_knn_all(df, n_neighbors=5):
    from sklearn.impute import KNNImputer
    knn = KNNImputer(n_neighbors=n_neighbors)
    numeric_cols = df.select_dtypes(include='number').columns
    df[numeric_cols] = knn.fit_transform(df[numeric_cols])
    return df

# --------------------------
# Streamlit App Starts Here
# --------------------------
st.set_page_config(layout="wide")
st.title("TafitiX")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = T("Upload")

tabs = [T("Upload"), T("Preprocessing"), T("Missing Data Analysis"), T("Data Overview"), T("Anomaly Detection")]
tab_selection = st.sidebar.radio("Navigation", tabs)
st.session_state.active_tab = tab_selection

st.markdown(f"You are here: ➤ **{st.session_state.active_tab}**")

# Upload Tab
if st.session_state.active_tab == T("Upload"):
    uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state["df"] = df
        st.success("File uploaded.")
        st.dataframe(df.head())

    if st.button("Next ➡", key="upload_next"):
        st.session_state.active_tab = T("Preprocessing")

# Preprocessing Tab
elif st.session_state.active_tab == T("Preprocessing"):
    if "df" not in st.session_state:
        st.warning("Please upload your data first.")
        st.stop()
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
        convert_col = st.selectbox("Select a column to convert type", df.columns)
        current_type = str(df[convert_col].dtype)
        st.markdown(f"**Current type:** {current_type}")
        convert_to = st.selectbox("Convert to:", ["Keep as is", "float", "int", "category", "string"])
        if convert_to != "Keep as is":
            try:
                if convert_to == "float":
                    df[convert_col] = pd.to_numeric(df[convert_col], errors='coerce').astype(float)
                elif convert_to == "int":
                    df[convert_col] = pd.to_numeric(df[convert_col], errors='coerce').astype("Int64")
                elif convert_to == "category":
                    df[convert_col] = df[convert_col].astype("category")
                elif convert_to == "string":
                    df[convert_col] = df[convert_col].astype(str)
                st.success(f"Converted {convert_col} to {convert_to}")
            except Exception as e:
                st.error(f"Conversion failed: {e}")
            st.session_state["df_processed"] = df

        drop_cols = st.multiselect("Select columns to drop", df.columns)
        if st.button("Apply Drops"):
            if drop_cols:
                df.drop(columns=drop_cols, inplace=True)
                st.success("Dropped selected columns.")
            st.session_state["df_processed"] = df

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back", key="pre_back"):
            st.session_state.active_tab = T("Upload")
    with col2:
        if st.button("Next ➡", key="pre_next"):
            st.session_state.active_tab = T("Data Overview")

# Data Overview Tab
elif st.session_state.active_tab == T("Data Overview"):
    if "df_processed" not in st.session_state:
        st.warning("Please preprocess your data first.")
        st.stop()
    df = st.session_state["df_processed"]

    st.subheader("Summary Statistics")
    st.dataframe(df.describe())

    st.subheader("Visualize a Variable")
    selected_var = st.selectbox("Select a variable", df.columns)
    if pd.api.types.is_numeric_dtype(df[selected_var]):
        st.plotly_chart(px.histogram(df, x=selected_var))
    else:
        vc = df[selected_var].value_counts().reset_index()
        vc.columns = [selected_var, "Count"]
        st.plotly_chart(px.bar(vc, x=selected_var, y="Count"))

    st.subheader("Outlier Detection (IQR)")
    num_col = st.selectbox("Numeric column", df.select_dtypes(include='number').columns)
    Q1, Q3 = df[num_col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    lb, ub = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    outliers = df[(df[num_col] < lb) | (df[num_col] > ub)]
    st.write(f"Outliers detected: {len(outliers)}")
    st.dataframe(outliers)

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back", key="overview_back"):
            st.session_state.active_tab = T("Preprocessing")
    with col2:
        if st.button("Next ➡", key="overview_next"):
            st.session_state.active_tab = T("Missing Data Analysis")

# Missing Data Analysis Tab
elif st.session_state.active_tab == T("Missing Data Analysis"):
    if "df_processed" not in st.session_state:
        st.warning("Please preprocess your data first.")
        st.stop()
    df = st.session_state["df_processed"]

    st.subheader("Missing Summary")
    missing_percent = df.isnull().mean() * 100
    st.dataframe(missing_percent[missing_percent > 0].reset_index().rename(columns={"index": "Column", 0: "Missing %"}))

    st.subheader("Imputation Strategy")
    strategy = st.selectbox("Method", ["Default Strategy", "Mean/Mode", "Median/Mode", "KNN", "Drop Rows"])
    if strategy == "Default Strategy":
        opts = {
            "drop_high_missing_cols": st.checkbox("Drop cols >50%", True),
            "drop_low_missing_rows": st.checkbox("Drop rows <1%", True),
            "impute_low_missing": st.checkbox("Impute <5%", True),
            "impute_moderate_missing": st.checkbox("KNN for 5%-50%", True),
            "knn_neighbors": st.slider("KNN Neighbors", 2, 10, 5)
        }
        df = apply_default_strategy(df, opts)
    elif strategy == "Mean/Mode":
        df = impute_mean_mode(df)
    elif strategy == "Median/Mode":
        df = impute_median_mode(df)
    elif strategy == "KNN":
        df = impute_knn_all(df)
    elif strategy == "Drop Rows":
        df = df.dropna()

    st.success("Imputation complete")
    st.session_state["df_imputed"] = df

    if st.checkbox("Preview Imputed Data"):
        st.dataframe(df.head())

    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    st.download_button("📥 Download Imputed Data", data=csv_buffer.getvalue(), file_name="imputed_data.csv", mime="text/csv")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("⬅ Back", key="impute_back"):
            st.session_state.active_tab = T("Data Overview")
    with col2:
        if st.button("Next ➡", key="impute_next"):
            st.session_state.active_tab = T("Anomaly Detection")

# Anomaly Detection Tab
elif st.session_state.active_tab == T("Anomaly Detection"):
    st.subheader("Anomaly Detection")
    if "df_imputed" not in st.session_state:
        st.warning("Please complete data imputation first.")
        st.stop()
    df = st.session_state["df_imputed"]

    method = st.radio("Method", ["Rule-Based", "Autoencoder", "Statistical"])

    if method == "Rule-Based":
        mask, reasons = detect_rule_based_anomalies(df)
        result = df[mask].copy()
        result["Reason"] = result.index.map(reasons)
        st.write(f"Anomalies found: {len(result)}")
        st.dataframe(result)
        st.download_button("📥 Download Anomalies", result.to_csv(index=False), "rule_based_anomalies.csv")

    elif method == "Autoencoder":
        numeric = df.select_dtypes(include='number').dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(numeric)

        inp = Input(shape=(X_scaled.shape[1],))
        x = Dense(16, activation='relu')(inp)
        out = Dense(X_scaled.shape[1], activation='linear')(x)
        auto = Model(inp, out)
        auto.compile(optimizer='adam', loss='mse')
        auto.fit(X_scaled, X_scaled, epochs=20, batch_size=32, verbose=0)

        pred = auto.predict(X_scaled)
        mse = np.mean((X_scaled - pred) ** 2, axis=1)
        st.plotly_chart(px.histogram(x=mse, nbins=50, title="Reconstruction Error"))

        th_method = st.radio("Thresholding", ["Manual", "Z-score", "IQR"])
        if th_method == "Manual":
            threshold = st.slider("Threshold", 0.0, float(np.max(mse)), float(np.mean(mse)))
        elif th_method == "Z-score":
            z_thresh = st.slider("Z-score", 0.0, 5.0, 3.0)
            threshold = np.mean(mse) + z_thresh * np.std(mse)
        else:
            q1, q3 = np.percentile(mse, [25, 75])
            iqr = q3 - q1
            mult = st.slider("IQR Multiplier", 1.0, 3.0, 1.5)
            threshold = q3 + mult * iqr

        mask = mse > threshold
        result = numeric[mask]
        st.write(f"Anomalies found: {len(result)}")
        st.dataframe(result)
        st.download_button("📥 Download Anomalies", result.to_csv(index=False), "autoencoder_anomalies.csv")

    elif method == "Statistical":
        col = st.selectbox("Select Variable", df.select_dtypes(include='number').columns)
        st.plotly_chart(px.histogram(df, x=col, nbins=50))

        stat = st.radio("Method", ["Z-score", "IQR"])
        if stat == "Z-score":
            z = (df[col] - df[col].mean()) / df[col].std()
            threshold = st.slider("Z-score Threshold", 0.0, 5.0, 3.0)
            mask = abs(z) > threshold
        else:
            q1, q3 = df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            multiplier = st.slider("IQR Multiplier", 1.0, 3.0, 1.5)
            mask = (df[col] < q1 - multiplier * iqr) | (df[col] > q3 + multiplier * iqr)

        result = df[mask]
        st.write(f"Anomalies found: {len(result)}")
        st.dataframe(result)
        st.download_button("📥 Download Anomalies", result.to_csv(index=False), "statistical_anomalies.csv")
