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
# Language Translation
# -----------------------------
lang = st.sidebar.selectbox("🌍 Select Language", ["English", "Swahili", "French"])

translations = {
    "Upload": {"English": "Upload", "Swahili": "Pakia", "French": "Téléverser"},
    "Preprocessing": {"English": "Preprocessing", "Swahili": "Usafishaji", "French": "Prétraitement"},
    "Missing Data Analysis": {"English": "Missing Data Analysis", "Swahili": "Uchanganuzi wa Data Isiyokuwepo", "French": "Analyse des données manquantes"},
    "Anomaly Detection": {"English": "Anomaly Detection", "Swahili": "Uchunguzi wa Kasoro", "French": "Détection d'anomalies"},
    "Next": {"English": "Next ➡", "Swahili": "Ifuatayo ➡", "French": "Suivant ➡"},
    "Back": {"English": "⬅ Back", "Swahili": "⬅ Nyuma", "French": "⬅ Retour"},
    "Run Rule-Based Anomaly Detection": {
        "English": "Run Rule-Based Anomaly Detection",
        "Swahili": "Endesha Ugunduzi wa Kasoro kwa Kutumia Sheria",
        "French": "Exécuter la détection d'anomalies basée sur des règles"
    },
    "Download": {
        "English": "📥 Download Imputed Data",
        "Swahili": "📥 Pakua Data Iliyokamilishwa",
        "French": "📥 Télécharger les données imputées"
    },
    "No anomalies": {
        "English": "No rule-based anomalies detected.",
        "Swahili": "Hakuna kasoro zilizogunduliwa kwa kutumia sheria.",
        "French": "Aucune anomalie basée sur des règles détectée."
    },
    "Anomalies found": {
        "English": "{} anomalies detected.",
        "Swahili": "Kasoro {} zimegunduliwa.",
        "French": "{} anomalies détectées."
    },
}

def T(key):
    return translations.get(key, {}).get(lang, key)

# -----------------------------
# Utility Functions (keep existing)
# -----------------------------
# [... existing functions remain unchanged ...]

# -----------------------------
# Streamlit UI Logic
# -----------------------------

st.title("TafitiX")

if "active_tab" not in st.session_state:
    st.session_state.active_tab = T("Upload")

breadcrumb = f"You are here: ➤ <span>{st.session_state.active_tab}</span>"
st.markdown(f'<div class="breadcrumb">{breadcrumb}</div>', unsafe_allow_html=True)

tabs = [T("Upload"), T("Preprocessing"), T("Missing Data Analysis"), T("Anomaly Detection")]
tab_selection = st.sidebar.radio("Navigation", tabs)
st.session_state.active_tab = tab_selection

# Update the tab-specific content below accordingly...
