import os
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Import your modules
from overview import Customer_overview
from models import load_model, save_model, winsorize, retrain_model
from prediction import FNOL_prediction
from retrain_dashboard import show_retraining_ui
from visualization import visualization_dashboard

# Load environment variables
load_dotenv(override=True)

# ----------------- Data Loading -----------------
@st.cache_data
def load_claims_data():
    """
    Load FNOL claims CSV safely.
    Uses relative path for portability.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "FNOL_DATA", "Claims_Policy_merged_cleaned.csv")

    if not os.path.exists(data_path):
        st.error(f"Claims CSV not found at: {data_path}")
        st.stop()
    return pd.read_csv(data_path)


# ----------------- Main App -----------------
def main():
    st.set_page_config(
        page_title="FNOL Claims Dashboard",
        page_icon="🏦",
        layout="wide"
    )

    # Load data (cached)
    if "claims_data" not in st.session_state:
        st.session_state["claims_data"] = load_claims_data()
    claims_data = st.session_state["claims_data"]

    # ---------------- Sidebar Navigation ----------------
    st.sidebar.title("Navigation")
    app_sections = {
        "🏠 Claim Overview": lambda: Customer_overview(Claims_df=claims_data),
        "📊 Visualizations": lambda: visualization_dashboard(claims_data=claims_data),
        "🧮 FNOL Prediction": lambda: FNOL_prediction(claims_data=claims_data),
        "🔄 Model Retraining": show_retraining_ui
    }
    selection = st.sidebar.radio("Go to", list(app_sections.keys()))

    # Execute selected section
    app_sections[selection]()


# ----------------- Run App -----------------
if __name__ == "__main__":
    main()
