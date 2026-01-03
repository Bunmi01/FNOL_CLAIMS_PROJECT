import pandas as pd
import streamlit as st
from overview import Customer_overview
from models import load_model, save_model, winsorize, retrain_model
from prediction import FNOL_prediction
from retrain_dashboard import show_retraining_ui
from visualization import visualization_dashboard
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)


def main():
    st.set_page_config(
        page_title="FNOL Claims Dashboard",
        page_icon="🏦",
        layout="wide"
    )

    # Load claims data if not already in session state
    if "claims_data" not in st.session_state:
        try:
            st.session_state["claims_data"] = pd.read_csv(
                r"C:\Users\adebo\OneDrive\Desktop\FNOL PROJECT\FNOL_DATA\Claims_Policy_merged_cleaned.csv"
            )
        except Exception as e:
            st.error(f"Error loading claims data: {e}")
            st.stop()

    claims_data = st.session_state["claims_data"]

    # Sidebar navigation
    st.sidebar.title("Navigation")
    app_sections = {
        "🏠 Claim Overview": lambda: Customer_overview(Claims_df=claims_data),
        "📊 Visualizations": lambda: visualization_dashboard(claims_data=claims_data),
        "🧮 FNOL Prediction": lambda: FNOL_prediction(claims_data=claims_data),
        "🔄 Model Retraining": show_retraining_ui
    }

    selection = st.sidebar.radio("Go to", list(app_sections.keys()))

    # Call the function for the selected section

    app_sections[selection]()  # Execute the corresponding function


if __name__ == "__main__":
    main()





