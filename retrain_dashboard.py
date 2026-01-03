import pandas as pd
import numpy as np
import streamlit as st
from models import retrain_model



def show_retraining_ui():
    st.header("🔄 Retraining Dashboard")

    st.markdown("Upload a new csv to retrain the FNOL model:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        new_data = pd.read_csv(uploaded_file)
        st.write("Preview of uploaded data:")
        st.dataframe(new_data.head)

        if st.button("Retrain Model"):
            with st.spinner("Retraining model..."):
                result = retrain_model(new_data)
                st.success("Retraining completed!")

                st.write(f"Old RMSE: {result["rmse_old"]:.2f}")
                st.write(f"New RMSE: {result["rmse_new"]:.2f}")

                if result["promoted"]:
                    st.balloons()
                    st.success("New model promoted to Production")
                else:
                    st.info("New model was NOT better, production model retrained")