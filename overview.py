import streamlit as st
import pandas as pd 

def Customer_overview(Claims_df):
    st.title("🏠 Overview of Customer Claim")
    st.markdown("Detailed overview of Insurance Claims Data and key metrics")

    # ----------- Top KPIs ---------------
    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        min_claim = Claims_df['Ultimate_Claim_Amount'].min()
        st.metric("Lowest Claim Amount", f"£{min_claim:,.2f}")

    with col2:
        max_claim = Claims_df['Ultimate_Claim_Amount'].max()
        st.metric("Highest Claim Amount", f"£{max_claim:,.2f}")

    with col3:
        youngest_driver = Claims_df['Driver_age_(years)'].min()
        st.metric("Youngest Driver", f"{youngest_driver} years")

    with col4:
        oldest_driver = Claims_df['Driver_age_(years)'].max()
        st.metric("Oldest Driver", f"{oldest_driver} years")

    st.markdown("---")


    # ----------- Additional Metrics ---------------
    st.subheader("Additional Metrics")

    col1, col2, col3 = st.columns(3)

    total_estimated_claim = Claims_df["Estimated_Claim_Amount"].sum()
    total_ultimate_claim = Claims_df["Ultimate_Claim_Amount"].sum()
    claim_variance = ((total_ultimate_claim - total_estimated_claim) / total_estimated_claim) * 100

    with col1:
        st.metric("Total Estimated Claims Value", f"£{total_estimated_claim:,.2f}")

    with col2:
        st.metric("Total Ultimate Claims Value", f"£{total_ultimate_claim:,.2f}")

    with col3:
        st.metric("Variance in Claim", f"{claim_variance:+.2f}%")

    st.markdown("---")


    # ---------- Claim Type Analysis ----------
    st.subheader("Claim Type Analysis")

    claim_type_analysis = Claims_df.groupby("Claim_Type").agg({
        "Estimated_Claim_Amount": ["sum", "mean", "count"],
        "Ultimate_Claim_Amount": ["sum", "mean"]
    }).round(2)

    claim_type_analysis.columns = [
        "Total_Est", "Avg_Est", "Claim_Count", "Total_Ult", "Avg_Ult"
    ]
    claim_type_analysis = claim_type_analysis.reset_index()

    # -------- Formatting for Display --------
    formatted_claim_type = claim_type_analysis.copy()

    for col in ["Total_Est", "Avg_Est", "Total_Ult", "Avg_Ult"]:
        formatted_claim_type[col] = formatted_claim_type[col].map(lambda x: f"£{x:,.2f}")

    formatted_claim_type["Claim_Count"] = formatted_claim_type["Claim_Count"].map(lambda x: f"{x:,}")

    formatted_claim_type = formatted_claim_type.rename(columns={
        "Claim_Type": "Claim Type",
        "Total_Est": "Total Estimated Amount (£)",
        "Avg_Est": "Average Estimated Amount (£)",
        "Claim_Count": "Number of Claims",
        "Total_Ult": "Total Ultimate Amount (£)",
        "Avg_Ult": "Average Ultimate Amount (£)"
    })

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Summary of Claim Type**")
        st.dataframe(formatted_claim_type, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Key Insights**")

        highest_claim_type = claim_type_analysis.loc[claim_type_analysis["Total_Ult"].idxmax()]
        lowest_claim_type = claim_type_analysis.loc[claim_type_analysis["Total_Ult"].idxmin()]

        st.info(
            f"The claim type with the highest total value is **{highest_claim_type['Claim_Type']}** "
            f"with **£{highest_claim_type['Total_Ult']:,.2f}**"
        )

        st.warning(
            f"The claim type with the lowest total value is **{lowest_claim_type['Claim_Type']}** "
            f"with **£{lowest_claim_type['Total_Ult']:,.2f}**"
        )

        total_claims_no = claim_type_analysis["Claim_Count"].sum()
        total_claims_value = claim_type_analysis["Total_Est"].sum()

        st.success(
            f"There are **{total_claims_no:,} claims** totalling **£{total_claims_value:,.2f}**"
        )

    st.markdown("---")

    # ----------- Traffic and Weather Analysis ---------------
    st.subheader("Traffic and Weather Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Distribution of Traffic Condition**")
        traffic_df = Claims_df["Traffic_Condition"].value_counts().reset_index()
        traffic_df.columns = ["Traffic Condition", "Number of Claims"]
        st.dataframe(traffic_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Distribution of Weather Condition**")
        weather_df = Claims_df["Weather_Condition"].value_counts().reset_index()
        weather_df.columns = ["Weather Condition", "Number of Claims"]
        st.dataframe(weather_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ----------- Weather Impact Analysis ---------------
    st.subheader("Weather Impact Analysis")

    weather_impact = Claims_df.groupby("Weather_Condition").agg({
        "Estimated_Claim_Amount": ["sum", "mean", "count"],
        "Ultimate_Claim_Amount": ["sum", "mean"]
    }).round(2)

    weather_impact.columns = [
        "Total_Est", "Avg_Est", "Claim_Count", "Total_Ult", "Avg_Ult"
    ]

    weather_impact = weather_impact.reset_index().sort_values("Total_Ult", ascending=False)

    formatted_weather = weather_impact.copy()

    for col in ["Total_Est", "Avg_Est", "Total_Ult", "Avg_Ult"]:
        formatted_weather[col] = formatted_weather[col].map(lambda x: f"£{x:,.2f}")

    formatted_weather["Claim_Count"] = formatted_weather["Claim_Count"].map(lambda x: f"{x:,}")

    formatted_weather = formatted_weather.rename(columns={
        "Weather_Condition": "Weather Condition",
        "Total_Est": "Total Estimated Amount (£)",
        "Avg_Est": "Average Estimated Amount (£)",
        "Claim_Count": "Number of Claims",
        "Total_Ult": "Total Ultimate Amount (£)",
        "Avg_Ult": "Average Ultimate Amount (£)"
    })

    st.dataframe(formatted_weather, use_container_width=True, hide_index=True)
