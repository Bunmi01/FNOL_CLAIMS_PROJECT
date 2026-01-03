import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st 



def visualization_dashboard(claims_data):
    st.title("📈 Claims Data Visualization Dashboard")
    st.markdown("### Select a visualization type")

    plot_options = [
        "Categorical Data Distribution", 
        "Monthly Claims Trends", 
        "Claims Amount Analysis", 
        "Driver Demographics"
    ]

    selection = st.selectbox("Choose plot type", plot_options)

    if selection == "Categorical Data Distribution":
        plot_categorical_distributions(claims_data)
    elif selection == "Monthly Claims Trends":
        plot_monthly_claims_settlements(claims_data)
    elif selection == "Claims Amount Analysis":
        plot_claim_amount_distributions(claims_data)
    elif selection == "Driver Demographics":
        plot_age_distributions(claims_data)

    
def plot_categorical_distributions(claims_data):
    """
    Plot distributions of categorical variables
    """    

    st.subheader("📊 Categorical Data Distribution")

    # ---------- PLOTS ----------
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()

    categorical_cols = [
        ("Traffic_Condition", "Traffic Conditions"),
        ("Weather_Condition", "Weather Conditions"),
        ("Claim_Type", "Claim Type"),
        ("Vehicle_Type", "Vehicle Type")
    ]

    for i, (col, title) in enumerate(categorical_cols):
        counts = claims_data[col].value_counts()

        palette = sns.color_palette("husl", len(counts))
        bars = sns.barplot(
            x=counts.index,
            y=counts.values,
            palette=palette,
            ax=axes[i]
        )

        axes[i].set_title(title, fontsize=14)
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Number of Claims")
        axes[i].tick_params(axis="x", rotation=45)

        # ---------- ADD COUNT LABELS ----------
        for bar, count in zip(bars.patches, counts.values):
            axes[i].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold"
            )

    plt.tight_layout(pad=2)
    st.pyplot(fig)

    st.markdown("---")

    # ---------- INSIGHTS ----------
    st.subheader("📌 Key Distribution Insights")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    most_common_weather = claims_data["Weather_Condition"].value_counts().idxmax()
    num_weather = claims_data["Weather_Condition"].value_counts().max()

    most_common_traffic = claims_data["Traffic_Condition"].value_counts().idxmax()
    num_traffic = claims_data["Traffic_Condition"].value_counts().max()

    most_common_claim_type = claims_data["Claim_Type"].value_counts().idxmax()
    num_claim_type = claims_data["Claim_Type"].value_counts().max()

    most_common_vehicle = claims_data["Vehicle_Type"].value_counts().idxmax()
    num_vehicle = claims_data["Vehicle_Type"].value_counts().max()

    metric_col1.metric(
        "Most Common Weather",
        most_common_weather,
        f"{num_weather:,} claims"
    )

    metric_col2.metric(
        "Most Common Traffic",
        most_common_traffic,
        f"{num_traffic:,} claims"
    )

    metric_col3.metric(
        "Most Common Claim Type",
        most_common_claim_type,
        f"{num_claim_type:,} claims"
    )

    metric_col4.metric(
        "Most Common Vehicle",
        most_common_vehicle,
        f"{num_vehicle:,} claims"
    )


def plot_monthly_claims_settlements(claims_data):
    """
    Display monthly claims and settlement trends 
    """

    # Ensure datetime
    claims_data['Accident_Date'] = pd.to_datetime(claims_data['Accident_Date'])
    claims_data['Settlement_Date'] = pd.to_datetime(claims_data['Settlement_Date'])

    # Create Year-Month
    claims_data['Accident_MonthYear'] = claims_data['Accident_Date'].dt.to_period('M').astype(str)
    claims_data['Settlement_MonthYear'] = claims_data['Settlement_Date'].dt.to_period('M').astype(str)

    monthly_claims = claims_data.groupby('Accident_MonthYear').size().reset_index(name='Number of Claims')
    monthly_settlements = claims_data.groupby('Settlement_MonthYear').size().reset_index(name='Number of Settlements')


    st.subheader("📈 Monthly Claims Frequency")
    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(x='Accident_MonthYear', y='Number of Claims', data=monthly_claims, marker='o', color='dodgerblue', ax=ax)
    ax.set_xticklabels(monthly_claims['Accident_MonthYear'], rotation=90)
    st.pyplot(fig)

    
    st.subheader("📈 Monthly Settlements Frequency")
    fig, ax = plt.subplots(figsize=(12,5))
    sns.lineplot(x='Settlement_MonthYear', y='Number of Settlements', data=monthly_settlements, marker='o', color='forestgreen', ax=ax)
    ax.set_xticklabels(monthly_settlements['Settlement_MonthYear'], rotation=90)
    st.pyplot(fig)
    
    

def plot_claim_amount_distributions(claims_data):
    """
    Plot distributions of estimated and ultimate claim amounts
    """
    
    st.subheader("📊 Claims Amount Distribution")
    fig, axes = plt.subplots(1, 2, figsize=(20, 6))

    sns.histplot(claims_data['Estimated_Claim_Amount'], bins=30, kde=True, color='purple', ax=axes[0])
    axes[0].set_title("Estimated Claims")
    axes[0].set_xlabel("Claim Amount")

    sns.histplot(claims_data['Ultimate_Claim_Amount'], bins=30, kde=True, color='green', ax=axes[1])
    axes[1].set_title("Ultimate Claims")
    axes[1].set_xlabel("Claim Amount")

    plt.tight_layout()
    st.pyplot(fig)

    # -------- Other Statistics ------

    st.subheader("📌  Claims Amount Statistics")

    # Aggregate ultimate claim stats
    ultimate_claim_stats = claims_data["Ultimate_Claim_Amount"].agg(Total="sum", Mean="mean", Median="median").round(2)

    # Convert to single-row DataFrame 
    ultimate_claim_stats_df = pd.DataFrame({
        "Total Claim": [f"£{ultimate_claim_stats['Total']:,.2f}"],
        "Average Claims": [f"£{ultimate_claim_stats['Mean']:,.2f}"],
        "Median Claims": [f"£{ultimate_claim_stats['Median']:,.2f}"]
    })

    st.dataframe(ultimate_claim_stats_df)


def plot_age_distributions(claims_data):
  

    st.subheader("📊 Age Distributions")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    sns.histplot(claims_data['Driver_age_(years)'], bins=30, kde=True, color='red', ax=axes[0])
    axes[0].set_title("Driver Age")
    axes[0].set_xlabel("Age (years)")
    axes[0].set_ylabel("Count")

    sns.histplot(claims_data['License_age_(years)'], bins=30, kde=True, color='blue', ax=axes[1])
    axes[1].set_title("License Age")
    axes[1].set_xlabel("Age (years)")
    axes[1].set_ylabel("Count")

    sns.histplot(claims_data['Vehicle_age_(years)'], bins=30, kde=True, color='green', ax=axes[2])
    axes[2].set_title("Vehicle Age")
    axes[2].set_xlabel("Age (years)")
    axes[2].set_ylabel("Count")

    plt.tight_layout()
    st.pyplot(fig)

   
    st.subheader("📌 Age Insights")
    oldest_driver = claims_data['Driver_age_(years)'].max()
    oldest_license = claims_data['License_age_(years)'].max()
    oldest_vehicle = claims_data['Vehicle_age_(years)'].max()

    # Convert to single-row DataFrame 
    age_stats_df = pd.DataFrame({
        "Oldest Driver": [f"{oldest_driver} years"],
        "Oldest License":[f"{oldest_license} years"],
        "Oldest Vehicle": [f"{oldest_vehicle} years"]
    })

    st.dataframe(age_stats_df)

