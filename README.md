# 🚗 FNOL Claims Analytics & Prediction Dashboard

A full end-to-end **First Notice of Loss (FNOL)** analytics and machine learning application built with **Streamlit**, designed to analyze insurance claims data, visualize trends, and predict ultimate claim amounts.

🔗 **Live App:**  
👉 https://fnolclaimsprojectv1.streamlit.app/

---

## 📌 Project Overview

This project provides an interactive dashboard for insurance claims analysis, including:

- Customer & claims overview KPIs
- Exploratory data analysis (EDA)
- Categorical and time-series visualizations
- Machine learning–based FNOL claim prediction
- Model retraining interface

The goal is to support **data-driven decision-making** in insurance claims management.

---

## 🧩 Application Features

### 🏠 Claim Overview
- Key KPIs (min/max claims, driver age metrics)
- Total estimated vs ultimate claim values
- Claim variance analysis
- Claim type, traffic, and weather impact analysis

### 📊 Visualizations
- Categorical distributions (traffic, weather, claim type, vehicle type)
- Monthly claims and settlements trends
- Claim amount distributions
- Driver, license, and vehicle age analysis

### 🧮 FNOL Prediction
- Predicts **ultimate claim amount** using incident and driver details
- Machine learning model with one-hot encoded categorical features
- Variance comparison between estimated and predicted claim amounts

### 🔄 Model Retraining
- Allows retraining the model on updated data
- Includes preprocessing steps such as winsorization
- Saves updated models for reuse

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** – UI & dashboard
- **Pandas / NumPy** – Data processing
- **Seaborn / Matplotlib** – Visualizations
- **Scikit-learn** – Machine learning
- **Joblib / Pickle** – Model persistence
- **Git & GitHub** – Version control
- **Streamlit Cloud** – Deployment

---

## 📂 Project Structure

