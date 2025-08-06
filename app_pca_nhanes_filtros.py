import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# --- Title ---
st.title("PCA Analysis on NHANES Dataset")
st.markdown("""
This app performs a **Principal Component Analysis (PCA)** on a sample of the NHANES dataset.  
Use the filters in the sidebar to select the subset of data you want to analyze.
""")

# --- Load sample NHANES dataset ---
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    columns = [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    ]
    return pd.read_csv(url, names=columns)

df = load_data()

# --- Sidebar filters ---
st.sidebar.header("Filters")
age_range = st.sidebar.slider("Select age range", int(df["Age"].min()), int(df["Age"].max()), (21, 50))
outcome_filter = st.sidebar.selectbox("Select outcome", options=["All", "Diabetic", "Non-Diabetic"])

filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]
if outcome_filter == "Diabetic":
    filtered_df = filtered_df[filtered_df["Outcome"] == 1]
elif outcome_filter == "Non-Diabetic":
    filtered_df = filtered_df[filtered_df["Outcome"] == 0]

# --- Select numeric features for PCA ---
features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]
X = filtered_df[features]

# --- Standardize the features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- PCA ---
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# --- Explained variance ---
explained_var = pca.explained_variance_ratio_

# --- Plots ---
st.subheader("Explained Variance by Components")
fig1, ax1 = plt.subplots()
sns.barplot(x=np.arange(1, len(explained_var)+1), y=explained_var, ax=ax1)
ax1.set_xlabel("Principal Component")
ax1.set_ylabel("Explained Variance Ratio")
st.pyplot(fig1)

# --- 2D Scatter plot with first 2 components ---
st.subheader("2D PCA Scatter Plot")
fig2, ax2 = plt.subplots()
scatter = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=filtered_df["Outcome"], cmap="viridis", alpha=0.6)
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
legend1 = ax2.legend(*scatter.legend_elements(), title="Outcome")
ax2.add_artist(legend1)
st.pyplot(fig2)

# --- PCA loadings heatmap ---
st.subheader("PCA Component Loadings")
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(features))], index=features)
fig3, ax3 = plt.subplots(figsize=(10, 6))
sns.heatmap(loadings.iloc[:, :5], annot=True, cmap="coolwarm", center=0, ax=ax3)
st.pyplot(fig3)
# Placeholder for your Streamlit app code
# Replace this with your actual Streamlit script
