import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mca

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestClassifier

# Streamlit config
st.set_page_config(page_title="PCA and MCA on NHANES", layout="wide")
st.title("PCA, MCA and Feature Selection on NHANES Data")

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/main/NHANES_Select_Chronic_Conditions_Prevalence_Estimates.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()  # Limpiar espacios en nombres de columnas
    return df

df = load_data()

# Renombrar columnas para mantener consistencia
df = df.rename(columns={
    "Age Group": "AgeGroup",
    "Race and Hispanic Origin": "Race and Hispanic origin",
    "Lower 95% CI Limit": "95% CI Lower",
    "Upper 95% CI Limit": "95% CI Upper",
    "Percent": "Prevalence"
})

st.write("Original dataset shape:", df.shape)

# --- Sidebar filters ---
st.sidebar.header("Filter data")

if "AgeGroup" in df.columns:
    age_group = st.sidebar.selectbox("Select Age Group", sorted(df["AgeGroup"].dropna().unique()))
    df = df[df["AgeGroup"] == age_group]

if "Sex" in df.columns:
    sex = st.sidebar.selectbox("Select Sex", sorted(df["Sex"].dropna().unique()))
    df = df[df["Sex"] == sex]

if "Measure" in df.columns:
    selected_conditions = st.sidebar.multiselect("Select Condition(s)", sorted(df["Measure"].dropna().unique()))
    if selected_conditions:
        df = df[df["Measure"].isin(selected_conditions)]

st.subheader("Filtered Dataset")
st.dataframe(df.head())

# --- Selección de columnas ---
categorical_cols = ["AgeGroup", "Sex", "Race and Hispanic origin"]
numerical_cols = ["Prevalence", "Standard Error", "95% CI Lower", "95% CI Upper"]

# Eliminar filas con valores faltantes
subset_cols = numerical_cols + categorical_cols + ["Measure"]
subset_cols = [col for col in subset_cols if col in df.columns]
df = df.dropna(subset=subset_cols)

# Convertir numéricas
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Encode target variable
target = "Measure"
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

X_num = df[numerical_cols]
y = df[target]

# --- PCA ---
st.subheader("PCA (Numerical Features)")

balance_pipeline = ImbPipeline([
    ("imputer", IterativeImputer(random_state=42)),
    ("scaler", StandardScaler()),
    ("adasyn", ADASYN(random_state=42))
])

X_balanced, y_balanced = balance_pipeline.fit_resample(X_num, y)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_balanced)

fig1, ax1 = plt.subplots()
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_balanced, cmap="viridis", alpha=0.6)
ax1.set_xlabel("PCA 1")
ax1.set_ylabel("PCA 2")
ax1.set_title("PCA with Preprocessing + ADASYN")
legend1 = ax1.legend(*scatter.legend_elements(), title="Condition")
ax1.add_artist(legend1)
st.pyplot(fig1)

# --- MCA ---
st.subheader("MCA (Categorical Features)")

df_cat = df[categorical_cols].astype("category")
mca_model = mca.MCA(df_cat)
mca_coords = mca_model.fs_r(N=2)

fig2, ax2 = plt.subplots()
ax2.scatter(mca_coords[:, 0], mca_coords[:, 1], alpha=0.5)
ax2.set_xlabel("MCA 1")
ax2.set_ylabel("MCA 2")
ax2.set_title("MCA on Categorical Features")
st.pyplot(fig2)

# --- Feature Importance ---
st.subheader("Feature Importance (Random Forest)")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_num, y)
importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": numerical_cols,
    "Importance": importances
}).sort_values("Importance", ascending=False)

fig3, ax3 = plt.subplots()
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax3)
ax3.set_title("Feature Importance")
st.pyplot(fig3)

# --- Correlation with Target ---
st.subheader("Correlation with Target")
correlations = [np.corrcoef(X_num[col], y)[0, 1] for col in numerical_cols]
corr_df = pd.DataFrame({
    "Feature": numerical_cols,
    "Abs Correlation": np.abs(correlations)
}).sort_values("Abs Correlation", ascending=False)

fig4, ax4 = plt.subplots()
sns.barplot(data=corr_df, x="Abs Correlation", y="Feature", ax=ax4)
ax4.set_title("Absolute Correlation with Condition")
st.pyplot(fig4)

# --- Recursive Feature Elimination ---
st.subheader("Recursive Feature Elimination (RFE)")
rfe_pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("rfe", RFE(LogisticRegression(max_iter=1000), n_features_to_select=3))
])
rfe_pipe.fit(X_num, y)
rfe_selected = rfe_pipe.named_steps["rfe"].support_
rfe_ranking = rfe_pipe.named_steps["rfe"].ranking_

rfe_df = pd.DataFrame({
    "Feature": numerical_cols,
    "Selected": rfe_selected,
    "Ranking": rfe_ranking
})
st.dataframe(rfe_df.sort_values("Ranking"))


