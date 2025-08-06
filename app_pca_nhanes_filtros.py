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
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # Necesario para activar IterativeImputer
from sklearn.impute import IterativeImputer
from imblearn.over_sampling import ADASYN

# Streamlit config
st.set_page_config(page_title="PCA and MCA on NHANES", layout="wide")
st.title("PCA, MCA and Feature Selection on NHANES Data")

# Load dataset from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/main/NHANES_Select_Chronic_Conditions_Prevalence_Estimates.csv"
    df = pd.read_csv(url)
    return df

df = load_data()
st.write("Original dataset shape:", df.shape)

# Sidebar filters
st.sidebar.header("Filter data")

if "AgeGroup" in df.columns:
    age_group = st.sidebar.selectbox("Select Age Group", sorted(df["AgeGroup"].dropna().unique()))
    df = df[df["AgeGroup"] == age_group]

if "Sex" in df.columns:
    sex = st.sidebar.selectbox("Select Sex", sorted(df["Sex"].dropna().unique()))
    df = df[df["Sex"] == sex]

if "Condition" in df.columns:
    selected_conditions = st.sidebar.multiselect("Select Condition(s)", sorted(df["Condition"].dropna().unique()))
    if selected_conditions:
        df = df[df["Condition"].isin(selected_conditions)]

st.subheader("Filtered Dataset")
st.dataframe(df.head())

# Preprocess
categorical_cols = ["AgeGroup", "Sex", "Race and Hispanic origin", "Education", "Marital Status"]
numerical_cols = ["Sample Size", "Prevalence", "Standard Error", "95% CI Lower", "95% CI Upper"]

df = df.dropna(subset=numerical_cols + categorical_cols + ["Condition"])

# Encode target variable
target = "Condition"
le = LabelEncoder()
df[target] = le.fit_transform(df[target])

X_num = df[numerical_cols]
y = df[target]

# --- PCA Pipeline ---
st.subheader("PCA (Numerical Features)")
# Pipeline completo con IterativeImputer
pca_pipeline = ImbPipeline([
    ("imputer", IterativeImputer(random_state=42)),
    ("scaler", StandardScaler()),
    ("adasyn", ADASYN(random_state=42)),
    ("pca", PCA(n_components=2))
])

X_pca, y_balanced = pca_pipeline.fit_resample(X_num, y)

fig1, ax1 = plt.subplots()
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_balanced, cmap="viridis", alpha=0.6)
ax1.set_xlabel("PCA 1")
ax1.set_ylabel("PCA 2")
ax1.set_title("PCA with Preprocessing + ADASYN")
legend1 = ax1.legend(*scatter.legend_elements(), title="Condition")
ax1.add_artist(legend1)
st.pyplot(fig1)

# --- MCA Pipeline ---
st.subheader("MCA (Categorical Features)")

df_cat = df[categorical_cols].astype("category")

# Renombrar la instancia para no sobrescribir el módulo
mca_model = mca.MCA(df_cat)

# Extraer coordenadas
mca_coords = mca_model.fs_r(N=2)

# Gráfica
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

# --- Correlation Filter ---
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

# --- Wrapper: RFE ---
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

