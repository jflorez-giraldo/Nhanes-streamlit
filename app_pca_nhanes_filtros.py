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

# Cargar datos
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/main/NHANES_Select_Chronic_Conditions_Prevalence_Estimates.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={
        "Age Group": "AgeGroup",
        "Race and Hispanic Origin": "RaceEthnicity",
        "Lower 95% CI Limit": "CI_Lower",
        "Upper 95% CI Limit": "CI_Upper",
        "Percent": "Prevalence"
    })
    return df

df = load_data()

# --- Agrupar condiciones similares ---
def group_conditions(text):
    if "diabetes" in text.lower():
        return "Diabetes"
    elif "cancer" in text.lower():
        return "Cancer"
    elif "hypertension" in text.lower() or "blood pressure" in text.lower():
        return "Hypertension"
    elif "asthma" in text.lower():
        return "Asthma"
    elif "stroke" in text.lower():
        return "Stroke"
    elif "heart" in text.lower():
        return "Heart Disease"
    elif "arthritis" in text.lower():
        return "Arthritis"
    elif "obesity" in text.lower():
        return "Obesity"
    elif "depression" in text.lower():
        return "Depression"
    else:
        return text.strip()

df["ConditionGroup"] = df["Measure"].astype(str).apply(group_conditions)

# --- Sidebar ---
st.sidebar.header("Filters")
use_grouped = st.sidebar.checkbox("Use grouped conditions", value=True)
target_col = "ConditionGroup" if use_grouped else "Measure"

age_opts = sorted(df["AgeGroup"].dropna().unique())
age_filter = st.sidebar.selectbox("Age Group", age_opts)
df = df[df["AgeGroup"] == age_filter]

sex_opts = sorted(df["Sex"].dropna().unique())
sex_filter = st.sidebar.selectbox("Sex", sex_opts)
df = df[df["Sex"] == sex_filter]

condition_opts = sorted(df[target_col].dropna().unique())
selected_conditions = st.sidebar.multiselect("Condition(s)", condition_opts, default=condition_opts[:3])
df = df[df[target_col].isin(selected_conditions)]

st.subheader("Filtered Data")
st.write(f"Dataset shape: {df.shape}")
st.dataframe(df)

if df.shape[0] < 20:
    st.warning("Warning: Very few rows left after filtering. Try selecting more conditions.")

# --- Preparar datos ---
categorical_cols = ["AgeGroup", "Sex", "RaceEthnicity"]
numerical_cols = ["Prevalence", "Standard Error", "CI_Lower", "CI_Upper"]
df = df.dropna(subset=numerical_cols + categorical_cols + [target_col])

# Convert to numeric
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Encode target
le = LabelEncoder()
df["target"] = le.fit_transform(df[target_col])
X_num = df[numerical_cols]
y = df["target"]

# --- PCA ---
st.subheader("PCA (Numerical Features)")
if len(np.unique(y)) > 1:
    pca_pipeline = ImbPipeline([
        ("imputer", IterativeImputer(random_state=42)),
        ("scaler", StandardScaler()),
        ("adasyn", ADASYN(random_state=42)),
    ])
    X_bal, y_bal = pca_pipeline.fit_resample(X_num, y)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_bal)

    fig1, ax1 = plt.subplots()
    scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_bal, cmap="tab10", alpha=0.6)
    ax1.set_xlabel("PCA 1")
    ax1.set_ylabel("PCA 2")
    ax1.set_title("PCA with ADASYN")
    ax1.legend(*scatter.legend_elements(), title="Condition")
    st.pyplot(fig1)
else:
    st.warning("Only one class found. PCA requires at least two classes.")

# --- MCA ---
st.subheader("MCA (Categorical Features)")
df_cat = df[categorical_cols].astype(str)
if df_cat.shape[0] > 2:
    mca_model = mca.MCA(df_cat)
    mca_coords = mca_model.fs_r(N=2)

    fig2, ax2 = plt.subplots()
    ax2.scatter(mca_coords[:, 0], mca_coords[:, 1], alpha=0.5)
    ax2.set_xlabel("MCA 1")
    ax2.set_ylabel("MCA 2")
    ax2.set_title("MCA of Categorical Features")
    st.pyplot(fig2)
else:
    st.warning("Too few rows for MCA.")

# --- Feature importance ---
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

# --- Correlation ---
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

# --- RFE ---
st.subheader("Recursive Feature Elimination (RFE)")
rfe_pipe = Pipeline([
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler()),
    ("rfe", RFE(LogisticRegression(max_iter=1000), n_features_to_select=3))
])
rfe_pipe.fit(X_num, y)
rfe_df = pd.DataFrame({
    "Feature": numerical_cols,
    "Selected": rfe_pipe.named_steps["rfe"].support_,
    "Ranking": rfe_pipe.named_steps["rfe"].ranking_
})
st.dataframe(rfe_df.sort_values("Ranking"))



