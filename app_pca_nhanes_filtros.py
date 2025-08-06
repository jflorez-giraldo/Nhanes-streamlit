import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import prince

from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

from sklearn.compose import ColumnTransformer
import prince

# Streamlit page config
st.set_page_config(page_title="PCA and MCA on NHANES", layout="wide")
st.title("PCA, MCA and Feature Selection on NHANES Data")

# Load dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datablist/sample-csv-files/main/files/people/people-100.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# Simulate NHANES-like data for demonstration
df = pd.read_csv("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", header=None)
df.columns = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
              "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

st.sidebar.header("Filter data")
age_min, age_max = st.sidebar.slider("Age range", int(df["Age"].min()), int(df["Age"].max()), 
                                     (int(df["Age"].min()), int(df["Age"].max())))
glucose_min, glucose_max = st.sidebar.slider("Glucose range", int(df["Glucose"].min()), int(df["Glucose"].max()), 
                                             (int(df["Glucose"].min()), int(df["Glucose"].max())))

filtered_df = df[(df["Age"] >= age_min) & (df["Age"] <= age_max) & 
                 (df["Glucose"] >= glucose_min) & (df["Glucose"] <= glucose_max)]

st.subheader("Filtered Dataset")
st.dataframe(filtered_df.head())

# Features and Target
features = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", 
            "BMI", "DiabetesPedigreeFunction", "Age"]
target = "Outcome"

X = filtered_df[features]
y = filtered_df[target]

# --- Pipeline with Scaling, Imputation, SMOTE, PCA ---
pca_pipeline = ImbPipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(n_components=2))
])

X_pca, y_balanced = pca_pipeline.fit_resample(X, y)

# Plot PCA
st.subheader("PCA (with SMOTE and preprocessing)")
fig1, ax1 = plt.subplots()
scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y_balanced, cmap='viridis', alpha=0.6)
legend = ax1.legend(*scatter.legend_elements(), title="Classes")
ax1.add_artist(legend)
ax1.set_xlabel("PCA 1")
ax1.set_ylabel("PCA 2")
ax1.set_title("PCA with Preprocessing + SMOTE")
st.pyplot(fig1)

# --- MCA Pipeline ---
st.subheader("Multiple Correspondence Analysis (MCA)")
mca_cols = ["Pregnancies", "Outcome"]
df_mca = filtered_df[mca_cols].astype("category")

mca = prince.MCA(n_components=2, random_state=42)
mca_result = mca.fit(df_mca).row_coordinates(df_mca)

fig2, ax2 = plt.subplots()
ax2.scatter(mca_result[0], mca_result[1], alpha=0.5)
ax2.set_xlabel("MCA 1")
ax2.set_ylabel("MCA 2")
ax2.set_title("MCA on Categorical Features")
st.pyplot(fig2)

# --- Feature Importance ---
st.subheader("Feature Importance (Random Forest)")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X, y)

importances = rf.feature_importances_
importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
importance_df = importance_df.sort_values("Importance", ascending=False)

fig3, ax3 = plt.subplots()
sns.barplot(data=importance_df, x="Importance", y="Feature", ax=ax3)
ax3.set_title("Feature Importance from Random Forest")
st.pyplot(fig3)

# --- Correlation Filtering ---
st.subheader("Correlation-based Filtering")
correlations = [np.corrcoef(filtered_df[feature], filtered_df[target])[0, 1] for feature in features]
corr_df = pd.DataFrame({"Feature": features, "Abs Correlation": np.abs(correlations)})
corr_df = corr_df.sort_values("Abs Correlation", ascending=False)

fig4, ax4 = plt.subplots()
sns.barplot(data=corr_df, x="Abs Correlation", y="Feature", ax=ax4)
ax4.set_title("Absolute Correlation with Target")
st.pyplot(fig4)

# --- Wrapper Method (RFE) ---
st.subheader("Wrapper Method: Recursive Feature Elimination (RFE)")
rfe_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("rfe", RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=5))
])

rfe_pipeline.fit(X, y)
rfe_mask = rfe_pipeline.named_steps["rfe"].support_
rfe_ranking = rfe_pipeline.named_steps["rfe"].ranking_

rfe_df = pd.DataFrame({"Feature": features, "Selected": rfe_mask, "Ranking": rfe_ranking})
st.dataframe(rfe_df.sort_values("Ranking"))
