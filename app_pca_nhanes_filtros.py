import pandas as pd
import numpy as np
import streamlit as st
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline  # ← esto es lo importante
#from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import ADASYN
import matplotlib.pyplot as plt
import seaborn as sns
import mca

# Configuración de la página
st.set_page_config(page_title="PCA NHANES", layout="wide")

# Título
st.title("Análisis PCA y MCA de Condiciones Crónicas - NHANES")

# Cargar datos desde GitHub
#@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/refs/heads/main/NHANES_Select_Chronic_Conditions_Prevalence_Estimates.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Revisar columnas esperadas
required_columns = [
    "Survey Years", "Sex", "Age Group", "Race and Hispanic Origin", 
    "Measure", "Percent", "Standard Error", "Lower 95% CI Limit", 
    "Upper 95% CI Limit", "Presentation Standard", "Note1", "Notea"
]

missing_cols = [col for col in required_columns if col not in df.columns]
if missing_cols:
    st.error(f"Faltan columnas críticas en el CSV: {missing_cols}")
    st.stop()

# Renombrar columnas para simplificar
df = df.rename(columns={
    "Survey Years": "Year",
    "Race and Hispanic Origin": "Race",
    "Age Group": "AgeGroup",
    "Percent": "Prevalence",
    "Measure": "Condition"
})

# Filtros
with st.sidebar:
    st.header("Filtros")
    #year_filter = st.multiselect("Años", sorted(df["Year"].dropna().unique()), default=None)
    sex_filter = st.multiselect("Sexo", sorted(df["Sex"].dropna().unique()), default=None)
    race_filter = st.multiselect("Raza/Origen", sorted(df["Race"].dropna().unique()), default=None)
    #age_filter = st.multiselect("Grupo de Edad", sorted(df["AgeGroup"].dropna().unique()), default=None)
    condition_filter = st.multiselect("Condición", sorted(df["Condition"].dropna().unique()), default=None)

# Aplicar filtros
filters = {
    #"Year": year_filter,
    "Sex": sex_filter,
    "Race": race_filter,
    #"AgeGroup": age_filter,
    "Condition": condition_filter
}

for col, values in filters.items():
    if values:
        df = df[df[col].isin(values)]

# Mostrar advertencia si el dataframe está vacío
if df.empty:
    st.warning("No hay datos que coincidan con los filtros seleccionados.")
    st.stop()

# Validación de clases en variable objetivo
if df["Condition"].nunique() <= 1:
    st.warning("Los filtros seleccionados resultan en una sola clase en la variable objetivo. Se requieren al menos dos para PCA y resampling.")
    st.stop()

# Preparar datos para PCA
X = df[["Prevalence"]].copy()
y = df["Condition"]

# Pipeline con ADASYN y PCA
pca_pipeline = Pipeline([
    ("imputer", IterativeImputer(random_state=42)),
    ("scaler", StandardScaler()),
    ("adasyn", ADASYN(random_state=42)),
    ("pca", PCA(n_components=2))
])

X_pca, y_pca = pca_pipeline.fit_resample(X, y)

# Visualización PCA
pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Condition"] = y_pca

st.subheader("Visualización PCA")
fig_pca, ax_pca = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Condition", palette="tab10", ax=ax_pca)
ax_pca.set_title("PCA con balanceo ADASYN")
st.pyplot(fig_pca)

# MCA
st.subheader("Análisis de Correspondencias Múltiples (MCA)")

cat_cols = ["Sex", "Race", "AgeGroup"]
df_cat = df[cat_cols].dropna()

# One-hot encoding
df_dummies = pd.get_dummies(df_cat)

# Validación de datos
if df_dummies.shape[0] < 2:
    st.warning("No hay suficientes datos para aplicar MCA.")
    st.stop()

mca_model = mca.MCA(df_dummies)
mca_coords = mca_model.fs_r(2)  # Primeras dos dimensiones

# Visualización MCA
mca_df = pd.DataFrame(mca_coords, columns=["Dim1", "Dim2"])
mca_df["Condition"] = df["Condition"].values[:len(mca_df)]

fig_mca, ax_mca = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=mca_df, x="Dim1", y="Dim2", hue="Condition", palette="Set2", ax=ax_mca)
ax_mca.set_title("MCA - Variables Categóricas")
st.pyplot(fig_mca)

# Información adicional
st.markdown("---")
st.markdown("**Datos:** Encuesta NHANES (EE.UU.) sobre prevalencia de condiciones crónicas. Datos descargados de [CDC](https://wwwn.cdc.gov/NHANE).")
st.markdown("**Autor del app:** Jonathan Florez / Adaptado por ChatGPT")




