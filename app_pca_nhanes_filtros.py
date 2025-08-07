import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import mca
import requests
from io import StringIO

# Título
st.title("🔬 NHANES PCA + MCA Dashboard")

# URL del archivo
CSV_URL = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/main/NHANES_Select_Chronic_Conditions_Prevalence_Estimates.csv"

# Descargar datos
@st.cache_data
def load_data(url):
    response = requests.get(url)
    df = pd.read_csv(StringIO(response.text))
    return df

df = load_data(CSV_URL)

# Validar columnas necesarias
if "Prevalence" not in df.columns and "Percent" not in df.columns:
    st.error("Faltan columnas críticas: se requiere 'Prevalence' o 'Percent'")
    st.stop()

# Asegurar una columna target para clasificación
df["Target"] = df["Prevalence"] if "Prevalence" in df.columns else df["Percent"]

# Sidebar filters
st.sidebar.header("🔍 Filtros")
cols_to_filter = ["Year", "Condition", "Measure", "Gender", "Race and Hispanic origin", "Age group"]
for col in cols_to_filter:
    if col in df.columns:
        options = sorted(df[col].dropna().unique())
        selected = st.sidebar.multiselect(f"{col}:", options)
        if selected:
            df = df[df[col].isin(selected)]

# Validar que haya suficientes clases
if df["Condition"].nunique() <= 1:
    st.warning("⚠️ Los filtros seleccionados resultan en una sola clase. Agrega más condiciones para análisis multiclase.")
    st.stop()

# Separar variables numéricas y categóricas
categorical_cols = df.select_dtypes(include="object").columns.tolist()
numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
if "Target" in numerical_cols:
    numerical_cols.remove("Target")  # No incluir la variable objetivo como entrada

# ===== PCA =====
st.subheader("📊 Análisis PCA (Componentes Principales)")

if numerical_cols:
    X_num = df[numerical_cols]
    y = df["Condition"]

    # Normalizar y balancear clases
    pca_pipeline = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote", SMOTE()),
        ("pca", PCA(n_components=2))
    ])

    try:
        X_pca, y_balanced = pca_pipeline.fit_resample(X_num, y)

        fig1, ax1 = plt.subplots(figsize=(10, 6))
        scatter = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=pd.factorize(y_balanced)[0], cmap="tab10", alpha=0.7)
        ax1.set_title("PCA - Componentes principales")
        ax1.set_xlabel("Componente 1")
        ax1.set_ylabel("Componente 2")
        ax1.legend(*scatter.legend_elements(), title="Clase")
        st.pyplot(fig1)

    except ValueError as e:
        st.error(f"Error durante el análisis PCA: {e}")
else:
    st.info("No hay suficientes variables numéricas para aplicar PCA.")

# ===== MCA =====
st.subheader("📈 Análisis de Correspondencias Múltiples (MCA)")

cat_for_mca = [col for col in categorical_cols if df[col].nunique() > 1]

if len(cat_for_mca) >= 2:
    df_cat = pd.get_dummies(df[cat_for_mca], drop_first=True)
    mca_model = mca.MCA(df_cat)
    coords = mca_model.fs_r(2)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.scatter(coords[:, 0], coords[:, 1], alpha=0.6)
    ax2.set_title("MCA - Análisis de Categorías")
    ax2.set_xlabel("Dimensión 1")
    ax2.set_ylabel("Dimensión 2")
    st.pyplot(fig2)
else:
    st.info("No hay suficientes columnas categóricas con más de una categoría para realizar MCA.")

# Mostrar tabla final filtrada
st.subheader("📄 Vista previa de los datos filtrados")
st.dataframe(df.head(50))



