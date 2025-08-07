
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import matplotlib.pyplot as plt
import seaborn as sns
import mca

st.set_page_config(page_title="PCA & ACM con Selección de Variables", layout="wide")
st.title("Análisis PCA y ACM con Selección de Variables - NHANES")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/refs/heads/main/NHANES_Select_Chronic_Conditions_Prevalence_Estimates.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

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
    sex_filter = st.multiselect("Sexo", sorted(df["Sex"].dropna().unique()))
    race_filter = st.multiselect("Raza/Origen", sorted(df["Race"].dropna().unique()))
    condition_filter = st.multiselect("Condición", sorted(df["Condition"].dropna().unique()))
    st.markdown("---")
    k_vars = st.slider("Número de variables a seleccionar", 2, 10, 5)

# Aplicar filtros
for col, values in {
    "Sex": sex_filter, "Race": race_filter, "Condition": condition_filter
}.items():
    if values:
        df = df[df[col].isin(values)]

if df.empty:
    st.warning("No hay datos tras aplicar los filtros.")
    st.stop()

# Variables predictoras y objetivo
num_cols = ["Prevalence", "Standard Error", "Lower 95% CI Limit", "Upper 95% CI Limit"]
cat_cols = ["Sex", "Race", "AgeGroup"]
X_num = df[num_cols]
X_cat = df[cat_cols]
y = df["Condition"]

# Pipeline numérico
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# Preprocesamiento y codificación
X_num_proc = numeric_pipeline.fit_transform(X_num)
X_cat_proc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(X_cat)
X_combined = np.hstack((X_num_proc, X_cat_proc))

# Selección de variables: filtro + embebido + envoltura
st.subheader("Selección de Variables y Evaluación")

# Filtro: SelectKBest
selector_filter = SelectKBest(score_func=f_classif, k=k_vars)
X_kbest = selector_filter.fit_transform(X_combined, y)

# Embebido: RandomForest
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_combined, y)
importances = forest.feature_importances_

# Envoltura: RFE
rfe_model = RFE(LogisticRegression(max_iter=1000), n_features_to_select=k_vars)
rfe_model.fit(X_combined, y)
X_rfe = rfe_model.transform(X_combined)
X_rfe = np.atleast_2d(X_rfe)

# Validación cruzada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=1000)

score_filter = cross_val_score(model, X_kbest, y, cv=cv).mean()
score_embed = cross_val_score(model, X_combined[:, np.argsort(importances)[-k_vars:]], y, cv=cv).mean()
score_rfe = cross_val_score(model, X_rfe, y, cv=cv).mean()

st.markdown(f"- **SelectKBest (filtro):** Accuracy promedio: {score_filter:.2f}")
st.markdown(f"- **RandomForest (embebido):** Accuracy promedio: {score_embed:.2f}")
st.markdown(f"- **RFE (envoltura):** Accuracy promedio: {score_rfe:.2f}")

# PCA
st.subheader("PCA con Variables Seleccionadas (RFE + SMOTE)")
pca_pipeline = ImbPipeline([
    ("smote", SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=2))
])
try:
    X_pca, y_pca = pca_pipeline.fit_resample(X_rfe, y)
except ValueError as e:
    st.error(f"Error al aplicar SMOTE: {e}")
    st.stop()

pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
pca_df["Condition"] = y_pca

fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Condition", palette="tab10", ax=ax1)
ax1.set_title("PCA con selección (RFE) + SMOTE")
st.pyplot(fig1)


# Extraer el modelo PCA del pipeline
pca_model = pca_pipeline.named_steps["pca"]

# Verifica si X tiene nombres de columnas (por si es numpy array)
feature_names = X.columns if hasattr(X, 'columns') else [f"Var{i}" for i in range(X.shape[1])]

# Calcular loadings
loadings = pd.DataFrame(
    pca_model.components_.T,
    columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],
    index=feature_names
)

# Mostrar en Streamlit
st.subheader("Loadings del PCA")
st.dataframe(loadings.round(3))

# Visualización (opcional)
fig, ax = plt.subplots()
for i in range(loadings.shape[0]):
    ax.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1],
             head_width=0.02, color='r')
    ax.text(loadings.iloc[i, 0]*1.15, loadings.iloc[i, 1]*1.15,
            loadings.index[i], ha='center', va='center')
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Loadings: Contribución de las Variables")
ax.grid()
st.pyplot(fig)


# ACM (MCA)
st.subheader("ACM sobre variables categóricas")
if X_cat.shape[0] >= 2:
    mca_model = mca.MCA(pd.get_dummies(X_cat))
    coords = mca_model.fs_r(2)
    mca_df = pd.DataFrame(coords, columns=["Dim1", "Dim2"])
    mca_df["Condition"] = y.values[:len(mca_df)]

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=mca_df, x="Dim1", y="Dim2", hue="Condition", palette="Set2", ax=ax2)
    ax2.set_title("Análisis de Correspondencias Múltiples (ACM)")
    st.pyplot(fig2)
else:
    st.warning("No hay suficientes datos para realizar el ACM.")

st.markdown("---")
st.markdown("**Fuente de datos:** NHANES - CDC. **Autor:** Jonathan Florez")
