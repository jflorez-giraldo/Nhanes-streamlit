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
st.title("PCA and MCA Analysis with Feature Selection - NHANES")

@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/jflorez-giraldo/Nhanes-streamlit/main/nhanes_2015_2016.csv"
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    return df

df = load_data()

# Definir función para asignar condición
def assign_condition(row):
    # Hipertensión: presión sistólica ≥ 140 o diastólica ≥ 90
    if (row["BPXSY1"] >= 140 or row["BPXSY2"] >= 140 or 
        row["BPXDI1"] >= 90 or row["BPXDI2"] >= 90):
        return "hypertension"
    # Diabetes simulada: IMC muy alto como proxy (≥30)
    elif row["BMXBMI"] >= 30:
        return "diabetes"
    # Colesterol alto (proxy con cintura abdominal, hombres >102 cm, mujeres >88 cm)
    elif ((row["RIAGENDR"] == 1 and row["BMXWAIST"] > 102) or 
          (row["RIAGENDR"] == 2 and row["BMXWAIST"] > 88)):
        return "high cholesterol"
    else:
        return "healthy"

# Crear la nueva columna
df["Condition"] = df.apply(assign_condition, axis=1)

# Verificar distribución
print(df["Condition"].value_counts())

# Diccionario de códigos por variable categórica
category_mappings = {
    "RIAGENDR": {
        1: "Male",
        2: "Female"
    },
    "DMDMARTL": {
        1: "Married",
        2: "Divorced",
        3: "Never married",
        4: "Widowed",
        5: "Separated",
        6: "Living with partner",
        77: "Refused",
        99: "Don't know"
    },
    "DMDEDUC2": {
        1: "Less than 9th grade",
        2: "9-11th grade (no diploma)",
        3: "High school/GED",
        4: "Some college or AA degree",
        5: "College graduate or above",
        7: "Refused",
        9: "Don't know"
    },
    "SMQ020": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "ALQ101": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    },
    "ALQ110": {
        1: "Every day",
        2: "5–6 days/week",
        3: "3–4 days/week",
        4: "1–2 days/week",
        5: "2–3 days/month",
        6: "Once a month or less",
        7: "Refused",
        9: "Don't know"
    },
    "RIDRETH1": {
        1: "Mexican American",
        2: "Other Hispanic",
        3: "Non-Hispanic White",
        4: "Non-Hispanic Black",
        5: "Other Race - Including Multi-Racial"
    },
    "DMDCITZN": {
        1: "Citizen by birth or naturalization",
        2: "Not a citizen of the U.S.",
        7: "Refused",
        9: "Don't know"
    },
    "HIQ210": {
        1: "Yes",
        2: "No",
        7: "Refused",
        9: "Don't know"
    }
}

def apply_categorical_mappings(df, mappings):
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)
    return df

df = apply_categorical_mappings(df, category_mappings)

col_map = {
    "SEQN": "Participant ID",
    "ALQ101": "Alcohol Intake - Past 12 months (Q1)",
    "ALQ110": "Alcohol Frequency",
    "ALQ130": "Alcohol Amount",
    "SMQ020": "Smoking Status",
    "RIAGENDR": "Gender",
    "RIDAGEYR": "Age (years)",
    "RIDRETH1": "Race/Ethnicity",
    "DMDCITZN": "Citizenship",
    "DMDEDUC2": "Education Level",
    "DMDMARTL": "Marital Status",
    "DMDHHSIZ": "Household Size",
    "WTINT2YR": "Interview Weight",
    "SDMVPSU": "Masked PSU",
    "SDMVSTRA": "Masked Stratum",
    "INDFMPIR": "Income to Poverty Ratio",
    "BPXSY1": "Systolic BP1",
    "BPXDI1": "Diastolic BP1",
    "BPXSY2": "Systolic BP2",
    "BPXDI2": "Diastolic BP2",
    "BMXWT": "Body Weight",
    "BMXHT": "Body Height",
    "BMXBMI": "Body Mass Index",
    "BMXLEG": "Leg Length",
    "BMXARML": "Arm Length",
    "BMXARMC": "Arm Circumference",
    "BMXWAIST": "Waist Circumference",
    "HIQ210": "Health Insurance Coverage"
}

df = df.rename(columns=col_map)

# Mostrar tipos de variables
st.subheader("Resumen del tipo de variables")
info_df = pd.DataFrame({
    "Column": df.columns,
    "Non-Null Count": df.notnull().sum().values,
    "Dtype": df.dtypes.values
})
st.dataframe(info_df)

# Detección automática de variables categóricas
categorical_vars = [col for col in df.columns 
                    if df[col].dtype == 'object' or 
                       df[col].nunique() <= 10]

# Detección automática de variables categóricas
categorical_vars = [col for col in df.columns 
                    if df[col].dtype == 'object' or 
                       df[col].nunique() <= 10]

# Crear un DataFrame con las clases únicas por variable
category_info = []
for col in categorical_vars:
    unique_vals = df[col].dropna().unique()
    category_info.append({
        "Variable": col,
        "Unique Classes": ", ".join(map(str, sorted(unique_vals)))
    })

category_df = pd.DataFrame(category_info)


# Filtros
with st.sidebar:
    st.header("Filters")
    Gender_filter = st.multiselect("Gender", sorted(df["Gender"].dropna().unique()))
    race_filter = st.multiselect("Race/Ethnicity", sorted(df["Race/Ethnicity"].dropna().unique()))
    condition_filter = st.multiselect("Condition", sorted(df["Condition"].dropna().unique()))
    #st.markdown("---")
    #k_vars = st.slider("Number of variables to select", 2, 10, 5)

# Aplicar filtros
for col, values in {
    "Gender": sex_filter, "Race": race_filter, "Condition": condition_filter
}.items():
    if values:
        df = df[df[col].isin(values)]

if df.empty:
    st.warning("No data available after applying filters.")
    st.stop()

# Mostrar primeros 10 registros del DataFrame filtrado
st.subheader("Preview of Filtered Data")
st.dataframe(df.head(10))

















## Variables predictoras y objetivo
#num_cols = ["Prevalence", "Standard Error", "Lower 95% CI Limit", "Upper 95% CI Limit"]
#cat_cols = ["Sex", "Race", "AgeGroup"]
#X_num = df[num_cols]
#X_cat = df[cat_cols]
#y = df["Condition"]

## Pipeline numérico
#numeric_pipeline = Pipeline([
#    ("imputer", SimpleImputer(strategy="mean")),
#    ("scaler", StandardScaler())
#])

## Preprocesamiento y codificación
#X_num_proc = numeric_pipeline.fit_transform(X_num)
#X_cat_proc = OneHotEncoder(sparse_output=False, handle_unknown="ignore").fit_transform(X_cat)
#X_combined = np.hstack((X_num_proc, X_cat_proc))

## Selección de variables: filtro + embebido + envoltura
#st.subheader("Selección de Variables y Evaluación")

## Filtro: SelectKBest
#selector_filter = SelectKBest(score_func=f_classif, k=k_vars)
#X_kbest = selector_filter.fit_transform(X_combined, y)

## Embebido: RandomForest
#forest = RandomForestClassifier(n_estimators=100, random_state=42)
#forest.fit(X_combined, y)
#importances = forest.feature_importances_

## Envoltura: RFE
#y = y.reset_index(drop=True)
#X_combined = pd.DataFrame(X_combined).reset_index(drop=True)

#rfe_model = RFE(LogisticRegression(max_iter=1000), n_features_to_select=k_vars)
#rfe_model.fit(X_combined, y)
#X_rfe = rfe_model.transform(X_combined)

## Asegurar que X_rfe es 2D correctamente
#if X_rfe.ndim == 1:
#    X_rfe = X_rfe.reshape(-1, 1)

## Validación cruzada
#cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#model = LogisticRegression(max_iter=1000)

#score_filter = cross_val_score(model, X_kbest, y, cv=cv).mean()
#selected_features = np.argsort(importances)[-k_vars:]
#score_embed = cross_val_score(model, X_combined.iloc[:, selected_features], y, cv=cv).mean()
#score_rfe = cross_val_score(model, X_rfe, y, cv=cv).mean()

## Mostrar métricas como gráfico
#st.subheader("Comparación de Técnicas de Selección de Variables")

#scores_df = pd.DataFrame({
#    "Técnica": ["Filtro (SelectKBest)", "Embebido (RandomForest)", "Envoltura (RFE)"],
#    "Accuracy Promedio": [score_filter, score_embed, score_rfe]
#})

#fig, ax = plt.subplots()
#sns.barplot(data=scores_df, x="Técnica", y="Accuracy Promedio", ax=ax)
#ax.set_ylim(0, 1)
#ax.set_title("Comparación de Precisión entre Técnicas de Selección de Variables")
#st.pyplot(fig)

## También mostramos las métricas en texto
#st.markdown(f"- **SelectKBest (filtro):** Accuracy promedio: {score_filter:.2f}")
#st.markdown(f"- **RandomForest (embebido):** Accuracy promedio: {score_embed:.2f}")
#st.markdown(f"- **RFE (envoltura):** Accuracy promedio: {score_rfe:.2f}")








#st.write("Forma de X_rfe:", X_rfe.shape)

## Convertir X_rfe en DataFrame si es array
#if isinstance(X_rfe, np.ndarray):
#    X_rfe = pd.DataFrame(X_rfe)

## Resetear índice de y para que se alinee con X_rfe
#y = y.reset_index(drop=True)

## Verifica si cada clase tiene suficientes muestras para SMOTE
#min_samples_per_class = y.value_counts().min()
#if min_samples_per_class < 6:
#    st.warning(f"No se puede aplicar SMOTE porque al menos una clase tiene menos de 6 muestras (actual: {min_samples_per_class}).")
#    st.stop()

## PCA
#st.subheader("PCA con Variables Seleccionadas (RFE + SMOTE)")
#pca_pipeline = ImbPipeline([
#    ("smote", SMOTE(random_state=42)),
#    ("scaler", StandardScaler()),
#    ("pca", PCA(n_components=2))
#])

#st.write(f"Tipo de pipeline: {type(pca_pipeline)}")

#try:
#    X_pca, y_pca = pca_pipeline.fit_resample(X_rfe, y)
#except ValueError as e:
#    st.error(f"Error al aplicar SMOTE: {e}")
#    st.stop()

#pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
#pca_df["Condition"] = y_pca

#fig1, ax1 = plt.subplots(figsize=(10, 6))
#sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Condition", palette="tab10", ax=ax1)
#ax1.set_title("PCA con selección (RFE) + SMOTE")
#st.pyplot(fig1)


## Extraer el modelo PCA del pipeline
#pca_model = pca_pipeline.named_steps["pca"]

## Verifica si X tiene nombres de columnas (por si es numpy array)
#feature_names = X.columns if hasattr(X, 'columns') else [f"Var{i}" for i in range(X.shape[1])]

## Calcular loadings
#loadings = pd.DataFrame(
    #pca_model.components_.T,
    #columns=[f"PC{i+1}" for i in range(pca_model.n_components_)],
    #index=feature_names
#)

## Mostrar en Streamlit
#st.subheader("Loadings del PCA")
#st.dataframe(loadings.round(3))

## Visualización (opcional)
#fig, ax = plt.subplots()
#for i in range(loadings.shape[0]):
#    ax.arrow(0, 0, loadings.iloc[i, 0], loadings.iloc[i, 1],
#             head_width=0.02, color='r')
#    ax.text(loadings.iloc[i, 0]*1.15, loadings.iloc[i, 1]*1.15,
#            loadings.index[i], ha='center', va='center')
#ax.set_xlabel("PC1")
#ax.set_ylabel("PC2")
#ax.set_title("Loadings: Contribución de las Variables")
#ax.grid()
#st.pyplot(fig)


## ACM (MCA)
#st.subheader("ACM sobre variables categóricas")
#if X_cat.shape[0] >= 2:
#    mca_model = mca.MCA(pd.get_dummies(X_cat))
#    coords = mca_model.fs_r(2)
#    mca_df = pd.DataFrame(coords, columns=["Dim1", "Dim2"])
#    mca_df["Condition"] = y.values[:len(mca_df)]

#    fig2, ax2 = plt.subplots(figsize=(10, 6))
#    sns.scatterplot(data=mca_df, x="Dim1", y="Dim2", hue="Condition", palette="Set2", ax=ax2)
#    ax2.set_title("Análisis de Correspondencias Múltiples (ACM)")
#    st.pyplot(fig2)
#else:
#    st.warning("No hay suficientes datos para realizar el ACM.")

#st.markdown("---")
#st.markdown("**Fuente de datos:** NHANES - CDC. **Autor:** Jonathan Florez")
