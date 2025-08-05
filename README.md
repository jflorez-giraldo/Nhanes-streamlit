# 🧪 PCA Analysis with NHANES Data - Streamlit App

This project presents a **step-by-step PCA (Principal Component Analysis)** on a simplified version of the NHANES dataset (National Health and Nutrition Examination Survey), aiming to identify latent patterns and classify diabetes risk.

![Streamlit App](https://static.streamlit.io/examples/dashboards.png)

---

## 📊 Features

- Missing data imputation with KNN
- Standardization of clinical variables
- Dimensionality reduction using PCA
- Visualization of principal components
- Patient clustering with KMeans
- Diabetes risk classification using logistic regression
- **Interactive filters** by age, glucose level, and diagnosis

---

## 🚀 How to Run

### 1. Option A: Run locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Then launch the app:

```bash
streamlit run app_pca_nhanes_filtros.py
```

---

### 2. Option B: Deploy on [Streamlit Cloud](https://streamlit.io/cloud)

1. Upload this repository to your GitHub account.
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Click **"New app"**, select your repo and the file `app_pca_nhanes_filtros.py`.
4. Done! The app will be publicly available online.

---

## 📁 Repository Structure

```
📦 pca-nhanes-streamlit
├── app_pca_nhanes_filtros.py      # Main Streamlit app code
├── requirements.txt               # Required dependencies
└── README.md                      # This file
```

---

## 📚 Dataset Source

The data is sourced from a public simplified version of NHANES, available here:  
🔗 https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset

---

## 📄 License

This project is licensed under the MIT License. Feel free to use, modify, and share it.
