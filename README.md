
# 🐾 PetFinder Adoption Prediction — Proyecto de Clasificación

Modelo de Machine Learning que predice la **velocidad de adopción** de mascotas a partir de sus características, productivizado en una app de Streamlit.

> Dataset: [PetFinder.my Adoption Prediction](https://www.kaggle.com/c/petfinder-adoption-prediction) — Kaggle  
> Variable objetivo: `AdoptionSpeed` (0 = mismo día · 1 = 1-7 días · 2 = 8-30 días · 3 = 31-90 días · 4 = no adoptado)

---

## 📁 Estructura del proyecto

```
DA-project-classification-Grupo-3/
├── data/
│   ├── train.csv               # Dataset original de Kaggle (NO se sube a git)
│   ├── train_clean.csv         # Dataset tras limpieza (generado por el notebook)
│   └── test.csv                # Dataset de test (NO se sube a git)
├── notebook/
│   ├── Limpieza_PetFinder.ipynb   # Limpieza y tratamiento de datos → train_clean.csv
│   └── EDA_PetFinder.ipynb        # Análisis exploratorio de datos
├── src/
│   └── train.py                # Entrenamiento + evaluación → models/ y reports/
├── app/
│   ├── app.py                  # App de Streamlit (predictor)
│   └── pages/
│       ├── 1_Dashboard.py      # Dashboard de KPIs y métricas
│       └── 2_Nuevos_Datos.py   # Pipeline de ingestión de datos nuevos
├── models/                     # model.joblib + metadata.json + feedback.db (generados)
├── reports/                    # Gráficos: ROC, matriz de confusión, feature importance
├── tests/
│   └── test_model.py           # Tests unitarios (pytest)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutar

### 1. Entorno virtual

```bash
python -m venv .venv

# Activar (Windows Git Bash)
source .venv/Scripts/activate

# Activar (Mac/Linux)
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Descarga el dataset

Descarga `train.csv` y `test.csv` desde [Kaggle](https://www.kaggle.com/c/petfinder-adoption-prediction/data) y colócalos en la carpeta `data/`.

### 3. Limpieza de datos

Ejecuta el notebook `notebook/Limpieza_PetFinder.ipynb` completo (Run All).  
Esto genera `data/train_clean.csv`.

### 4. EDA

Ejecuta el notebook `notebook/EDA_PetFinder.ipynb` completo (Run All).

### 5. Entrenamiento del modelo

```bash
python src/train.py --data data/train_clean.csv
```

Genera automáticamente:
- `models/model.joblib` — modelo entrenado
- `models/metadata.json` — métricas y schema
- `reports/confusion_matrix.png`
- `reports/roc_curve.png`
- `reports/feature_importance.png`

### 6. App de Streamlit

```bash
streamlit run app/app.py
```

Abre [http://localhost:8501](http://localhost:8501)

### 7. Tests

```bash
pytest -q
```

---

## 📊 Resultados del modelo

| Métrica | Valor |
|---|---|
| Algoritmo | GradientBoosting |
| Accuracy (test) | ~39% |
| F1 Macro | ~29% |
| ROC AUC (OvR) | ~0.65 |
| Overfitting | < 5% ✅ |

> El accuracy refleja la dificultad del problema: 5 clases desbalanceadas.  
> Un clasificador aleatorio obtendría solo un 20%. El modelo casi duplica ese valor.

---

## ✅ Cobertura del brief

| Requisito | Estado | Dónde |
|---|---|---|
| Modelo de clasificación funcional | ✅ | `src/train.py` |
| EDA con visualizaciones | ✅ | `notebook/EDA_PetFinder.ipynb` |
| Limpieza de datos | ✅ | `notebook/Limpieza_PetFinder.ipynb` |
| Overfitting < 5% | ✅ | Medido y testeado |
| Productivización (Streamlit) | ✅ | `app/app.py` |
| Dashboard de KPIs | ✅ | `app/pages/1_Dashboard.py` |
| Métricas: accuracy, precision, recall, F1, ROC AUC, matriz de confusión | ✅ | `reports/` |
| Feature importance | ✅ | `reports/feature_importance.png` |
| Ensemble (GradientBoosting / RandomForest) | ✅ | `src/train.py` |
| Validación cruzada (5-Fold) | ✅ | `src/train.py` |
| Ajuste de hiperparámetros (GridSearchCV) | ✅ | `src/train.py` |
| Recogida de feedback en producción | ✅ | SQLite — `models/feedback.db` |
| Pipeline de ingestión de datos nuevos | ✅ | `app/pages/2_Nuevos_Datos.py` |
| Tests unitarios | ✅ | `tests/test_model.py` |

---

## 👥 Equipo — Grupo 3

| Integrante | Responsabilidad |
|---|---|
| **Yasira** | Verificación de datos, validación del modelo (overfitting, residuos), Streamlit |
| **Rita** | apoyo Streamlit, documentación |
| **Romina** |  Pipeline de datos, construcción del Acceptance Index, EDA, modelado |


Proyecto desarrollado en el marco del bootcamp de Data Analytics.

---

## 📌 Notas

- Los archivos `data/train.csv`, `data/test.csv` y `data/train_clean.csv` **no se suben a GitHub** (ver `.gitignore`).
- El modelo se regenera ejecutando `src/train.py`. No se sube `model.joblib` a git.
- La base de datos de feedback (`models/feedback.db`) y de nuevos datos (`models/new_data.db`) se crean automáticamente al lanzar la app.
