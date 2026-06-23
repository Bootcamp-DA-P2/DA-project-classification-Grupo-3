# ✈️ Predicción de satisfacción de pasajeros — Proyecto de Clasificación

Modelo de Machine Learning que predice la satisfacción de los clientes de una
aerolínea a partir de sus datos, productivizado en una app de Streamlit.

## 📁 Estructura
```
airline-satisfaction/
├── data/                 # train.csv (NO se sube a git)
├── src/
│   ├── eda.py            # Análisis exploratorio  -> reports/eda/
│   └── train.py          # Entrenamiento + evaluación -> models/ y reports/
├── app/
│   └── app.py            # App de Streamlit (productivización)
├── models/               # model.joblib + metadata.json + feedback.db (generados)
├── reports/              # Gráficos: EDA, ROC, matriz confusión, importancias
├── tests/
│   └── test_model.py     # Tests unitarios (pytest)
├── requirements.txt
├── Dockerfile
└── README.md
```

## 🚀 Cómo ejecutar

```bash
# 1. Entorno
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Coloca el dataset en data/train.csv
#    (ajusta --target si la columna objetivo se llama distinto)

# 3. EDA
python src/eda.py --data data/train.csv --target satisfaction

# 4. Entrenamiento (genera el modelo y todas las métricas)
python src/train.py --data data/train.csv --target satisfaction

# 5. App
streamlit run app/app.py

# 6. Tests
pytest -q
```

### 🐳 Docker (nivel avanzado)
```bash
docker build -t airline-app .
docker run -p 8501:8501 airline-app
# Abrir http://localhost:8501
```

## ✅ Cobertura del brief
- Modelo de clasificación funcional ✔ (`train.py`)
- EDA con visualizaciones ✔ (`eda.py`)
- Control de overfitting < 5% ✔ (se mide y se testea)
- Productivización ✔ (`app/app.py`)
- Métricas: accuracy, precision, recall, F1, ROC AUC, matriz de confusión ✔
- Feature importance ✔
- Ensemble (RandomForest / GradientBoosting) ✔
- Validación cruzada (5-Fold) ✔
- Ajuste de hiperparámetros (GridSearchCV) ✔
- Recogida de feedback y de datos nuevos (SQLite) ✔
- Dockerización ✔ · Base de datos ✔ · Tests unitarios ✔

## 🌿 Flujo de Git (ramas y commits limpios)
```
main          # solo versiones estables
develop       # integración
feature/eda            -> Romi
feature/modeling       -> Yasira
feature/streamlit-app  -> Rita
```
Convención de commits (Conventional Commits):
`feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`
Ej: `feat: add GridSearch tuning to training pipeline`
```bash
git checkout -b feature/modeling
git add . && git commit -m "feat: baseline model comparison with CV"
git push -u origin feature/modeling
# luego Pull Request -> develop -> main
```
