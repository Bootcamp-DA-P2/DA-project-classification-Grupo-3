"""
Entrenamiento del modelo de clasificación (satisfacción de pasajeros).

Uso:
    python src/train.py --data data/train.csv --target satisfaction

Qué hace:
  1. Carga y limpieza básica del CSV.
  2. Detecta automáticamente columnas numéricas y categóricas.
  3. Construye un Pipeline de preprocesado (imputación + escalado + one-hot).
  4. Compara varios modelos con validación cruzada (5-Fold).
  5. Optimiza el mejor con GridSearchCV.
  6. Evalúa en test: accuracy, precision, recall, F1, ROC AUC, matriz de confusión.
  7. Mide el OVERFITTING (train vs test) y avisa si supera el 5%.
  8. Calcula feature importance.
  9. Guarda: models/model.joblib  +  models/metadata.json  +  gráficos en reports/.

El esquema de features se guarda en metadata.json para que la app de Streamlit
construya el formulario sola, sin depender de los nombres exactos de columnas.
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report, RocCurveDisplay, ConfusionMatrixDisplay,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

REPORTS_DIR = "reports"
MODELS_DIR = "models"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_clean(path, target):
    df = pd.read_csv(path)

    # Quitar columnas índice/id típicas de este dataset
    drop_cols = [c for c in df.columns
                 if c.lower() in ("unnamed: 0", "id", "index", "")]
    df = df.drop(columns=drop_cols, errors="ignore")

    if target not in df.columns:
        raise ValueError(
            f"No encuentro la columna objetivo '{target}'. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    # Eliminar filas sin etiqueta y duplicados
    df = df.dropna(subset=[target]).drop_duplicates().reset_index(drop=True)
    return df


def encode_target(y):
    """Convierte el target a 0/1. Devuelve (y_codificado, mapa, clases)."""
    if y.dtype == object or str(y.dtype).startswith("category"):
        classes = sorted(y.unique().tolist())
        # Heurística: la clase "positiva" es 'satisfied' si existe
        pos = next((c for c in classes if "satisf" in str(c).lower()
                    and "neutral" not in str(c).lower()
                    and "dissatisf" not in str(c).lower()), classes[-1])
        mapping = {c: (1 if c == pos else 0) for c in classes}
        return y.map(mapping).astype(int), mapping, classes
    else:
        return y.astype(int), {0: 0, 1: 1}, sorted(y.unique().tolist())


def build_schema(X, num_cols, cat_cols):
    """Esquema de features para que la app construya el formulario sola."""
    schema = {"numeric": {}, "categorical": {}}
    for c in num_cols:
        schema["numeric"][c] = {
            "min": float(X[c].min()),
            "max": float(X[c].max()),
            "median": float(X[c].median()),
        }
    for c in cat_cols:
        schema["categorical"][c] = sorted(
            [str(v) for v in X[c].dropna().unique().tolist()]
        )
    return schema


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv")
    parser.add_argument("--target", default="satisfaction")
    args = parser.parse_args()

    print(f"Cargando {args.data} ...")
    df = load_and_clean(args.data, args.target)
    print(f"Filas: {len(df)} | Columnas: {df.shape[1]}")

    y, target_map, target_classes = encode_target(df[args.target])
    X = df.drop(columns=[args.target])

    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    print(f"Numéricas: {len(num_cols)} | Categóricas: {len(cat_cols)}")

    schema = build_schema(X, num_cols, cat_cols)

    # Preprocesado
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # --- Comparación de modelos con validación cruzada ---
    # max_depth limitado en árboles => ayuda a mantener el overfitting < 5%
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    print("\n=== Validación cruzada (5-Fold, accuracy) ===")
    cv_results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5,
                                 scoring="accuracy", n_jobs=-1)
        cv_results[name] = float(scores.mean())
        print(f"{name:20s}  acc CV = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_results, key=cv_results.get)
    print(f"\nMejor modelo base: {best_name}")

    # --- Ajuste de hiperparámetros del mejor modelo (grids pequeños = rápido) ---
    grids = {
        "LogisticRegression": {"clf__C": [0.1, 1.0, 10.0]},
        "RandomForest": {"clf__n_estimators": [200, 300],
                         "clf__max_depth": [10, 14, 18]},
        "GradientBoosting": {"clf__n_estimators": [150, 250],
                             "clf__learning_rate": [0.05, 0.1],
                             "clf__max_depth": [2, 3]},
    }
    best_pipe = Pipeline([("prep", preprocessor), ("clf", candidates[best_name])])
    print(f"\nGridSearchCV sobre {best_name} ...")
    grid = GridSearchCV(best_pipe, grids[best_name], cv=5,
                        scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print(f"Mejores hiperparámetros: {grid.best_params_}")

    # --- Evaluación ---
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_proba_test = model.predict_proba(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    overfit = train_acc - test_acc

    metrics = {
        "accuracy": float(test_acc),
        "precision": float(precision_score(y_test, y_pred_test, average='macro')),
        "recall": float(recall_score(y_test, y_pred_test, average='macro')),
        "f1": float(f1_score(y_test, y_pred_test, average='macro')),
        "roc_auc": float(roc_auc_score(y_test, y_proba_test, multi_class='ovr')),
        "train_accuracy": float(train_acc),
        "overfitting_gap": float(overfit),
}


    print("\n=== Métricas en TEST ===")
    for k, v in metrics.items():
        print(f"{k:18s}: {v:.4f}")
    print("\n", classification_report(y_test, y_pred_test))

    status = "OK ✅ (< 5%)" if overfit < 0.05 else "REVISAR ⚠️ (>= 5%)"
    print(f"Overfitting (train_acc - test_acc) = {overfit*100:.2f}%  -> {status}")

    # --- Gráficos ---
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test)).plot(cmap="Blues")
    plt.title("Matriz de confusión")
    plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png", bbox_inches="tight", dpi=120)
    plt.close()

    # RocCurveDisplay.from_predictions(y_test, y_proba_test)
    plt.title(f"Curva ROC (AUC = {metrics['roc_auc']:.3f})")
    # plt.savefig(f"{REPORTS_DIR}/roc_curve.png", bbox_inches="tight", dpi=120)
    plt.close()

    # --- Feature importance (permutación, robusta para cualquier modelo) ---
    try:
        perm = permutation_importance(
            model, X_test, y_test, n_repeats=5,
            random_state=RANDOM_STATE, n_jobs=-1, scoring="f1"
        )
        imp = (pd.Series(perm.importances_mean, index=X.columns)
               .sort_values(ascending=False))
        imp.head(15).iloc[::-1].plot(kind="barh", figsize=(8, 6))
        plt.title("Feature importance (permutación)")
        plt.tight_layout()
        plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=120)
        plt.close()
        top_features = imp.head(15).round(4).to_dict()
    except Exception as e:
        print(f"Aviso: no se pudo calcular feature importance: {e}")
        top_features = {}

    # --- Guardado ---
    joblib.dump(model, f"{MODELS_DIR}/model.joblib")
    metadata = {
        "target": args.target,
        "target_map": {str(k): v for k, v in target_map.items()},
        "target_classes": [str(c) for c in target_classes],
        "best_model": best_name,
        "best_params": {k: str(v) for k, v in grid.best_params_.items()},
        "metrics": metrics,
        "cv_results": cv_results,
        "schema": schema,
        "top_features": top_features,
    }
    with open(f"{MODELS_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nGuardado: {MODELS_DIR}/model.joblib y {MODELS_DIR}/metadata.json")
    print(f"Gráficos en: {REPORTS_DIR}/")


if __name__ == "__main__":
    main()
