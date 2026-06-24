"""
Entrenamiento del modelo de clasificación - PetFinder.my Adoption Prediction.

Uso:
    python src/train.py --data data/train.csv

Qué hace:
  1. Carga y limpieza del CSV de PetFinder.
  2. Ingeniería de features específica (has_name, has_description, is_free...).
  3. Construye un Pipeline de preprocesado (imputación + escalado).
  4. Compara varios modelos con validación cruzada (5-Fold).
  5. Optimiza el mejor con GridSearchCV.
  6. Evalúa en test: accuracy, precision, recall, F1, ROC AUC,
     matriz de confusión y curva ROC multiclase.
  7. Mide el OVERFITTING (train vs test) y avisa si supera el 5%.
  8. Calcula feature importance.
  9. Guarda: models/model.joblib + models/metadata.json + gráficos en reports/.
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
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, roc_curve, auc,
)

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
TARGET = "AdoptionSpeed"

REPORTS_DIR = "reports"
MODELS_DIR  = "models"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

# ── Etiquetas legibles para AdoptionSpeed ─────────────────────────────────────
SPEED_LABELS = {
    0: "Mismo día",
    1: "1-7 días",
    2: "8-30 días",
    3: "31-90 días",
    4: "No adoptado",
}


# ── 1. Carga y limpieza ───────────────────────────────────────────────────────
def load_and_clean(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # Columnas que no aportan al modelo
    drop_cols = ["Name", "Description", "PetID", "RescuerID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    return df


# ── 2. Ingeniería de features ─────────────────────────────────────────────────
def feature_engineering(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Añade features derivadas útiles para el modelo."""
    df = df_raw.copy()

    # Volvemos a leer el CSV original para acceder a Name y Description
    # (ya los habíamos eliminado, así que los recibimos como parámetro separado)
    return df


def feature_engineering_from_raw(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Recibe el CSV original (con todas las columnas) y devuelve el DataFrame
    listo para entrenar, incluyendo features derivadas.
    """
    df = df_raw.copy()

    # ── Features derivadas ──
    # ¿Tiene nombre? Las mascotas con nombre se adoptan más rápido
    df["has_name"] = df["Name"].notna().astype(int)

    # ¿Tiene descripción?
    df["has_description"] = df["Description"].notna().astype(int)

    # Longitud de la descripción (más texto → más información → más confianza)
    df["desc_length"] = df["Description"].fillna("").apply(len)

    # ¿Es gratuita?
    df["is_free"] = (df["Fee"] == 0).astype(int)

    # ¿Tiene fotos?
    df["has_photo"] = (df["PhotoAmt"].fillna(0) > 0).astype(int)

    # ¿Es mestizo? (Breed1 == 307 en PetFinder = Mixed Breed)
    df["is_mixed_breed"] = (df["Breed1"] == 307).astype(int)

    # ¿Anuncio con varias mascotas?
    df["is_multi"] = (df["Quantity"] > 1).astype(int)

    # Cachorro: menor o igual a 3 meses
    df["is_puppy"] = (df["Age"] <= 3).astype(int)

    # Cuidado completo: vacunado + desparasitado + esterilizado (todos = 1)
    df["full_care"] = (
        (df["Vaccinated"] == 1) &
        (df["Dewormed"]   == 1) &
        (df["Sterilized"] == 1)
    ).astype(int)

    # Eliminar columnas no usadas en el modelo
    drop_cols = ["Name", "Description", "PetID", "RescuerID"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    return df


# ── 3. Schema para la app ─────────────────────────────────────────────────────
def build_schema(X: pd.DataFrame) -> dict:
    """Guarda los rangos de cada feature para que la app construya el formulario."""
    schema = {"numeric": {}, "categorical": {}}
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    for c in num_cols:
        schema["numeric"][c] = {
            "min":    float(X[c].min()),
            "max":    float(X[c].max()),
            "median": float(X[c].median()),
        }
    return schema


# ── 4. Gráfico curva ROC multiclase (One-vs-Rest) ────────────────────────────
def plot_roc_multiclass(y_test, y_proba, classes, path):
    y_bin = label_binarize(y_test, classes=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, cls in enumerate(classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{SPEED_LABELS[cls]} (AUC={roc_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curva ROC multiclase (One-vs-Rest)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


# ── 5. Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/train.csv",
                        help="Ruta al train.csv de PetFinder")
    args = parser.parse_args()

    # Carga
    print(f"Cargando {args.data} ...")
    df_raw = pd.read_csv(args.data)
    print(f"Filas originales: {len(df_raw)} | Columnas: {df_raw.shape[1]}")

    # Feature engineering
    df = feature_engineering_from_raw(df_raw)
    print(f"Filas tras limpieza: {len(df)} | Features finales: {df.shape[1] - 1}")

    y = df[TARGET].astype(int)
    X = df.drop(columns=[TARGET])

    # Todas las columnas son numéricas tras el FE
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    print(f"Features numéricas: {len(num_cols)}")
    print(f"Clases: {sorted(y.unique().tolist())} → {[SPEED_LABELS[c] for c in sorted(y.unique())]}")
    print(f"\nDistribución del target:\n{y.value_counts().sort_index().rename(SPEED_LABELS)}")

    schema = build_schema(X)

    # Preprocesado (solo numérico tras FE)
    preprocessor = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # ── Comparación de modelos con validación cruzada ──────────────────────────
    candidates = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=8,
            min_samples_leaf=20, min_samples_split=40,
            class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }

    print("\n=== Validación cruzada (5-Fold, F1 macro) ===")
    cv_results = {}
    for name, clf in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("clf", clf)])
        scores = cross_val_score(pipe, X_train, y_train, cv=5,
                                 scoring="f1_macro", n_jobs=-1)
        cv_results[name] = float(scores.mean())
        print(f"{name:20s}  F1 macro CV = {scores.mean():.4f} (+/- {scores.std():.4f})")

    best_name = max(cv_results, key=cv_results.get)
    print(f"\nMejor modelo base: {best_name}")

    # ── GridSearchCV ────────────────────────────────────────────────────────────
    grids = {
        "LogisticRegression": {"clf__C": [0.1, 1.0, 10.0]},
        "RandomForest": {
            "clf__n_estimators":      [200, 300],
            "clf__max_depth":         [5, 6, 7],
            "clf__min_samples_leaf":  [25, 35, 50],
        },
        "GradientBoosting": {
            "clf__n_estimators":  [150, 250],
            "clf__learning_rate": [0.05, 0.1],
            "clf__max_depth":     [2, 3],
        },
    }

    best_pipe = Pipeline([("prep", preprocessor), ("clf", candidates[best_name])])
    print(f"\nGridSearchCV sobre {best_name} ...")
    grid = GridSearchCV(best_pipe, grids[best_name], cv=5,
                        scoring="f1_macro", n_jobs=-1, verbose=1)
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print(f"Mejores hiperparámetros: {grid.best_params_}")

    # ── Evaluación ──────────────────────────────────────────────────────────────
    y_pred_train  = model.predict(X_train)
    y_pred_test   = model.predict(X_test)
    y_proba_test  = model.predict_proba(X_test)

    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)
    overfit   = train_acc - test_acc

    metrics = {
        "accuracy":         float(test_acc),
        "precision_macro":  float(precision_score(y_test, y_pred_test, average="macro")),
        "recall_macro":     float(recall_score(y_test,    y_pred_test, average="macro")),
        "f1_macro":         float(f1_score(y_test,        y_pred_test, average="macro")),
        "roc_auc_ovr":      float(roc_auc_score(y_test, y_proba_test,
                                                multi_class="ovr", average="macro")),
        "train_accuracy":   float(train_acc),
        "overfitting_gap":  float(overfit),
    }

    print("\n=== Métricas en TEST ===")
    for k, v in metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    print("\n=== Classification Report ===")
    print(classification_report(
        y_test, y_pred_test,
        target_names=[SPEED_LABELS[c] for c in sorted(y.unique())]
    ))

    status = "OK ✅ (< 5%)" if overfit < 0.05 else "REVISAR ⚠️ (>= 5%)"
    print(f"Overfitting (train_acc - test_acc) = {overfit * 100:.2f}%  → {status}")

    # ── Gráficos ────────────────────────────────────────────────────────────────
    # Matriz de confusión
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        confusion_matrix(y_test, y_pred_test),
        display_labels=[SPEED_LABELS[c] for c in sorted(y.unique())]
    ).plot(cmap="Blues", ax=ax, xticks_rotation=30)
    ax.set_title("Matriz de Confusión - PetFinder AdoptionSpeed")
    plt.tight_layout()
    plt.savefig(f"{REPORTS_DIR}/confusion_matrix.png", dpi=120)
    plt.close()
    print(f"Guardado: {REPORTS_DIR}/confusion_matrix.png")

    # Curva ROC multiclase
    classes = sorted(y.unique().tolist())
    plot_roc_multiclass(y_test, y_proba_test, classes,
                        f"{REPORTS_DIR}/roc_curve.png")
    print(f"Guardado: {REPORTS_DIR}/roc_curve.png")

    # Feature importance (permutación)
    try:
        print("\nCalculando feature importance (puede tardar unos segundos)...")
        perm = permutation_importance(
            model, X_test, y_test,
            n_repeats=5, random_state=RANDOM_STATE,
            n_jobs=-1, scoring="f1_macro"
        )
        imp = (pd.Series(perm.importances_mean, index=X.columns)
               .sort_values(ascending=False))
        imp.head(15).iloc[::-1].plot(kind="barh", figsize=(9, 6), color="#3498db")
        plt.title("Feature Importance (permutación) - Top 15")
        plt.xlabel("Reducción media en F1 macro")
        plt.tight_layout()
        plt.savefig(f"{REPORTS_DIR}/feature_importance.png", dpi=120)
        plt.close()
        print(f"Guardado: {REPORTS_DIR}/feature_importance.png")
        top_features = imp.head(15).round(4).to_dict()
    except Exception as e:
        print(f"Aviso: no se pudo calcular feature importance: {e}")
        top_features = {}

    # ── Guardado del modelo y metadata ─────────────────────────────────────────
    joblib.dump(model, f"{MODELS_DIR}/model.joblib")

    metadata = {
        "target":         TARGET,
        "target_classes": classes,
        "target_labels":  {str(k): v for k, v in SPEED_LABELS.items()},
        "best_model":     best_name,
        "best_params":    {k: str(v) for k, v in grid.best_params_.items()},
        "features":       num_cols,
        "metrics":        metrics,
        "cv_results":     cv_results,
        "schema":         schema,
        "top_features":   top_features,
    }
    with open(f"{MODELS_DIR}/metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Guardado: {MODELS_DIR}/model.joblib")
    print(f"✅ Guardado: {MODELS_DIR}/metadata.json")
    print(f"✅ Gráficos en: {REPORTS_DIR}/")
    print("\n¡Entrenamiento completado!")


if __name__ == "__main__":
    main()
