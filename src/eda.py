"""
Análisis Exploratorio de Datos (EDA) — para Romi.

Uso:
    python src/eda.py --data data/train.csv --target satisfaction

Genera en reports/eda/:
  - target_distribution.png   (balance de clases)
  - histograms.png            (distribución de numéricas)
  - correlation_matrix.png    (matriz de correlación)
  - <cat>_vs_target.png       (satisfacción por cada categórica)
  - summary.txt               (describe + nulos + balance)

Estos gráficos son los que pide el brief para el nivel esencial.
Pásalos directamente a la presentación de negocio y al informe técnico.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

OUT = "reports/eda"
os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/train.csv")
    p.add_argument("--target", default="satisfaction")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    df = df.drop(columns=[c for c in df.columns
                          if c.lower() in ("unnamed: 0", "id", "index")],
                 errors="ignore")

    # Resumen
    with open(f"{OUT}/summary.txt", "w", encoding="utf-8") as f:
        f.write("=== INFO ===\n")
        f.write(f"Filas: {len(df)}  Columnas: {df.shape[1]}\n\n")
        f.write("=== NULOS ===\n")
        f.write(df.isna().sum().to_string() + "\n\n")
        f.write("=== DESCRIBE ===\n")
        f.write(df.describe(include="all").to_string() + "\n")

    # Balance del target
    plt.figure(figsize=(6, 4))
    sns.countplot(x=args.target, data=df)
    plt.title("Distribución de la variable objetivo")
    plt.tight_layout()
    plt.savefig(f"{OUT}/target_distribution.png", dpi=120)
    plt.close()

    num = df.select_dtypes(include=np.number)
    cat = df.select_dtypes(exclude=np.number).drop(columns=[args.target],
                                                   errors="ignore")

    # Histogramas
    if not num.empty:
        num.hist(figsize=(16, 12), bins=30)
        plt.suptitle("Histogramas de variables numéricas")
        plt.tight_layout()
        plt.savefig(f"{OUT}/histograms.png", dpi=110)
        plt.close()

        # Matriz de correlación
        plt.figure(figsize=(14, 11))
        sns.heatmap(num.corr(), annot=False, cmap="coolwarm", center=0)
        plt.title("Matriz de correlación")
        plt.tight_layout()
        plt.savefig(f"{OUT}/correlation_matrix.png", dpi=120)
        plt.close()

    # Satisfacción por cada categórica
    for c in cat.columns:
        plt.figure(figsize=(7, 4))
        sns.countplot(x=c, hue=args.target, data=df)
        plt.title(f"{args.target} por {c}")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(f"{OUT}/{c}_vs_target.png".replace(" ", "_"), dpi=110)
        plt.close()

    print(f"EDA generado en {OUT}/")


if __name__ == "__main__":
    main()
