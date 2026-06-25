"""
Tests unitarios (nivel avanzado).
Ejecutar:  pytest -q   (desde la raíz, con el modelo ya entrenado)
"""

import json
import os

import joblib
import pandas as pd
import pytest

META = "models/metadata.json"
MODEL = "models/model.joblib"

needs_model = pytest.mark.skipif(
    not (os.path.exists(META) and os.path.exists(MODEL)),
    reason="Entrena primero: python src/train.py",
)


@needs_model
def test_metadata_has_schema():
    meta = json.load(open(META, encoding="utf-8"))
    assert "schema" in meta
    assert meta["schema"]["numeric"] or meta["schema"]["categorical"]


@needs_model
def test_minimum_metrics():
    """El modelo debe superar un mínimo aceptable y no sobreajustar."""
    m = json.load(open(META, encoding="utf-8"))["metrics"]
    assert m["accuracy"] >= 0.30, "Accuracy por debajo del mínimo aceptable"
    assert m["f1_macro"] >= 0.20, "F1 macro por debajo del mínimo aceptable"
    assert m["overfitting_gap"] < 0.05, "Overfitting >= 5%"


@needs_model
def test_model_predicts_one_row():
    meta = json.load(open(META, encoding="utf-8"))
    model = joblib.load(MODEL)
    row = {}
    for c, opts in meta["schema"]["categorical"].items():
        row[c] = opts[0]
    for c, info in meta["schema"]["numeric"].items():
        row[c] = info["median"]
    pred = model.predict(pd.DataFrame([row]))
    assert len(pred) == 1
