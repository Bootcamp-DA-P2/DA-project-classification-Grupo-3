"""
App de Streamlit: recibe los datos de un cliente y devuelve la predicción.

Ejecutar (desde la raíz del proyecto):
    streamlit run app/app.py

Construye el formulario automáticamente a partir de models/metadata.json,
así que no hay que tocar nada aunque cambien las columnas del dataset.
Incluye:
  - Predicción con probabilidad.
  - Recogida de feedback (¿acertó el modelo?) guardada en SQLite
    -> sirve para el nivel medio (monitorización) y para reentrenar.
"""

import json
import os
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import streamlit as st

MODEL_PATH = "models/model.joblib"
META_PATH = "models/metadata.json"
DB_PATH = "models/feedback.db"

st.set_page_config(page_title="Predicción de satisfacción", page_icon="✈️",
                   layout="centered")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            ts TEXT, inputs TEXT, prediction TEXT,
            probability REAL, feedback TEXT
        )
    """)
    conn.commit()
    return conn


def log_prediction(conn, inputs, prediction, prob, feedback=None):
    conn.execute(
        "INSERT INTO predictions VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(), json.dumps(inputs, ensure_ascii=False),
         str(prediction), float(prob), feedback),
    )
    conn.commit()


# ---------------------------------------------------------------------------
if not (os.path.exists(MODEL_PATH) and os.path.exists(META_PATH)):
    st.error("No encuentro el modelo. Ejecuta primero:  python src/train.py")
    st.stop()

model, meta = load_artifacts()
conn = init_db()
schema = meta["schema"]
classes = meta["target_classes"]

st.title("✈️ Predicción de satisfacción del cliente")
st.caption(f"Modelo: {meta['best_model']}  ·  "
           f"F1: {meta['metrics']['f1']:.3f}  ·  "
           f"ROC AUC: {meta['metrics']['roc_auc']:.3f}")

st.subheader("Introduce los datos del cliente")

inputs = {}
cols = st.columns(2)
i = 0

# Campos categóricos -> selectbox
for name, options in schema["categorical"].items():
    with cols[i % 2]:
        inputs[name] = st.selectbox(name, options)
    i += 1

# Campos numéricos -> slider con rango del dataset
for name, info in schema["numeric"].items():
    with cols[i % 2]:
        lo, hi, med = info["min"], info["max"], info["median"]
        if lo == hi:
            hi = lo + 1
        step = 1.0 if (hi - lo) > 20 else 0.1
        inputs[name] = st.slider(name, float(lo), float(hi), float(med), step=step)
    i += 1

if st.button("Predecir", type="primary"):
    X = pd.DataFrame([inputs])
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0].max()

    # Traducir 0/1 a etiqueta legible
    inv_map = {v: k for k, v in meta["target_map"].items()}
    label = inv_map.get(int(pred), str(pred))

    if "satisf" in label.lower() and "dissatisf" not in label.lower() \
            and "neutral" not in label.lower():
        st.success(f"### Resultado: {label}  ({prob*100:.1f}% de confianza)")
    else:
        st.warning(f"### Resultado: {label}  ({prob*100:.1f}% de confianza)")

    log_prediction(conn, inputs, label, prob)
    st.session_state["last"] = (inputs, label, prob)

# Feedback (monitorización en producción)
if "last" in st.session_state:
    st.divider()
    st.write("¿La predicción fue correcta?")
    c1, c2 = st.columns(2)
    if c1.button("👍 Acertó"):
        inp, lab, pr = st.session_state["last"]
        log_prediction(conn, inp, lab, pr, feedback="correct")
        st.toast("Feedback guardado")
    if c2.button("👎 Falló"):
        inp, lab, pr = st.session_state["last"]
        log_prediction(conn, inp, lab, pr, feedback="wrong")
        st.toast("Feedback guardado")

# Mini panel de monitorización
with st.expander("📊 Monitorización (feedback recogido)"):
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    st.write(f"Predicciones registradas: {len(df)}")
    fb = df[df["feedback"].notna()]
    if len(fb):
        acc = (fb["feedback"] == "correct").mean()
        st.metric("Accuracy según feedback de usuarios", f"{acc*100:.1f}%")
    st.dataframe(df.tail(20), use_container_width=True)
