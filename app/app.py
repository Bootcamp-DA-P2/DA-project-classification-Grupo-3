"""
App de Streamlit — PetFinder Adoption Speed Predictor.

Ejecutar desde la raíz del proyecto (DA-project-classification-Grupo-3/):
    streamlit run app/app.py

Requiere:
    models/model.joblib
    models/metadata.json

Generados por:
    python src/train.py --data data/train.csv
"""

import json
import os
import sqlite3
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Rutas ─────────────────────────────────────────────────────────────────────
MODEL_PATH = "models/model.joblib"
META_PATH  = "models/metadata.json"
DB_PATH    = "models/feedback.db"

# ── Etiquetas de velocidad de adopción ────────────────────────────────────────
SPEED_LABELS = {
    "0": "Mismo día 🏃",
    "1": "1–7 días 📅",
    "2": "8–30 días 🗓️",
    "3": "31–90 días ⏳",
    "4": "No adoptado 😢",
}

SPEED_COLORS = {
    "0": "success",
    "1": "success",
    "2": "warning",
    "3": "warning",
    "4": "error",
}

# ── Nombres amigables para las features ───────────────────────────────────────
FRIENDLY_NAMES = {
    "Type":         "Tipo de animal",
    "Age":          "Edad (meses)",
    "Breed1":       "Raza principal",
    "Breed2":       "Raza secundaria",
    "Gender":       "Género",
    "Color1":       "Color principal",
    "Color2":       "Color secundario",
    "Color3":       "Color terciario",
    "MaturitySize": "Tamaño adulto",
    "FurLength":    "Longitud del pelo",
    "Vaccinated":   "Vacunado",
    "Dewormed":     "Desparasitado",
    "Sterilized":   "Esterilizado",
    "Health":       "Estado de salud",
    "Quantity":     "Cantidad de mascotas",
    "Fee":          "Tarifa de adopción (MYR)",
    "State":        "Estado (Malasia)",
    "VideoAmt":     "Número de vídeos",
    "PhotoAmt":     "Número de fotos",
    "has_name":     "¿Tiene nombre?",
    "has_description": "¿Tiene descripción?",
    "desc_length":  "Longitud descripción (caracteres)",
    "is_free":      "¿Adopción gratuita?",
    "has_photo":    "¿Tiene fotos?",
    "is_mixed_breed": "¿Es mestizo?",
    "is_multi":     "¿Anuncio con varias mascotas?",
    "is_puppy":     "¿Es cachorro? (≤ 3 meses)",
    "full_care":    "¿Cuidado completo? (vacuna + desparasitado + esterilizado)",
}

# ── Opciones para variables categóricas codificadas ───────────────────────────
CATEGORICAL_OPTIONS = {
    "Type":         {1: "Perro 🐶", 2: "Gato 🐱"},
    "Gender":       {1: "Macho", 2: "Hembra", 3: "Mixto"},
    "MaturitySize": {1: "Pequeño", 2: "Mediano", 3: "Grande", 4: "Extra grande"},
    "FurLength":    {1: "Corto", 2: "Mediano", 3: "Largo"},
    "Vaccinated":   {1: "Sí", 2: "No", 3: "No sé"},
    "Dewormed":     {1: "Sí", 2: "No", 3: "No sé"},
    "Sterilized":   {1: "Sí", 2: "No", 3: "No sé"},
    "Health":       {1: "Saludable", 2: "Lesión menor", 3: "Lesión grave"},
    "has_name":     {1: "Sí", 0: "No"},
    "has_description": {1: "Sí", 0: "No"},
    "is_free":      {1: "Sí", 0: "No"},
    "has_photo":    {1: "Sí", 0: "No"},
    "is_mixed_breed": {1: "Sí", 0: "No"},
    "is_multi":     {1: "Sí", 0: "No"},
    "is_puppy":     {1: "Sí", 0: "No"},
    "full_care":    {1: "Sí", 0: "No"},
}

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="PetFinder — Predictor de Adopción",
    page_icon="🐾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .main-header p {
        color: #666;
        font-size: 1rem;
    }
    .metric-pill {
        background: #f0f4ff;
        border-radius: 8px;
        padding: 0.4rem 0.8rem;
        font-size: 0.85rem;
        color: #3355cc;
        display: inline-block;
        margin: 0.2rem;
    }
    .section-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1a1a2e;
        margin-bottom: 0.8rem;
        padding-bottom: 0.3rem;
        border-bottom: 2px solid #e8ecf5;
    }
    .result-box {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    .feedback-row {
        display: flex;
        gap: 1rem;
        justify-content: center;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e8ecf5;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


# ── Cache: carga del modelo y metadata ───────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(META_PATH, encoding="utf-8") as f:
        meta = json.load(f)
    return model, meta


# ── Base de datos de feedback ─────────────────────────────────────────────────
def init_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            ts          TEXT,
            inputs      TEXT,
            prediction  TEXT,
            probability REAL,
            feedback    TEXT
        )
    """)
    conn.commit()
    return conn


def log_prediction(conn, inputs: dict, prediction: str,
                   prob: float, feedback: str = None):
    conn.execute(
        "INSERT INTO predictions (ts, inputs, prediction, probability, feedback) "
        "VALUES (?,?,?,?,?)",
        (datetime.now().isoformat(),
         json.dumps(inputs, ensure_ascii=False),
         prediction, float(prob), feedback),
    )
    conn.commit()


# ── Comprobación de archivos necesarios ───────────────────────────────────────
if not os.path.exists(MODEL_PATH) or not os.path.exists(META_PATH):
    st.error(
        "⚠️ No se encontró el modelo entrenado. "
        "Ejecuta primero:\n\n`python src/train.py --data data/train.csv`"
    )
    st.stop()

model, meta = load_artifacts()
conn = init_db()

schema       = meta["schema"]
features     = meta["features"]          # lista ordenada de columnas que espera el modelo
target_labels = meta["target_labels"]    # {"0": "Mismo día", ...}
metrics      = meta["metrics"]
best_model   = meta["best_model"]
top_features = meta.get("top_features", {})

# ── Cabecera ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🐾 PetFinder — Predictor de Adopción</h1>
    <p>Introduce los datos de la mascota para estimar su velocidad de adopción</p>
</div>
""", unsafe_allow_html=True)

# Métricas del modelo en pills
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
col_m1.markdown(f'<span class="metric-pill">🤖 Modelo: {best_model}</span>', unsafe_allow_html=True)
col_m2.markdown(f'<span class="metric-pill">🎯 Accuracy: {metrics["accuracy"]:.1%}</span>', unsafe_allow_html=True)
col_m3.markdown(f'<span class="metric-pill">📊 F1 macro: {metrics["f1_macro"]:.1%}</span>', unsafe_allow_html=True)
col_m4.markdown(f'<span class="metric-pill">📈 ROC AUC: {metrics["roc_auc_ovr"]:.1%}</span>', unsafe_allow_html=True)

st.divider()

# ── Formulario de entrada ─────────────────────────────────────────────────────
st.markdown('<p class="section-title">Datos de la mascota</p>', unsafe_allow_html=True)

inputs = {}
num_schema = schema.get("numeric", {})

# Separamos features en bloques temáticos para mejor UX
BLOCKS = {
    "🐶 Información básica": ["Type", "Age", "Gender", "Quantity", "is_puppy", "is_multi"],
    "🧬 Raza y físico":      ["Breed1", "Breed2", "MaturitySize", "FurLength",
                               "Color1", "Color2", "Color3", "is_mixed_breed"],
    "💉 Salud y cuidados":   ["Vaccinated", "Dewormed", "Sterilized", "Health", "full_care"],
    "📸 Anuncio":            ["Fee", "PhotoAmt", "VideoAmt", "State",
                               "has_name", "has_description", "desc_length",
                               "is_free", "has_photo"],
}

# Agrupamos las features en columnas dentro de cada bloque
for block_title, block_features in BLOCKS.items():
    # Solo mostramos el bloque si alguna de sus features está en el modelo
    block_active = [f for f in block_features if f in features]
    if not block_active:
        continue

    with st.expander(block_title, expanded=True):
        cols = st.columns(3)
        for idx, feat in enumerate(block_active):
            label = FRIENDLY_NAMES.get(feat, feat)
            with cols[idx % 3]:
                if feat in CATEGORICAL_OPTIONS:
                    options_dict = CATEGORICAL_OPTIONS[feat]
                    selected_label = st.selectbox(
                        label,
                        options=list(options_dict.values()),
                        key=feat,
                    )
                    # Recuperamos el valor numérico
                    inputs[feat] = [k for k, v in options_dict.items()
                                    if v == selected_label][0]
                elif feat in num_schema:
                    info = num_schema[feat]
                    lo, hi, med = float(info["min"]), float(info["max"]), float(info["median"])
                    if lo == hi:
                        hi = lo + 1
                    step = 1.0 if (hi - lo) > 20 else 0.1
                    inputs[feat] = st.slider(label, lo, hi, med, step, key=feat)
                else:
                    # Feature binaria derivada sin schema explícito
                    inputs[feat] = st.selectbox(label, [0, 1],
                                                format_func=lambda x: "Sí" if x else "No",
                                                key=feat)

# Features que no están en ningún bloque (por si acaso)
remaining = [f for f in features if f not in inputs]
if remaining:
    with st.expander("➕ Otros datos", expanded=False):
        cols = st.columns(3)
        for idx, feat in enumerate(remaining):
            label = FRIENDLY_NAMES.get(feat, feat)
            with cols[idx % 3]:
                if feat in num_schema:
                    info = num_schema[feat]
                    lo, hi, med = float(info["min"]), float(info["max"]), float(info["median"])
                    if lo == hi:
                        hi = lo + 1
                    step = 1.0 if (hi - lo) > 20 else 0.1
                    inputs[feat] = st.slider(label, lo, hi, med, step, key=feat)
                else:
                    inputs[feat] = st.number_input(label, value=0, key=feat)

st.divider()

# ── Predicción ────────────────────────────────────────────────────────────────
if st.button("🔍 Predecir velocidad de adopción", type="primary", use_container_width=True):

    # Construimos el DataFrame en el orden exacto que espera el modelo
    row = {f: inputs.get(f, 0) for f in features}
    X = pd.DataFrame([row])[features]

    pred_num  = int(model.predict(X)[0])
    proba_arr = model.predict_proba(X)[0]
    prob      = float(proba_arr.max())
    label     = target_labels.get(str(pred_num), str(pred_num))

    # Color según resultado
    color_type = SPEED_COLORS.get(str(pred_num), "info")
    if color_type == "success":
        st.success(f"### {label}", icon="✅")
    elif color_type == "warning":
        st.warning(f"### {label}", icon="⏳")
    else:
        st.error(f"### {label}", icon="😢")

    st.write(f"**Confianza:** {prob:.1%}")

    # Distribución de probabilidades por clase
    classes = meta.get("target_classes", list(range(len(proba_arr))))
    prob_df = pd.DataFrame({
        "Velocidad": [target_labels.get(str(c), str(c)) for c in classes],
        "Probabilidad": proba_arr,
    }).sort_values("Probabilidad", ascending=False)
    st.bar_chart(prob_df.set_index("Velocidad")["Probabilidad"])

    # Guardamos en BD
    log_prediction(conn, inputs, label, prob)
    st.session_state["last"] = (inputs, label, prob)

# ── Feedback ──────────────────────────────────────────────────────────────────
if "last" in st.session_state:
    st.divider()
    st.markdown("**¿La predicción fue correcta?**")
    c1, c2 = st.columns(2)
    if c1.button("👍 Sí, acertó", use_container_width=True):
        inp, lab, pr = st.session_state["last"]
        log_prediction(conn, inp, lab, pr, feedback="correct")
        st.toast("✅ Feedback guardado — ¡gracias!")
    if c2.button("👎 No, falló", use_container_width=True):
        inp, lab, pr = st.session_state["last"]
        log_prediction(conn, inp, lab, pr, feedback="wrong")
        st.toast("📝 Feedback guardado — lo tendremos en cuenta")

# ── Panel de monitorización ───────────────────────────────────────────────────
with st.expander("📊 Monitorización (feedback recogido)", expanded=False):
    df_log = pd.read_sql_query("SELECT * FROM predictions", conn)
    st.write(f"**Predicciones registradas:** {len(df_log)}")

    fb = df_log[df_log["feedback"].notna()]
    if len(fb):
        acc_feedback = (fb["feedback"] == "correct").mean()
        c1, c2 = st.columns(2)
        c1.metric("Accuracy según usuarios", f"{acc_feedback:.1%}")
        c2.metric("Predicciones con feedback", len(fb))

        dist = fb["prediction"].value_counts().reset_index()
        dist.columns = ["Clase predicha", "Feedback recibido"]
        st.dataframe(dist, use_container_width=True, hide_index=True)
    else:
        st.info("Aún no hay feedback registrado. Realiza una predicción y valídala.")

    if len(df_log):
        st.markdown("**Últimas 20 predicciones:**")
        st.dataframe(
            df_log[["ts", "prediction", "probability", "feedback"]].tail(20),
            use_container_width=True,
            hide_index=True,
        )

# ── Feature importance (si existe) ───────────────────────────────────────────
if top_features:
    with st.expander("🔬 Features más importantes del modelo", expanded=False):
        imp_df = (pd.Series(top_features)
                    .sort_values(ascending=False)
                    .reset_index())
        imp_df.columns = ["Feature", "Importancia (reducción F1)"]
        imp_df["Feature"] = imp_df["Feature"].map(
            lambda x: FRIENDLY_NAMES.get(x, x)
        )
        st.bar_chart(imp_df.set_index("Feature")["Importancia (reducción F1)"])