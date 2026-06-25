"""
Pipeline de ingestión de datos nuevos — PetFinder Adoption Prediction.

Página de Streamlit para recoger datos reales de nuevas mascotas
y guardarlos en SQLite para futuros reentrenamientos del modelo.

Se coloca en: app/pages/2_Nuevos_Datos.py
Ejecutar desde la raíz: streamlit run app/app.py
"""

import json
import os
import sqlite3
from datetime import datetime

import pandas as pd
import streamlit as st

# ── Rutas ──────────────────────────────────────────────────────────────────────
DB_NEW_DATA = "models/new_data.db"
META_PATH   = "models/metadata.json"

SPEED_LABELS = {
    0: "Mismo día",
    1: "1-7 días",
    2: "8-30 días",
    3: "31-90 días",
    4: "No adoptado",
}

st.set_page_config(
    page_title="Nuevos Datos — PetFinder",
    page_icon="📥",
    layout="wide",
)

st.markdown("""
<style>
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1a1a2e;
        border-bottom: 2px solid #e8ecf5;
        padding-bottom: 0.4rem;
        margin: 1.2rem 0 1rem 0;
    }
    .info-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 4px solid #3498db;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Base de datos de nuevos datos ─────────────────────────────────────────────
def init_new_data_db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_NEW_DATA)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS new_pets (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            ts              TEXT,
            Type            INTEGER,
            Age             INTEGER,
            Gender          INTEGER,
            MaturitySize    INTEGER,
            FurLength       INTEGER,
            Vaccinated      INTEGER,
            Dewormed        INTEGER,
            Sterilized      INTEGER,
            Health          INTEGER,
            Quantity        INTEGER,
            Fee             REAL,
            PhotoAmt        INTEGER,
            VideoAmt        INTEGER,
            has_name        INTEGER,
            has_description INTEGER,
            is_free         INTEGER,
            AdoptionSpeed   INTEGER
        )
    """)
    conn.commit()
    return conn


def insertar_mascota(conn, datos: dict):
    cols = ", ".join(datos.keys())
    placeholders = ", ".join(["?"] * len(datos))
    conn.execute(
        f"INSERT INTO new_pets (ts, {cols}) VALUES (?, {placeholders})",
        [datetime.now().isoformat()] + list(datos.values()),
    )
    conn.commit()


conn = init_new_data_db()

# ── Cabecera ───────────────────────────────────────────────────────────────────
st.markdown("# 📥 Ingesta de Nuevos Datos")
st.markdown("""
<div class="info-box">
    <b>¿Para qué sirve esta página?</b><br>
    Permite registrar datos reales de mascotas <b>con su resultado de adopción conocido</b>.
    Estos datos se almacenan en una base de datos SQLite y podrán usarse para
    <b>reentrenar el modelo</b> en el futuro con información más reciente.
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Formulario ─────────────────────────────────────────────────────────────────
st.markdown('<p class="section-title">🐾 Datos de la mascota</p>', unsafe_allow_html=True)

with st.form("nueva_mascota", clear_on_submit=True):

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Información básica**")
        tipo     = st.selectbox("Tipo de animal", [1, 2],
                                format_func=lambda x: "Perro 🐶" if x == 1 else "Gato 🐱")
        edad     = st.number_input("Edad (meses)", min_value=1, max_value=144, value=3)
        genero   = st.selectbox("Género", [1, 2, 3],
                                format_func=lambda x: {1:"Macho",2:"Hembra",3:"Mixto"}[x])
        cantidad = st.number_input("Cantidad de mascotas", min_value=1, max_value=10, value=1)

    with col2:
        st.markdown("**Características físicas**")
        tamaño   = st.selectbox("Tamaño adulto", [1, 2, 3, 4],
                                format_func=lambda x: {1:"Pequeño",2:"Mediano",
                                                        3:"Grande",4:"Extra grande"}[x])
        pelo     = st.selectbox("Largo del pelo", [1, 2, 3],
                                format_func=lambda x: {1:"Corto",2:"Mediano",3:"Largo"}[x])
        vacuna   = st.selectbox("Vacunado", [1, 2, 3],
                                format_func=lambda x: {1:"Sí",2:"No",3:"No sé"}[x])
        despar   = st.selectbox("Desparasitado", [1, 2, 3],
                                format_func=lambda x: {1:"Sí",2:"No",3:"No sé"}[x])
        esteril  = st.selectbox("Esterilizado", [1, 2, 3],
                                format_func=lambda x: {1:"Sí",2:"No",3:"No sé"}[x])
        salud    = st.selectbox("Estado de salud", [1, 2, 3],
                                format_func=lambda x: {1:"Saludable",
                                                        2:"Lesión menor",
                                                        3:"Lesión grave"}[x])

    with col3:
        st.markdown("**Anuncio**")
        tarifa   = st.number_input("Tarifa de adopción (MYR)", min_value=0, max_value=500, value=0)
        fotos    = st.number_input("Número de fotos", min_value=0, max_value=20, value=3)
        videos   = st.number_input("Número de vídeos", min_value=0, max_value=10, value=0)
        nombre   = st.selectbox("¿Tiene nombre?", [1, 0],
                                format_func=lambda x: "Sí" if x else "No")
        descrip  = st.selectbox("¿Tiene descripción?", [1, 0],
                                format_func=lambda x: "Sí" if x else "No")

    st.divider()
    st.markdown('<p class="section-title">🎯 Resultado real de adopción</p>',
                unsafe_allow_html=True)
    st.info("Este campo es el más importante: indica cuándo fue adoptada realmente la mascota.")

    adoption_speed = st.selectbox(
        "Velocidad de adopción real",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: f"{x} — {SPEED_LABELS[x]}",
    )

    submitted = st.form_submit_button(
        "💾 Guardar datos de esta mascota",
        type="primary",
        use_container_width=True,
    )

if submitted:
    datos = {
        "Type":            tipo,
        "Age":             edad,
        "Gender":          genero,
        "MaturitySize":    tamaño,
        "FurLength":       pelo,
        "Vaccinated":      vacuna,
        "Dewormed":        despar,
        "Sterilized":      esteril,
        "Health":          salud,
        "Quantity":        cantidad,
        "Fee":             tarifa,
        "PhotoAmt":        fotos,
        "VideoAmt":        videos,
        "has_name":        nombre,
        "has_description": descrip,
        "is_free":         int(tarifa == 0),
        "AdoptionSpeed":   adoption_speed,
    }
    insertar_mascota(conn, datos)
    st.success(f"✅ Mascota guardada correctamente — AdoptionSpeed: {SPEED_LABELS[adoption_speed]}")

# ── Tabla de datos recogidos ───────────────────────────────────────────────────
st.divider()
st.markdown('<p class="section-title">📊 Datos recogidos hasta ahora</p>',
            unsafe_allow_html=True)

df_new = pd.read_sql_query("SELECT * FROM new_pets ORDER BY id DESC", conn)

if df_new.empty:
    st.info("Aún no hay datos registrados. Usa el formulario para añadir mascotas.")
else:
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total registros", len(df_new))
    col_b.metric("Perros", (df_new["Type"] == 1).sum())
    col_c.metric("Gatos",  (df_new["Type"] == 2).sum())

    df_new["AdoptionSpeed"] = df_new["AdoptionSpeed"].map(SPEED_LABELS)
    st.dataframe(
        df_new[["ts","Type","Age","Gender","Fee","PhotoAmt",
                "Vaccinated","AdoptionSpeed"]].rename(columns={
            "ts": "Fecha",
            "Type": "Tipo",
            "Age": "Edad",
            "Gender": "Género",
            "Fee": "Tarifa",
            "PhotoAmt": "Fotos",
            "Vaccinated": "Vacunado",
            "AdoptionSpeed": "Adopción",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Exportar para reentrenamiento ──────────────────────────────────────────
    st.divider()
    st.markdown('<p class="section-title">⬇️ Exportar para reentrenamiento</p>',
                unsafe_allow_html=True)
    st.markdown(
        "Descarga los nuevos datos en CSV para combinarlos con el dataset original "
        "y reentrenar el modelo con información más reciente."
    )

    df_export = pd.read_sql_query("SELECT * FROM new_pets", conn)
    csv = df_export.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar new_data.csv",
        data=csv,
        file_name="new_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(
        "💡 Para reentrenar: combina `data/train_clean.csv` con `new_data.csv` "
        "y ejecuta `python src/train.py --data data/train_combined.csv`"
    )
