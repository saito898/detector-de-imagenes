from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# ESTILOS VISUALES
# ============================================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ============================================================
   FONDO GENERAL
   ============================================================ */

.stApp {
    background: #f7f4fc;
}

.main {
    background: #f7f4fc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ============================================================
   OCULTAR ELEMENTOS NATIVOS
   ============================================================ */

#MainMenu {
    visibility: hidden;
}

footer {
    visibility: hidden;
}

header[data-testid="stHeader"] {
    background: transparent;
}

/* ============================================================
   SIDEBAR
   ============================================================ */

section[data-testid="stSidebar"] {
    background: linear-gradient(
        180deg,
        #32145f 0%,
        #4d2380 50%,
        #622c91 100%
    );
}

section[data-testid="stSidebar"] > div {
    padding: 2rem 1.25rem;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

.sidebar-title {
    font-size: 23px;
    font-weight: 800;
    margin-bottom: 5px;
}

.sidebar-subtitle {
    color: #dfd2f3 !important;
    font-size: 13px;
    line-height: 1.5;
    margin-bottom: 25px;
}

/* ============================================================
   SLIDERS
   ============================================================ */

section[data-testid="stSidebar"] [data-testid="stSlider"] {
    margin-bottom: 22px;
}

section[data-testid="stSidebar"] [data-testid="stSlider"] label {
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] [data-testid="stSlider"] div[role="slider"] {
    background-color: #c9a7f4 !important;
}

/* ============================================================
   NUMBER INPUT
   ============================================================ */

section[data-testid="stSidebar"] input {
    background: rgba(255,255,255,0.10) !important;
    color: white !important;
    border: 1px solid rgba(255,255,255,0.25) !important;
    border-radius: 10px !important;
}

/* ============================================================
   HERO
   ============================================================ */

.hero {
    position: relative;
    overflow: hidden;

    background: linear-gradient(
        135deg,
        #32145f 0%,
        #5b2590 48%,
        #8246b8 100%
    );

    border-radius: 26px;
    padding: 42px 45px;
    margin-bottom: 25px;

    box-shadow:
        0 18px 40px rgba(74, 37, 120, 0.20);
}

.hero::before {
    content: "";
    position: absolute;

    width: 280px;
    height: 280px;

    border-radius: 50%;

    background: rgba(255,255,255,0.08);

    right: -80px;
    top: -120px;
}

.hero::after {
    content: "";

    position: absolute;

    width: 190px;
    height: 190px;

    border-radius: 50%;

    background: rgba(255,255,255,0.06);

    right: 130px;
    bottom: -120px;
}

.hero-content {
    position: relative;
    z-index: 2;
}

.hero-label {
    color: #e8dcfa;

    font-size: 12px;
    font-weight: 700;

    letter-spacing: 2px;
    text-transform: uppercase;

    margin-bottom: 10px;
}

.hero-title {
    color: white;

    font-size: 42px;
    font-weight: 800;

    line-height: 1.1;

    margin: 0;
}

.hero-description {
    color: #eee7fa;

    font-size: 16px;
    line-height: 1.6;

    max-width: 720px;

    margin-top: 14px;
}

/* ============================================================
   INTRO
   ============================================================ */

.intro {
    background: white;

    border: 1px solid #e8def4;

    border-radius: 18px;

    padding: 18px 22px;

    margin-bottom: 28px;

    color: #6b6379;

    font-size: 14px;
    line-height: 1.6;

    box-shadow:
        0 5px 18px rgba(66, 42, 95, 0.05);
}

/* ============================================================
   TÍTULOS
   ============================================================ */

h1,
h2,
h3 {
    color: #49226f !important;
    font-weight: 800 !important;
}

/* ============================================================
   CÁMARA
   ============================================================ */

div[data-testid="stCameraInput"] {
    background: white;

    border: 1px solid #e6dcef;

    border-radius: 20px;

    padding: 12px;

    box-shadow:
        0 7px 22px rgba(66, 42, 95, 0.07);
}

/* ============================================================
   BOTONES
   ============================================================ */

button {
    border-radius: 12px !important;
}

/* ============================================================
   ALERTAS
   ============================================================ */

div[data-testid="stAlert"] {
    border-radius: 14px !important;
}

/* ============================================================
   TARJETAS
   ============================================================ */

.card {
    background: white;

    border: 1px solid #e6dcef;

    border-radius: 20px;

    padding: 24px;

    box-shadow:
        0 7px 22px rgba(66, 42, 95, 0.06);

    height: 100%;
}

.card-title {
    color: #49226f;

    font-size: 19px;
    font-weight: 800;

    margin-bottom: 6px;
}

.card-description {
    color: #81778f;

    font-size: 13px;

    margin-bottom: 18px;
}

/* ============================================================
   TARJETA DE ESTADO
   ============================================================ */

.status-card {
    background: linear-gradient(
        135deg,
        #f2eafa,
        #faf8fd
    );

    border: 1px solid #e4d7f0;

    border-radius: 17px;

    padding: 17px 20px;

    margin-bottom: 20px;
}

.status-title {
    color: #512879;

    font-weight: 800;

    font-size: 14px;

    margin-bottom: 5px;
}

.status-text {
    color: #766c82;

    font-size: 13px;
}

/* ============================================================
   TABLA
   ============================================================ */

[data-testid="stDataFrame"] {
    border: 1px solid #e4d9ef;

    border-radius: 14px;

    overflow: hidden;
}

/* ============================================================
   GRÁFICA
   ============================================================ */

[data-testid="stVegaLiteChart"] {
    background: white;

    border-radius: 15px;
}

/* ============================================================
   DIVISOR
   ============================================================ */

hr {
    border: none !important;

    border-top: 1px solid #e4dced !important;

    margin: 35px 0 !important;
}

/* ============================================================
   FOOTER
   ============================================================ */

.footer-custom {
    text-align: center;

    color: #91869e;

    font-size: 13px;

    padding: 20px 0 5px;
}

/* ============================================================
   RESPONSIVE
   ============================================================ */

@media (max-width: 768px) {

    .hero {
        padding: 30px 25px;
    }

    .hero-title {
        font-size: 31px;
    }

    .hero-description {
        font-size: 14px;
    }

    .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown("""
<div class="hero">

    <div class="hero-content">

        <div class="hero-label">
            ✦ VISIÓN ARTIFICIAL
        </div>

        <div class="hero-title">
            🔍 Detección de Objetos
        </div>

        <div class="hero-description">
            Captura una imagen con tu cámara y descubre automáticamente
            los objetos presentes utilizando un modelo YOLOv5.
        </div>

    </div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# INTRODUCCIÓN
# ============================================================

st.markdown("""
<div class="intro">

    <b>¿Cómo funciona?</b><br>

    Captura una imagen utilizando tu cámara. El modelo de inteligencia
    artificial analizará la imagen y señalará los objetos detectados,
    junto con su categoría y nivel de confianza.

</div>
""", unsafe_allow_html=True)


# ============================================================
# CARGAR MODELO
# ============================================================

@st.cache_resource
def load_model():

    try:

        from ultralytics import YOLO

        model = YOLO("yolov5su.pt")

        return model

    except Exception as e:

        st.error(
            f"❌ Error al cargar el modelo: {str(e)}"
        )

        return None


with st.spinner("Cargando modelo YOLOv5..."):

    model = load_model()


# ============================================================
# CONFIGURACIÓN
# ============================================================

if model:

    with st.sidebar:

        st.markdown("""
        <div class="sidebar-title">
            ⚙️ Parámetros
        </div>

        <div class="sidebar-subtitle">
            Ajusta la configuración utilizada
            por el detector de objetos.
        </div>
        """, unsafe_allow_html=True)

        st.subheader("Configuración de detección")

        conf_threshold = st.slider(
            "Confianza mínima",
            0.0,
            1.0,
            0.25,
            0.01
        )

        iou_threshold = st.slider(
            "Umbral IoU",
            0.0,
            1.0,
            0.45,
            0.01
        )

        max_det = st.number_input(
            "Detecciones máximas",
            10,
            2000,
            1000,
            10
        )


    # ========================================================
    # CÁMARA
    # ========================================================

    st.markdown("## 📷 Captura una imagen")

    st.markdown("""
    <div class="card-description">
        Utiliza la cámara para tomar una fotografía que será
        analizada por el modelo de detección.
    </div>
    """, unsafe_allow_html=True)

    picture = st.camera_input(
        "Capturar imagen",
        key="camera"
    )


    # ========================================================
    # PROCESAMIENTO
    # ========================================================

    if picture:

        bytes_data = picture.getvalue()

        # Decodificar con Pillow
        pil_img = Image.open(
            io.BytesIO(bytes_data)
        ).convert("RGB")

        np_img = np.array(
            pil_img
        )[..., ::-1]

        # ====================================================
        # DETECCIÓN
        # ====================================================

        with st.spinner("Detectando objetos..."):

            try:

                results = model(
                    np_img,
                    conf=conf_threshold,
                    iou=iou_threshold,
                    max_det=int(max_det)
                )

            except Exception as e:

                st.error(
                    f"Error durante la detección: {str(e)}"
                )

                st.stop()


        result = results[0]

        boxes = result.boxes

        annotated = result.plot()

        annotated_rgb = annotated[:, :, ::-1]


        # ====================================================
        # RESULTADOS
        # ====================================================

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("## 📊 Resultados de la detección")


        col1, col2 = st.columns(
            [1.15, 0.85],
            gap="large"
        )


        # ====================================================
        # IMAGEN
        # ====================================================

        with col1:

            st.markdown("""
            <div class="card">

                <div class="card-title">
                    🖼️ Imagen con detecciones
                </div>

                <div class="card-description">
                    El modelo ha marcado los objetos encontrados
                    directamente sobre la imagen.
                </div>

            </div>
            """, unsafe_allow_html=True)

            st.image(
                annotated_rgb,
                use_container_width=True
            )


        # ====================================================
        # OBJETOS DETECTADOS
        # ====================================================

        with col2:

            st.markdown("""
            <div class="card">

                <div class="card-title">
                    🎯 Objetos detectados
                </div>

                <div class="card-description">
                    Resumen de las categorías encontradas
                    en la imagen.
                </div>

            </div>
            """, unsafe_allow_html=True)


            if boxes is not None and len(boxes) > 0:

                label_names = model.names

                category_count = {}

                category_conf = {}


                for box in boxes:

                    cat = int(
                        box.cls.item()
                    )

                    conf = float(
                        box.conf.item()
                    )

                    category_count[cat] = (
                        category_count.get(cat, 0) + 1
                    )

                    category_conf.setdefault(
                        cat,
                        []
                    ).append(conf)


                data = [

                    {
                        "Categoría":
                            label_names[cat],

                        "Cantidad":
                            count,

                        "Confianza promedio":
                            f"{np.mean(category_conf[cat]):.2f}"
                    }

                    for cat, count
                    in category_count.items()

                ]


                df = pd.DataFrame(data)


                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )


                st.markdown(
                    "<br>",
                    unsafe_allow_html=True
                )


                st.markdown("""
                <div class="card-title">
                    📈 Cantidad por categoría
                </div>
                """, unsafe_allow_html=True)


                st.bar_chart(
                    df.set_index(
                        "Categoría"
                    )["Cantidad"]
                )


            else:

                st.info(
                    "No se detectaron objetos con "
                    "los parámetros actuales."
                )

                st.caption(
                    "Prueba a reducir el umbral de "
                    "confianza en la barra lateral."
                )


# ============================================================
# ERROR DE MODELO
# ============================================================

else:

    st.error(
        "No se pudo cargar el modelo. "
        "Verifica las dependencias e inténtalo nuevamente."
    )

    st.stop()


# ============================================================
# INFORMACIÓN FINAL
# ============================================================

st.markdown("---")

st.markdown("""
<div class="status-card">

    <div class="status-title">
        🤖 Acerca de la aplicación
    </div>

    <div class="status-text">
        Detección de objetos mediante YOLOv5 +
        Streamlit + PyTorch.
    </div>

</div>
""", unsafe_allow_html=True)


st.markdown("""
<div class="footer-custom">
    Hecho con 💜 usando YOLOv5, PyTorch y Streamlit
</div>
""", unsafe_allow_html=True)
