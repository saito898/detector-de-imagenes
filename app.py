from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

# ============================================================
# CONFIGURACIÓN
# ============================================================

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# ESTILOS
# ============================================================

st.markdown("""
<style>

/* ============================================================
   FUENTE
   ============================================================ */

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ============================================================
   FONDO
   ============================================================ */

.stApp {
    background: #f7f4fc;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ============================================================
   OCULTAR ELEMENTOS
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
        #4d2380 55%,
        #622c91 100%
    );
}

section[data-testid="stSidebar"] > div {
    padding-top: 2rem;
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: white !important;
}

section[data-testid="stSidebar"] label {
    color: #eee7fa !important;
}

/* ============================================================
   HERO
   ============================================================ */

.hero-container {
    background: linear-gradient(
        135deg,
        #32145f 0%,
        #5b2590 50%,
        #8246b8 100%
    );

    border-radius: 26px;

    padding: 38px 44px;

    margin-bottom: 25px;

    box-shadow:
        0 18px 40px rgba(74, 37, 120, 0.20);
}

.hero-container h1 {
    color: white !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin: 5px 0 10px 0 !important;
}

.hero-container p {
    color: #eee7fa !important;
    font-size: 16px !important;
    line-height: 1.6 !important;
    margin: 0 !important;
}

.hero-label {
    color: #dfcff3 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase;
}

/* ============================================================
   TÍTULOS
   ============================================================ */

h1, h2, h3 {
    color: #49226f !important;
    font-weight: 800 !important;
}

h2 {
    margin-top: 25px !important;
}

/* ============================================================
   TEXTO NORMAL
   ============================================================ */

p {
    color: #514b5c;
}

.stCaption {
    color: #82788e !important;
}

/* ============================================================
   TARJETAS
   ============================================================ */

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: white;
    border-radius: 20px;
    border: 1px solid #e6dcef;
}

/* ============================================================
   CÁMARA
   ============================================================ */

div[data-testid="stCameraInput"] {
    background: white;
    border: 1px solid #e6dcef;
    border-radius: 20px;
    padding: 10px;
    box-shadow: 0 7px 22px rgba(66, 42, 95, 0.07);
}

/* ============================================================
   FILE / INPUTS
   ============================================================ */

input,
textarea {
    border-radius: 12px !important;
}

/* ============================================================
   BOTONES
   ============================================================ */

.stButton > button {
    border: none !important;
    border-radius: 13px !important;

    background: linear-gradient(
        135deg,
        #65359b,
        #8246b8
    ) !important;

    color: white !important;

    font-weight: 700 !important;

    padding: 11px 20px !important;

    box-shadow:
        0 8px 18px rgba(91, 48, 130, 0.20);

    transition: all 0.2s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 12px 25px rgba(91, 48, 130, 0.28);
}

/* ============================================================
   SLIDERS
   ============================================================ */

section[data-testid="stSidebar"] [data-testid="stSlider"] {
    margin-bottom: 20px;
}

/* ============================================================
   TABLA
   ============================================================ */

[data-testid="stDataFrame"] {
    border-radius: 14px;
    overflow: hidden;
}

/* ============================================================
   EXPANDERS
   ============================================================ */

div[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #e6dcef !important;
    border-radius: 16px !important;
}

/* ============================================================
   ALERTAS
   ============================================================ */

div[data-testid="stAlert"] {
    border-radius: 14px !important;
}

/* ============================================================
   GRÁFICAS
   ============================================================ */

[data-testid="stVegaLiteChart"] {
    background: white;
    border-radius: 15px;
    padding: 8px;
}

/* ============================================================
   DIVISOR
   ============================================================ */

hr {
    border: none !important;
    border-top: 1px solid #e3d9ed !important;
    margin: 35px 0 !important;
}

/* ============================================================
   FOOTER
   ============================================================ */

.footer-text {
    text-align: center;
    color: #91869e;
    font-size: 13px;
    padding: 15px 0;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown("""
<div class="hero-container">

    <div class="hero-label">
        ✦ VISIÓN ARTIFICIAL
    </div>

    <h1>
        🔍 Detección de Objetos
    </h1>

    <p>
        Captura una imagen con tu cámara y descubre automáticamente
        los objetos presentes utilizando un modelo YOLOv5.
    </p>

</div>
""", unsafe_allow_html=True)


# ============================================================
# DESCRIPCIÓN
# ============================================================

st.info(
    "📌 Captura una imagen utilizando tu cámara. "
    "El modelo de inteligencia artificial analizará la imagen "
    "y señalará los objetos detectados junto con su categoría "
    "y nivel de confianza."
)


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
# APLICACIÓN
# ============================================================

if model:

    # ========================================================
    # SIDEBAR
    # ========================================================

    with st.sidebar:

        st.title("⚙️ Parámetros")

        st.caption(
            "Ajusta la configuración utilizada "
            "por el detector de objetos."
        )

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

    st.subheader("📷 Capturar imagen")

    st.caption(
        "Utiliza la cámara para tomar una fotografía "
        "que será analizada por el modelo."
    )

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

        st.markdown("---")

        st.subheader("📊 Resultados de la detección")

        col1, col2 = st.columns(
            [1.15, 0.85],
            gap="large"
        )

        # ====================================================
        # IMAGEN
        # ====================================================

        with col1:

            st.markdown("### 🖼️ Imagen con detecciones")

            st.caption(
                "Los objetos encontrados están marcados "
                "directamente sobre la imagen."
            )

            st.image(
                annotated_rgb,
                use_container_width=True
            )

        # ====================================================
        # OBJETOS
        # ====================================================

        with col2:

            st.markdown("### 🎯 Objetos detectados")

            st.caption(
                "Resumen de las categorías encontradas "
                "en la imagen."
            )

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

                st.markdown("### 📈 Cantidad por categoría")

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

st.caption(
    "🤖 Acerca de la aplicación: "
    "Detección de objetos con YOLOv5 + Streamlit + PyTorch."
)

st.markdown(
    '<div class="footer-text">'
    'Hecho con 💜 usando YOLOv5, PyTorch y Streamlit'
    '</div>',
    unsafe_allow_html=True
)
