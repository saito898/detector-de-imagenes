from PIL import Image
import io
import streamlit as st
import numpy as np
import pandas as pd
import torch

# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# =========================================================
# ESTILOS
# =========================================================

st.markdown("""
<style>

/* -------------------- FONDO GENERAL -------------------- */

.stApp {
    background: #f7f4fc;
}

/* -------------------- SIDEBAR -------------------- */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #32145f 0%, #4d2380 100%);
}

section[data-testid="stSidebar"] * {
    color: white !important;
}

section[data-testid="stSidebar"] .stSlider > div > div > div {
    color: white !important;
}

/* -------------------- TITULOS -------------------- */

h1 {
    color: #49226f !important;
    font-size: 42px !important;
    font-weight: 800 !important;
    margin-bottom: 5px !important;
}

h2 {
    color: #49226f !important;
    font-weight: 750 !important;
}

h3 {
    color: #5b2d82 !important;
    font-weight: 700 !important;
}

/* -------------------- TEXTO -------------------- */

p {
    color: #5f5570;
}

[data-testid="stCaptionContainer"] {
    color: #756a82;
}

/* -------------------- LINEAS -------------------- */

hr {
    border: none;
    height: 1px;
    background: #e3d9ed;
    margin: 25px 0;
}

/* -------------------- BOTON CAMARA -------------------- */

button {
    border-radius: 10px !important;
}

/* -------------------- DATAFRAME -------------------- */

[data-testid="stDataFrame"] {
    border: 1px solid #e4d9ed;
    border-radius: 14px;
    overflow: hidden;
}

/* -------------------- METRICAS / INFORMACION -------------------- */

[data-testid="stAlert"] {
    border-radius: 14px;
}

/* -------------------- CONTENEDORES -------------------- */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 18px;
    border-color: #e3d9ed;
    background: white;
}

/* -------------------- BOTON PRINCIPAL -------------------- */

.stButton > button {
    background: #5b2590;
    color: white;
    border: none;
    border-radius: 10px;
}

.stButton > button:hover {
    background: #7138a6;
    color: white;
}

/* -------------------- ESPACIADO -------------------- */

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* -------------------- FILE UPLOADER / CAMERA -------------------- */

[data-testid="stCameraInput"] {
    background: white;
    border-radius: 18px;
    padding: 10px;
}

/* -------------------- GRAFICA -------------------- */

[data-testid="stVegaLiteChart"] {
    background: white;
    border-radius: 18px;
    padding: 10px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# MODELO
# =========================================================

@st.cache_resource
def load_model():
    try:
        from ultralytics import YOLO
        model = YOLO("yolov5su.pt")
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        return None


# =========================================================
# ENCABEZADO
# =========================================================

st.title("🔍 Detección de Objetos en Tiempo Real")

st.caption(
    "Captura una imagen con tu cámara y utiliza inteligencia artificial "
    "para identificar los objetos presentes."
)

st.write("")


# =========================================================
# CARGAR MODELO
# =========================================================

with st.spinner("Cargando modelo YOLOv5..."):
    model = load_model()


# =========================================================
# SI EL MODELO CARGÓ CORRECTAMENTE
# =========================================================

if model:

    # -----------------------------------------------------
    # SIDEBAR
    # -----------------------------------------------------

    with st.sidebar:

        st.title("⚙️ Parámetros")

        st.subheader("Configuración de detección")

        st.write("Ajusta los valores según el nivel de precisión que necesites.")

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

        st.divider()

        st.caption(
            "Los valores más altos de confianza hacen que el modelo "
            "sea más estricto al detectar objetos."
        )


    # -----------------------------------------------------
    # CAMARA
    # -----------------------------------------------------

    st.subheader("📷 Captura una imagen")

    st.write(
        "Utiliza la cámara para tomar una fotografía que será analizada "
        "automáticamente."
    )

    picture = st.camera_input(
        "Capturar imagen",
        key="camera"
    )


    # -----------------------------------------------------
    # PROCESAMIENTO
    # -----------------------------------------------------

    if picture:

        bytes_data = picture.getvalue()

        pil_img = Image.open(
            io.BytesIO(bytes_data)
        ).convert("RGB")

        np_img = np.array(pil_img)[..., ::-1]


        # -------------------------------------------------
        # DETECCIÓN
        # -------------------------------------------------

        with st.spinner("🔎 Detectando objetos..."):

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


        # -------------------------------------------------
        # RESULTADOS
        # -------------------------------------------------

        st.divider()

        col1, col2 = st.columns(
            [1.4, 1],
            gap="large"
        )


        # -------------------------------------------------
        # IMAGEN
        # -------------------------------------------------

        with col1:

            st.subheader("🖼️ Imagen con detecciones")

            st.image(
                annotated_rgb,
                use_container_width=True
            )


        # -------------------------------------------------
        # OBJETOS DETECTADOS
        # -------------------------------------------------

        with col2:

            st.subheader("📊 Objetos detectados")

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
                        "Categoría": label_names[cat],
                        "Cantidad": count,
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


                st.write("")

                st.subheader("Cantidad por categoría")

                st.bar_chart(
                    df.set_index("Categoría")["Cantidad"]
                )


            else:

                st.info(
                    "No se detectaron objetos con los parámetros actuales."
                )

                st.caption(
                    "Prueba a reducir el umbral de confianza "
                    "en la barra lateral."
                )


# =========================================================
# ERROR DEL MODELO
# =========================================================

else:

    st.error(
        "No se pudo cargar el modelo. "
        "Verifica las dependencias e inténtalo nuevamente."
    )

    st.stop()


# =========================================================
# PIE DE PÁGINA
# =========================================================

st.divider()

st.caption(
    "🔍 Detección de objetos con YOLOv5 + Streamlit + PyTorch."
)
