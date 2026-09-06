import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io

from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4


# ==================================================
# PAGE
# ==================================================

st.set_page_config(
    page_title="Brain Tumor Detection",
    page_icon="🧠",
    layout="wide"
)
st.markdown("""
<style>

/* ================================
   ANIMATED BACKGROUND
   ================================ */

.stApp {
    background: linear-gradient(
        -45deg,
        #071a2b,
        #0b2942,
        #123b5d,
        #071a2b
    );

    background-size: 400% 400%;

    animation: gradientBG 15s ease infinite;
}


/* Moving gradient */

@keyframes gradientBG {

    0% {
        background-position: 0% 50%;
    }

    50% {
        background-position: 100% 50%;
    }

    100% {
        background-position: 0% 50%;
    }
}


/* ================================
   FLOATING ANIMATION
   ================================ */

.stApp::before {

    content: "";

    position: fixed;

    width: 250px;
    height: 250px;

    border-radius: 50%;

    background: rgba(0, 200, 255, 0.12);

    top: 10%;
    left: 5%;

    filter: blur(5px);

    animation: floating1 8s ease-in-out infinite;

    z-index: 0;
}


.stApp::after {

    content: "";

    position: fixed;

    width: 350px;
    height: 350px;

    border-radius: 50%;

    background: rgba(0, 150, 255, 0.10);

    bottom: 5%;
    right: 5%;

    filter: blur(5px);

    animation: floating2 10s ease-in-out infinite;

    z-index: 0;
}


/* Floating animation */

@keyframes floating1 {

    0% {
        transform: translate(0px, 0px);
    }

    50% {
        transform: translate(100px, 80px);
    }

    100% {
        transform: translate(0px, 0px);
    }
}


@keyframes floating2 {

    0% {
        transform: translate(0px, 0px);
    }

    50% {
        transform: translate(-100px, -80px);
    }

    100% {
        transform: translate(0px, 0px);
    }
}


/* ================================
   CONTENT
   ================================ */

.block-container {

    position: relative;

    z-index: 1;

    padding-top: 40px;

}


/* ================================
   TITLE
   ================================ */

h1 {

    color: white !important;

    font-size: 42px !important;

    font-weight: 800 !important;

}


/* ================================
   NORMAL TEXT
   ================================ */

p {

    color: #e5f2ff;

}


/* ================================
   UPLOAD BOX
   ================================ */

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.10);

    border: 2px dashed rgba(255,255,255,0.5);

    border-radius: 20px;

    padding: 25px;

    backdrop-filter: blur(12px);

}


/* ================================
   RESULT BOX
   ================================ */

.result-card {

    background: rgba(255,255,255,0.12);

    border-radius: 20px;

    padding: 25px;

    border: 1px solid rgba(255,255,255,0.25);

    backdrop-filter: blur(15px);

}


/* ================================
   BUTTON
   ================================ */

.stDownloadButton button {

    border-radius: 12px;

    font-weight: bold;

    padding: 12px 25px;

}


/* ================================
   SIDEBAR
   ================================ */

section[data-testid="stSidebar"] {

    background: rgba(5,20,35,0.95);

}


/* ================================
   IMAGE
   ================================ */

img {

    border-radius: 15px;

}

</style>
""", unsafe_allow_html=True)
st.title("🧠 Brain Tumor Detection")
st.write("AI-Based Brain MRI Image Classification")

st.info(
    "Upload a brain MRI image. The CNN model will classify "
    "the image and show an AI-highlighted region."
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_my_model():
    model_url = "https://huggingface.co/Priya-ai16/brain_tumor_model/resolve/main/brain_tumor_model.keras"

    model_path = tf.keras.utils.get_file(
        "brain_tumor_model.keras",
        model_url
    )

    return tf.keras.models.load_model(model_path)


model = load_my_model()


# ==================================================
# CLASSES
# ==================================================

class_names = [
    "glioma",
    "meningioma",
    "notumor",
    "pituitary"
]


# ==================================================
# INFORMATION
# ==================================================

information = {
    "glioma":
        "Glioma is a type of tumor that develops from glial cells.",

    "meningioma":
        "Meningioma develops from the membranes surrounding the brain.",

    "pituitary":
        "Pituitary tumor develops in or around the pituitary gland.",

    "notumor":
        "The model classified this image as the no-tumor class."
}


# ==================================================
# FIND LAST CONVOLUTION LAYER
# ==================================================

def get_last_conv_layer(model):

    for layer in reversed(model.layers):

        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer

    return None


# ==================================================
# GRAD CAM
# ==================================================

def grad_cam(model, image, class_index):

    last_conv = get_last_conv_layer(model)

    if last_conv is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[
            last_conv.output,
            model(model.inputs)
        ]
    )

    with tf.GradientTape() as tape:

        conv_output, predictions = grad_model(
            image
        )

        class_score = predictions[:, class_index]

    gradients = tape.gradient(
        class_score,
        conv_output
    )

    pooled_gradients = tf.reduce_mean(
        gradients,
        axis=(0, 1, 2)
    )

    conv_output = conv_output[0]

    heatmap = tf.reduce_sum(
        conv_output * pooled_gradients,
        axis=-1
    )

    heatmap = tf.maximum(
        heatmap,
        0
    )

    maximum = tf.reduce_max(
        heatmap
    )

    if maximum > 0:
        heatmap = heatmap / maximum

    return heatmap.numpy()


# ==================================================
# CREATE HIGHLIGHT IMAGE
# ==================================================

def create_highlight(original, heatmap):

    original_array = np.array(original)

    h = original_array.shape[0]
    w = original_array.shape[1]

    heatmap_img = Image.fromarray(
        np.uint8(heatmap * 255)
    )

    heatmap_img = heatmap_img.resize(
        (w, h)
    )

    heatmap_array = np.array(
        heatmap_img
    ) / 255.0

    # Find strongest AI region
    y, x = np.unravel_index(
        np.argmax(heatmap_array),
        heatmap_array.shape
    )

    fig, ax = plt.subplots(
        figsize=(6, 6)
    )

    ax.imshow(
        original_array
    )

    ax.imshow(
        heatmap_array,
        cmap="jet",
        alpha=0.45
    )

    # Red circle
    circle = plt.Circle(
        (x, y),
        40,
        fill=False,
        linewidth=4,
        edgecolor="red"
    )

    ax.add_patch(circle)

    ax.text(
        x,
        y - 50,
        "AI Highlighted Region",
        color="red",
        fontsize=11,
        fontweight="bold",
        ha="center"
    )

    ax.axis("off")

    buffer = io.BytesIO()

    plt.savefig(
        buffer,
        format="png",
        bbox_inches="tight"
    )

    plt.close()

    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


# ==================================================
# UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)


# ==================================================
# PROCESS IMAGE
# ==================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    st.subheader("📷 Uploaded MRI")

    st.image(
        image,
        width=400
    )

    # Resize
    resized = image.resize(
        (224, 224)
    )

    image_array = np.array(
        resized
    ) / 255.0

    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # Prediction
    prediction = model.predict(
        image_array,
        verbose=0
    )

    index = int(
        np.argmax(prediction)
    )

    predicted_class = class_names[index]

    confidence = float(
        prediction[0][index]
    ) * 100


    # ==================================================
    # RESULT
    # ==================================================

    st.subheader("🔍 Prediction Result")

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"Predicted Class: {predicted_class.upper()}"
        )

    with col2:

        st.info(
            f"Confidence: {confidence:.2f}%"
        )


    # ==================================================
    # GRAD CAM
    # ==================================================

    try:

        heatmap = grad_cam(
            model,
            image_array,
            index
        )

        if heatmap is not None:

            highlighted = create_highlight(
                image,
                heatmap
            )

            st.subheader(
                "🔴 AI Highlighted Region"
            )

            st.image(
                highlighted,
                width=500
            )

            st.caption(
                "The red circle shows the region that "
                "most influenced the AI prediction."
            )

    except Exception as e:

        st.warning(
            "AI highlighting could not be generated."
        )

        st.write(
            "Prediction is still available."
        )


    # ==================================================
    # INFORMATION
    # ==================================================

    st.subheader(
        f"ℹ️ About {predicted_class.upper()}"
    )

    st.write(
        information[predicted_class]
    )


    # ==================================================
    # PDF REPORT
    # ==================================================

    pdf_buffer = io.BytesIO()

    pdf = canvas.Canvas(
        pdf_buffer,
        pagesize=A4
    )

    width, height = A4

    pdf.setFont(
        "Helvetica-Bold",
        20
    )

    pdf.drawString(
        50,
        height - 60,
        "Brain Tumor Detection Report"
    )

    pdf.setFont(
        "Helvetica",
        12
    )

    pdf.drawString(
        50,
        height - 100,
        f"Prediction: {predicted_class.upper()}"
    )

    pdf.drawString(
        50,
        height - 125,
        f"Confidence: {confidence:.2f}%"
    )

    pdf.drawString(
        50,
        height - 160,
        "Information:"
    )

    # Split information into lines
    text = information[predicted_class]

    pdf.drawString(
        50,
        height - 180,
        text[:100]
    )

    # Save uploaded MRI
    original_buffer = io.BytesIO()

    image.save(
        original_buffer,
        format="PNG"
    )

    original_buffer.seek(0)

    # Temporary image file
    with open(
        "mri_report.png",
        "wb"
    ) as f:

        f.write(
            original_buffer.getvalue()
        )

    pdf.drawImage(
        "mri_report.png",
        50,
        height - 520,
        width=300,
        height=300
    )

    pdf.setFont(
        "Helvetica",
        9
    )

    pdf.drawString(
        50,
        50,
        "This AI result is for educational/research purposes only."
    )

    pdf.save()

    pdf_buffer.seek(0)


    # ==================================================
    # DOWNLOAD
    # ==================================================

    st.subheader(
        "📄 Download Report"
    )

    st.download_button(
        "⬇️ Download PDF Report",
        data=pdf_buffer,
        file_name="Brain_Tumor_Report.pdf",
        mime="application/pdf"
    )