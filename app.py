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


# ==================================================
# ANIMATED BACKGROUND
# ==================================================

st.markdown("""
<style>

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

.block-container {

    position: relative;
    z-index: 1;
    padding-top: 40px;

}

h1 {

    color: white !important;
    font-size: 42px !important;
    font-weight: 800 !important;

}

p {
    color: #e5f2ff;
}

[data-testid="stFileUploader"] {

    background: rgba(255,255,255,0.10);

    border: 2px dashed rgba(255,255,255,0.5);

    border-radius: 20px;

    padding: 25px;

    backdrop-filter: blur(12px);

}

.stDownloadButton button {

    border-radius: 12px;

    font-weight: bold;

    padding: 12px 25px;

}

section[data-testid="stSidebar"] {

    background: rgba(5,20,35,0.95);

}

img {
    border-radius: 15px;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# TITLE
# ==================================================

st.title("🧠 Brain Tumor Detection")

st.write(
    "AI-Based Brain MRI Image Classification"
)

st.info(
    "Upload a brain MRI image. The CNN model will "
    "classify the image and show the prediction "
    "probabilities and AI-highlighted region."
)


# ==================================================
# LOAD MODEL
# ==================================================

@st.cache_resource
def load_my_model():

    return tf.keras.models.load_model(
        "model/brain_tumor_model.keras"
    )


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
# RECOMMENDED NEXT STEP
# ==================================================

guidance = {

    "glioma":
        "Consult a qualified doctor or radiologist for professional "
        "evaluation and confirmation of the MRI result.",

    "meningioma":
        "Consult a qualified doctor or radiologist for professional "
        "evaluation and confirmation of the MRI result.",

    "pituitary":
        "Consult a qualified doctor or radiologist for professional "
        "evaluation and confirmation of the MRI result.",

    "notumor":
        "The model predicts the no-tumor class. If symptoms or "
        "clinical concerns exist, consult a qualified doctor "
        "or radiologist for professional evaluation."
}


# ==================================================
# GRAD CAM
# ==================================================

def grad_cam(model, image, class_index):

    input_layer = tf.keras.Input(
        shape=(224, 224, 3)
    )

    x = input_layer

    last_conv_output = None

    for layer in model.layers:

        if isinstance(
            layer,
            tf.keras.layers.InputLayer
        ):
            continue

        x = layer(x)

        if isinstance(
            layer,
            tf.keras.layers.Conv2D
        ):
            last_conv_output = x

    if last_conv_output is None:
        return None

    grad_model = tf.keras.models.Model(
        inputs=input_layer,
        outputs=[
            last_conv_output,
            x
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

    original_array = np.array(
        original
    )

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

    circle = plt.Circle(
        (x, y),
        40,
        fill=False,
        linewidth=4,
        edgecolor="red"
    )

    ax.add_patch(
        circle
    )

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

    return Image.open(
        buffer
    ).convert("RGB")


# ==================================================
# UPLOAD
# ==================================================

uploaded_file = st.file_uploader(
    "📤 Upload MRI Image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


# ==================================================
# PROCESS IMAGE
# ==================================================

if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")


    # ==================================================
    # ORIGINAL MRI
    # ==================================================

    st.subheader(
        "📷 Uploaded MRI"
    )

    st.image(
        image,
        width=400
    )


    # ==================================================
    # PREPROCESS
    # ==================================================

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


    # ==================================================
    # PREDICTION
    # ==================================================

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
    # RISK STATUS
    # ==================================================

    st.subheader(
        "🚨 AI Result Status"
    )

    if predicted_class == "notumor":

        st.success(
            f"✅ AI Result: No Tumor Class\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    else:

        st.error(
            f"⚠️ AI Result: {predicted_class.upper()} Class\n\n"
            f"Confidence: {confidence:.2f}%"
        )

    st.caption(
        "This status is based only on the CNN model prediction "
        "and must not be considered a medical diagnosis."
    )


    # ==================================================
    # RESULT
    # ==================================================

    st.subheader(
        "🔍 Prediction Result"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.success(
            f"Predicted Class: "
            f"{predicted_class.upper()}"
        )

    with col2:

        st.info(
            f"Confidence: "
            f"{confidence:.2f}%"
        )


    # ==================================================
    # PROBABILITIES
    # ==================================================

    st.subheader(
        "📊 Prediction Probabilities"
    )

    for i in range(
        len(class_names)
    ):

        probability = (
            float(prediction[0][i])
            * 100
        )

        st.write(
            f"{class_names[i].upper()}: "
            f"{probability:.2f}%"
        )

        st.progress(
            min(probability / 100, 1.0)
        )


    # ==================================================
    # GRAD CAM
    # ==================================================

    highlighted = None

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
                "🧠 MRI Analysis"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    "📷 Original MRI"
                )

                st.image(
                    image,
                    width=400
                )

            with col2:

                st.write(
                    "🔴 AI Highlighted Region"
                )

                st.image(
                    highlighted,
                    width=400
                )

            st.caption(
                "The highlighted region shows the area "
                "that most influenced the AI prediction. "
                "It is not an exact tumor boundary."
            )

    except Exception:

        st.warning(
            "AI highlighting could not be generated."
        )

        st.write(
            "The prediction result is still available."
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
    # RECOMMENDED NEXT STEP
    # ==================================================

    st.subheader(
        "💡 Recommended Next Step"
    )

    st.info(
        guidance[predicted_class]
    )


    # ==================================================
    # MEDICAL DISCLAIMER
    # ==================================================

    st.warning(
        "⚠️ This application is an AI research/educational "
        "project. The prediction and highlighted region are "
        "not a medical diagnosis. Please consult a qualified "
        "doctor or radiologist for professional interpretation "
        "of MRI scans."
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
        f"Prediction: "
        f"{predicted_class.upper()}"
    )

    pdf.drawString(
        50,
        height - 125,
        f"Confidence: "
        f"{confidence:.2f}%"
    )

    pdf.drawString(
        50,
        height - 160,
        "Prediction Probabilities:"
    )

    y_position = height - 180

    for i in range(
        len(class_names)
    ):

        probability = (
            float(prediction[0][i])
            * 100
        )

        pdf.drawString(
            60,
            y_position,
            f"{class_names[i].upper()}: "
            f"{probability:.2f}%"
        )

        y_position -= 20


    # ==================================================
    # INFORMATION IN PDF
    # ==================================================

    pdf.drawString(
        50,
        y_position - 10,
        "Information:"
    )

    pdf.setFont(
        "Helvetica",
        10
    )

    pdf.drawString(
        50,
        y_position - 30,
        information[predicted_class]
    )


    # ==================================================
    # SAVE MRI
    # ==================================================

    original_buffer = io.BytesIO()

    image.save(
        original_buffer,
        format="PNG"
    )

    original_buffer.seek(0)

    with open(
        "mri_report.png",
        "wb"
    ) as f:

        f.write(
            original_buffer.getvalue()
        )


    # ==================================================
    # ADD MRI TO PDF
    # ==================================================

    pdf.drawImage(
        "mri_report.png",
        50,
        height - 600,
        width=250,
        height=250
    )


    # ==================================================
    # DISCLAIMER IN PDF
    # ==================================================

    pdf.setFont(
        "Helvetica",
        8
    )

    pdf.drawString(
        50,
        50,
        "This AI result is for educational/research "
        "purposes only and is not a medical diagnosis."
    )


    # ==================================================
    # SAVE PDF
    # ==================================================

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