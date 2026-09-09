
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

st.set_page_config(
    page_title="Waste Classification System",
    page_icon="♻️",
    layout="centered"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown("""
<style>

.stApp {
    background-color: #f4f8f4;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 850px;
}

/* Header */
.header {
    background: linear-gradient(135deg, #176b4d, #2e8b68);
    padding: 28px;
    border-radius: 16px;
    text-align: center;
    color: white;
    margin-bottom: 25px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
}

.header h1 {
    margin: 0;
    font-size: 36px;
    font-weight: 700;
}

.header p {
    margin-top: 8px;
    font-size: 16px;
    color: white !important;
}

/* Section titles */
.section-title {
    color: #176b4d !important;
    font-size: 23px;
    font-weight: 600;
    margin-top: 10px;
    margin-bottom: 10px;
}

/* Result card */
.result-card {
    background-color: white;
    border-left: 6px solid #2e8b68;
    padding: 22px;
    border-radius: 12px;
    margin-top: 15px;
    box-shadow: 0 3px 10px rgba(0,0,0,0.06);
}

.result-label {
    color: #6b7770 !important;
    font-size: 14px;
}

.result-class {
    color: #176b4d !important;
    font-size: 30px;
    font-weight: 700;
}

.confidence {
    color: #26332d !important;
    font-size: 17px;
    margin-top: 8px;
}

.confidence strong {
    color: #176b4d !important;
}

/* Upload label and text */
.stFileUploader label,
.stFileUploader p {
    color: #26332d !important;
}

/* Probability section */
.stExpander {
    background-color: white;
    border: 1px solid #d9e6dd;
    border-radius: 10px;
}

.stExpander summary {
    color: #176b4d !important;
    font-weight: 600;
}

.stExpander p,
.stExpander span,
.stExpander label,
.stExpander div {
    color: #26332d !important;
}

/* Make markdown text inside expander visible */
.stExpander [data-testid="stMarkdownContainer"] p {
    color: #26332d !important;
    opacity: 1 !important;
}

.stExpander [data-testid="stMarkdownContainer"] strong {
    color: #176b4d !important;
    opacity: 1 !important;
}

/* Buttons */
.stButton > button {
    background-color: #176b4d;
    color: white !important;
    border: none;
    border-radius: 9px;
    padding: 10px 20px;
    font-size: 16px;
    font-weight: 600;
}

.stButton > button:hover {
    background-color: #12553d;
    color: white !important;
}

/* Info message */
.stAlert p {
    color: #26332d !important;
}

/* Footer */
.footer {
    text-align: center;
    color: #718078 !important;
    font-size: 13px;
    margin-top: 35px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------
st.markdown("""
<div class="header">
    <h1>♻️ Waste Classification System</h1>
    <p>Identify common types of waste using a trained image classification model.</p>
</div>
""", unsafe_allow_html=True)


# -----------------------------
# Model
# -----------------------------
MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "waste_classifier_mobilenetv2_realworld_128x128.keras"
)


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()


categories = [
    "cardboard",
    "glass",
    "metal",
    "paper",
    "plastic",
    "trash"
]


# -----------------------------
# Upload Section
# -----------------------------
st.markdown(
    '<div class="section-title">Upload an image</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Choose a waste image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# Classification
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Selected image",
        use_container_width=True
    )

    if st.button(
        "🔍 Classify Image",
        use_container_width=True
    ):

        # Resize image
        img = image.resize((128, 128))

        # Normalize pixel values
        img_array = np.array(img).astype("float32") / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Make prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        predicted_index = np.argmax(prediction[0])

        predicted_class = categories[predicted_index]

        confidence = prediction[0][predicted_index] * 100


        # -----------------------------
        # Result
        # -----------------------------
        st.markdown(
            '<div class="section-title">Classification Result</div>',
            unsafe_allow_html=True
        )

        result_html = f"""
<div class="result-card">
    <div class="result-label">Predicted category</div>
    <div class="result-class">{predicted_class.capitalize()}</div>
    <div class="confidence">
        Confidence: <strong>{confidence:.2f}%</strong>
    </div>
</div>
"""

        st.markdown(
            result_html,
            unsafe_allow_html=True
        )

        st.progress(
            float(confidence / 100)
        )


        # -----------------------------
        # Prediction Probabilities
        # -----------------------------
        with st.expander("View prediction probabilities"):

            for i, category in enumerate(categories):

                probability = prediction[0][i] * 100

                st.markdown(
                    f'<p style="color:#26332d !important; '
                    f'opacity:1 !important; margin-bottom:5px;">'
                    f'<strong style="color:#176b4d !important;">'
                    f'{category.capitalize()}</strong> — '
                    f'{probability:.2f}%</p>',
                    unsafe_allow_html=True
                )

                st.progress(
                    float(probability / 100)
                )

else:

    st.info(
        "Upload a waste image above to get started."
    )


# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<div class="footer">
    Waste Classification System • MobileNetV2
</div>
""", unsafe_allow_html=True)
