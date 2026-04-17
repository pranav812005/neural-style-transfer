import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from PIL import Image, ImageOps
import os
import io
import json

# ---------------- LOGIN ----------------
def check_login(username, password):
    try:
        with open("users.json") as f:
            users = json.load(f)
        for user in users:
            if user["username"] == username and user["password"] == password:
                return True
    except:
        return False
    return False

st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if not check_login(username, password):
    st.warning("⚠️ Please login first")
    st.stop()

# ---------------- UI ----------------
st.set_page_config(page_title="AI Style Transfer", layout="wide")

st.markdown("""
<style>
.title {
    text-align: center;
    font-size: 40px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🎨 AI Style Transfer</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload image and apply styles</div>", unsafe_allow_html=True)

st.divider()

# ---------------- LOAD IMAGE ----------------
def load_image(image):
    img = Image.open(image)
    img = ImageOps.exif_transpose(img)
    img = img.resize((512, 512))
    img = np.array(img) / 255.0
    img = img.astype(np.float32)
    img = img[np.newaxis, ...]
    return img

# ---------------- STYLE MAP ----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

style_map = {
    "Van Gogh": os.path.join(BASE_DIR, "styles", "vangogh.jpg"),
    "Picasso": os.path.join(BASE_DIR, "styles", "picasso.jpg"),
    "Sketch": os.path.join(BASE_DIR, "styles", "sketch.jpg"),
    "Watercolor": os.path.join(BASE_DIR, "styles", "watercolor.jpg")
}

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙️ Controls")

content_file = st.sidebar.file_uploader("Upload Image", type=["jpg", "png"])
style_option = st.sidebar.selectbox("Choose Style", list(style_map.keys()))
apply = st.sidebar.button("Apply Style")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

model = load_model()

# ---------------- PROCESS ----------------
if content_file and apply:

    content = load_image(content_file)
    style = load_image(style_map[style_option])

    with st.spinner("Processing..."):
        output = model(tf.constant(content), tf.constant(style))[0]

    output_img = tf.keras.preprocessing.image.array_to_img(output[0])
    output_img = ImageOps.exif_transpose(output_img)

    col1, col2 = st.columns(2)

    with col1:
        st.image(content_file, caption="Original")

    with col2:
        st.image(output_img, caption="Styled")

    # SAVE HISTORY (auto folder create)
    os.makedirs("history", exist_ok=True)
    output_path = f"history/{username}_{style_option}.png"
    output_img.save(output_path)

    # DOWNLOAD
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")

    st.download_button("Download Image", data=buf.getvalue(), file_name="output.png")

# ---------------- HISTORY ----------------
st.subheader("📂 History")

if os.path.exists("history"):
    for file in os.listdir("history"):
        if file.startswith(username):
            st.image(f"history/{file}", width=200)