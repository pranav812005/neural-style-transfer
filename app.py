import streamlit as st
st.set_page_config(page_title="AI Style Transfer", layout="wide")

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
        return users.get(username) == password
    except:
        return False

st.sidebar.title("🔐 Login")

username = st.sidebar.text_input("Username")
password = st.sidebar.text_input("Password", type="password")

if not check_login(username, password):
    st.warning("⚠️ Please login first")
    st.stop()

# ---------------- UI ----------------
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
apply = st.sidebar.button("🚀 Apply Style")

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    return hub.load("https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2")

model = load_model()

# ---------------- PROCESS ----------------
if content_file and apply:

    if not os.path.exists(style_map[style_option]):
        st.error("❌ Style image not found!")
        st.stop()

    content = load_image(content_file)
    style = load_image(style_map[style_option])

    with st.spinner("🎨 Applying AI magic..."):
        output = model(tf.constant(content), tf.constant(style))[0]

    output_img = tf.keras.preprocessing.image.array_to_img(output[0])
    output_img = ImageOps.exif_transpose(output_img)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📸 Original")
        st.image(content_file, use_container_width=True)

    with col2:
        st.subheader("✨ Stylized")
        st.image(output_img, use_container_width=True)

    # 💾 SAVE HISTORY
    os.makedirs("history", exist_ok=True)
    output_path = f"history/{username}_{style_option}.png"
    output_img.save(output_path)

    # 📥 DOWNLOAD
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")

    st.download_button(
        "📥 Download Image",
        data=buf.getvalue(),
        file_name="styled_image.png",
        mime="image/png"
    )

# ---------------- HISTORY ----------------
st.divider()
st.subheader("📂 Your History")

if os.path.exists("history"):
    files = os.listdir("history")
    user_files = [f for f in files if f.startswith(username)]

    if user_files:
        cols = st.columns(3)
        for i, file in enumerate(user_files):
            with cols[i % 3]:
                st.image(f"history/{file}", use_container_width=True)
    else:
        st.info("No history yet 🚀")