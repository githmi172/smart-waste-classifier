import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np
import json

# ඇප් එකේ පිටුවේ සැකසුම් (Page Configuration)
st.set_page_config(page_title="Smart Waste Classifier", layout="centered")

st.title("♻️ Smart Waste Classification System")
st.write("කරුණාකර අපද්‍රව්‍ය වර්ගීකරණය සඳහා රූපයක් (Image) අප්ලෝඩ් කරන්න.")

# 1. මොඩල් එක සහ ක්ලාස් නේම්ස් ලෝඩ් කිරීම
@st.cache_resource
def load_my_model():
    # ඔයාගේ මොඩල් ෆයිල් එකේ නම මෙතන තියෙන නමට සමාන විය යුතුයි
    model = tf.keras.models.load_model('waste_classifier_model.keras')
    return model

model = load_my_model()

with open('class_names.json', 'r') as f:
    class_names = json.load(f)

# 2. ඉමේජ් එකක් අප්ලෝඩ් කරන තැන (File Uploader)
uploaded_file = st.file_uploader("රූපයක් තෝරන්න...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # අප්ලෝඩ් කරපු ඉමේජ් එක ඇප් එකේ පෙන්වීම
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_container_width=True)
    
    st.write("⏳ විශ්ලේෂණය කරමින් පවතිනවා (Classifying)...")
    
    # MobileNetV2 එකට ගැළපෙන පරිදි ඉමේජ් එක 224x224 වලට සකස් කිරීම
    size = (224, 224)
    image_resized = ImageOps.fit(image, size, Image.Resampling.LENS)
    img_array = np.asarray(image_resized)
    img_array_expanded = np.expand_dims(img_array, axis=0) # Batch dimension එක එකතු කිරීම
    
    # 3. අනාවැකිය ලබා ගැනීම (Prediction)
    predictions = model.predict(img_array_expanded)[0]
    best_index = np.argmax(predictions)
    
    predicted_class = class_names[best_index]
    confidence = predictions[best_index] * 100
    
    # 4. ප්‍රතිඵල පෙන්වීම
    st.subheader(f"අනාවැකිය (Prediction): **{predicted_class}**")
    st.write(f"**නිවැරදිතාවයේ ප්‍රතිශතය (Confidence Level):** {confidence:.2f}%")
    
    # 5. සියලුම ක්ලාස් වල සම්භාවිතාවන් පෙන්වීම (Probabilities for all classes)
    st.write("### සියලුම වර්ගයන්ගේ සම්භාවිතාවන්:")
    for idx, class_name in enumerate(class_names):
        prob = float(predictions[idx])
        st.write(f"- **{class_name}**: {prob*100:.2f}%")
        st.progress(prob)