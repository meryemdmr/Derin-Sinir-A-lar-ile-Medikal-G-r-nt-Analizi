import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Sayfa tasarımı ve başlığı
st.set_page_config(page_title="Zatürre Teşhis Sistemi", layout="centered")

st.title("Zx 🩺 Derin Sinir Ağları ile Medikal Görüntü Analizi")
st.write("Lütfen analiz edilmesini istediğiniz Göğüs Röntgeni (X-Ray) fotoğrafını yükleyin.")

# Modelin her seferinde tekrar yüklenip sistemi yavaşlatmasını engellemek için önbelleğe alıyoruz
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('zaturre_modeli.h5')

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")

# Fotoğraf yükleme butonu
uploaded_file = st.file_uploader("Bir Röntgen Resmi Seçin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Resmi yükle ve ekranda göster
    image = Image.open(uploaded_file)
    st.image(image, caption='Yüklenen Röntgen Resmi', use_column_width=True)
    
    st.write("🔄 Model analiz ediyor, lütfen bekleyin...")
    
    # Modelin eğitildiği resim boyutu (Genelde 224x224 veya 150x150 olur, kendi modeline göre değiştirebilirsin)
    img_resized = image.resize((224, 224)) 
    img_array = np.array(img_resized)
    
    # Eğer resim siyah-beyaz ise 3 kanallı RGB formatına çeviriyoruz
    if len(img_array.shape) == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
        
    # Normalizasyon ve Batch boyutu ekleme (1, 224, 224, 3)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Tahmin yapma
    prediction = model.predict(img_array)
    confidence = float(prediction[0][0])
    
    st.subheader("📊 Analiz Sonucu:")
    # Model çıktısına göre sınıflandırma (0.5 eşik değeridir)
    if confidence > 0.5:
        st.error(f"🚨 Sonuç: ZATÜRRE (Pneumonia) şüphesi tespit edildi.")
        st.write(f"Güven Oranı: %{confidence * 100:.2f}")
    else:
        st.success(f"✅ Sonuç: SAĞLIKLI (Normal) Akciğer.")
        st.write(f"Güven Oranı: %{(1 - confidence) * 100:.2f}")
