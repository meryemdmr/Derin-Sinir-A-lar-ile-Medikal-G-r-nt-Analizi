import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Sayfa tasarımı ve başlığı
st.set_page_config(page_title="Zatürre Teşhis Sistemi", layout="centered")

st.title("🩺 Derin Sinir Ağları ile Medikal Görüntü Analizi")
st.write("Lütfen analiz edilmesini istediğiniz Göğüs Röntgeni (X-Ray) fotoğrafını yükleyin.")

# Modelin versiyon hatalarını önlemek için compile=False ile önbelleğe alıp yüklüyoruz
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('zaturre_modeli.h5', compile=False)

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
    
    try:
        # Orijinal kodundaki gibi resmi Siyah-Beyaz (L) yapıp 150x150 boyutuna getiriyoruz
        resim_gri = image.convert('L')
        resim_boyutlu = resim_gri.resize((150, 150))
        
        # Normalize et ve dizileştir
        resim_dizisi = np.array(resim_boyutlu) / 255.0
        
        # Boyutları modelin beklediği (1, 150, 150, 1) şekline getiriyoruz
        resim_dizisi = np.expand_dims(resim_dizisi, axis=(0, -1))
        
        # Tahmin yapma
        prediction = model.predict(resim_dizisi)
        confidence = float(prediction[0][0])
        
        st.subheader("📊 Analiz Sonucu:")
        
        if confidence > 0.5:
            olasilik = confidence * 100
            st.error(f"🚨 Sonuç: ZATÜRRE (Pneumonia) şüphesi tespit edildi.")
            st.write(f"Güven Oranı: %{olasilik:.2f}")
            
            # Orijinal kodundaki kritiklik mantığı
            if olasilik > 85:
                st.warning("Durum: KRİTİK")
            else:
                st.info("Durum: ORTA")
        else:
            st.success(f"✅ Sonuç: SAĞLIKLI (Normal) Akciğer.")
            st.write(f"Güven Oranı: %{(1 - confidence) * 100:.2f}")
            
    except Exception as tahmin_hatasi:
        st.error(f"Görüntü işlenirken veya tahmin edilirken hata oluştu: {tahmin_hatasi}")
