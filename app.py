import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Sayfayı geniş modda açıyoruz (Tasarımın yan yana tam sığması için)
st.set_page_config(page_title="Medikal Radyoloji Analiz Portalı", layout="wide")

# Tasarımını birebir yansıtmak için özel CSS enjeksiyonu
st.markdown("""
<style>
    /* Genel arka plan ve yazı tipi (Koyu Tema) */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
        background-color: #030a16 !important;
        color: #f8fafc !important;
        font-family: 'Courier New', Courier, monospace !important;
    }
    
    /* Üst Karşılama Paneli */
    .portal-header {
        background-color: #091224;
        border: 1px solid #132247;
        padding: 25px;
        border-radius: 6px;
        margin-bottom: 20px;
    }
    
    /* İstatistik Kartları Satırı */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .metric-card {
        flex: 1;
        background-color: #091224;
        border: 1px solid #132247;
        padding: 20px;
        border-radius: 6px;
        text-align: center;
    }
    
    /* Bölüm Başlıkları */
    .section-title {
        font-size: 14px;
        color: #cbd5e1;
        letter-spacing: 2px;
        margin-bottom: 15px;
        font-weight: bold;
    }
    
    /* Panel Kutuları (Giriş ve Rapor Alanı) */
    .box-panel {
        background-color: #091224;
        border: 1px solid #132247;
        border-radius: 6px;
        padding: 35px;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
    }

    /* Orijinal yükleyiciyi gizleyip tasarıma uydurma */
    [data-testid="stFileUploader"] {
        background-color: #060d1a !important;
        border: 1px dashed #1e293b !important;
        border-radius: 6px !important;
    }
    
    /* Butonların Ortak Stili */
    div.stButton > button {
        width: 100% !important;
        font-family: 'Courier New', Courier, monospace !important;
        font-weight: bold !important;
        border-radius: 4px !important;
        padding: 12px 20px !important;
        transition: 0.3s;
    }
    
    /* ANALİZİ BAŞLAT Butonu (Turkuaz/Mavi) */
    div.stButton > button[key="start_btn"] {
        background-color: #00a3c4 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Temizle Butonu (Koyu Gri/Mavi) */
    div.stButton > button[key="clear_btn"] {
        background-color: #334155 !important;
        color: white !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 1. Başlık ve Bilgi Paneli (Orijinal Tasarımın)
st.markdown("""
<div class="portal-header">
    <p style="color: #38bdf8; margin: 0; font-size: 12px; letter-spacing: 2px;">○ Yapay Zeka Destekli Görüntü Analizi</p>
    <h1 style="color: white; margin: 5px 0 10px 0; font-size: 28px; font-weight: bold; font-family: 'Courier New', monospace;">Medikal Radyoloji Analiz Portalı</h1>
    <p style="color: #94a3b8; margin: 0; font-size: 13px;">Göğüs Röntgeni → Derin Öğrenme Modeli → Pnömoni Teşhisi</p>
    <div style="margin-top: 15px;">
        <span style="border: 1px solid #132247; background-color: #060d1a; color: #38bdf8; padding: 5px 12px; border-radius: 4px; font-size: 11px; letter-spacing: 1px;">
            CNN MODELİ · %92.6 DOĞRULUK · GÖĞÜS RÖNTGENİ
        </span>
    </div>
</div>
""", unsafe_allow_html=True)

# 2. İstatistik Kartları (Orijinal Tasarımın)
st.markdown("""
<div class="metric-container">
    <div class="metric-card">
        <h2 style="color: white; margin: 0; font-size: 26px; font-weight: bold;">5.863</h2>
        <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 11px; letter-spacing: 1px;">EĞİTİM GÖRÜNTÜSÜ</p>
    </div>
    <div class="metric-card">
        <h2 style="color: white; margin: 0; font-size: 26px; font-weight: bold;">%92,6</h2>
        <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 11px; letter-spacing: 1px;">MODEL DOĞRULUĞU</p>
    </div>
    <div class="metric-card">
        <h2 style="color: white; margin: 0; font-size: 26px; font-weight: bold;">CNN</h2>
        <p style="color: #94a3b8; margin: 5px 0 0 0; font-size: 11px; letter-spacing: 1px;">MİMARİSİ</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Arka Planda Model Yükleme Adımı
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('zaturre_modeli.h5', compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yüklenirken bir hata oluştu: {e}")

# 3. İki Sütunlu Yan Yana Düzen
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-title">[ 01 ] GÖRÜNTÜ GİRİŞİ</p>', unsafe_allow_html=True)
    
    # Dosya yükleyici
    uploaded_file = st.file_uploader("Röntgen Görüntüsü", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    st.markdown('<p style="color: #475569; font-size: 11px; margin-top: 5px; font-weight: bold;">▲ DICOM/JPG/PNG formatları desteklenir</p>', unsafe_allow_html=True)
    
    # Butonlar yan yana
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_analiz = st.button("○ ANALİZİ BAŞLAT", key="start_btn")
    with btn_col2:
        st.button("↻ Temizle", key="clear_btn")

with col2:
    st.markdown('<p class="section-title">[ 02 ] YAPAY ZEKA RAPORU</p>', unsafe_allow_html=True)
    
    report_placeholder = st.empty()
    
    # Eğer henüz resim yüklenmediyse veya analize basılmadıysa orijinal bekleme ekranı görünür
    if not uploaded_file:
        report_placeholder.markdown("""
        <div class="box-panel">
            <div style="font-size: 32px; color: #475569; margin-bottom: 10px;">⬡</div>
            <div style="color: white; font-weight: bold; font-size: 13px; letter-spacing: 2px;">RAPOR İÇİN BEKLENİYOR</div>
            <div style="color: #475569; font-size: 11px; margin-top: 5px;">Görüntü yükleyip analize başlarken</div>
        </div>
        """, unsafe_allow_html=True)

# Model Analiz Mantığı
if uploaded_file:
    image = Image.open(uploaded_file)
    
    # Resmi sol taraftaki panelin hemen altında önizleme olarak gösterelim
    with col1:
        st.image(image, caption='Yüklenen Röntgen Resmi', use_container_width=True)
        
    if start_analiz:
        with col2:
            report_placeholder.markdown("""
            <div class="box-panel">
                <div style="color: #38bdf8; font-size: 13px; letter-spacing: 1px;">🔄 MODEL GÖRÜNTÜYÜ ANALİZ EDİYOR...</div>
            </div>
            """, unsafe_allow_html=True)
            
        try:
            # Modelinin orijinal görüntü işleme ayarları (150x150, Grayscale)
            resim_gri = image.convert('L')
            resim_boyutlu = resim_gri.resize((150, 150))
            resim_dizisi = np.array(resim_boyutlu) / 255.0
            resim_dizisi = np.expand_dims(resim_dizisi, axis=(0, -1))
            
            prediction = model.predict(resim_dizisi)
            confidence = float(prediction[0][0])
            
            # Sonuç raporunu orijinal tasarım kutunun içine basıyoruz
            if confidence > 0.5:
                olasilik = confidence * 100
                durum_text = "KRİTİK" if olasilik > 85 else "ORTA"
                durum_color = "#ef4444" if olasilik > 85 else "#f59e0b"
                
                report_placeholder.markdown(f"""
                <div class="box-panel" style="border-color: #ef4444 !important;">
                    <h3 style="color: #ef4444; margin: 0; font-size: 16px; font-weight: bold; letter-spacing: 1px;">🚨 YAPAY ZEKA RAPORU</h3>
                    <p style="color: white; font-size: 15px; margin: 15px 0 5px 0; font-weight: bold;">Sonuç: ZATÜRRE (Pneumonia) Şüphesi</p>
                    <p style="color: #94a3b8; margin: 0; font-size: 13px;">Güven Oranı: <span style="color: #ef4444; font-weight: bold;">%{olasilik:.2f}</span></p>
                    <div style="margin-top: 15px; background-color: {durum_color}; color: black; padding: 5px 15px; border-radius: 4px; font-weight: bold; font-size: 11px; letter-spacing: 1px;">
                        DURUM: {durum_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                olasilik = (1 - confidence) * 100
                report_placeholder.markdown(f"""
                <div class="box-panel" style="border-color: #22c55e !important;">
                    <h3 style="color: #22c55e; margin: 0; font-size: 16px; font-weight: bold; letter-spacing: 1px;">✅ YAPAY ZEKA RAPORU</h3>
                    <p style="color: white; font-size: 15px; margin: 15px 0 5px 0; font-weight: bold;">Sonuç: SAĞLIKLI (Normal)</p>
                    <p style="color: #94a3b8; margin: 0; font-size: 13px;">Güven Oranı: <span style="color: #22c55e; font-weight: bold;">%{olasilik:.2f}</span></p>
                </div>
                """, unsafe_allow_html=True)
        except Exception as tahmin_hatasi:
            report_placeholder.error(f"Hata: {tahmin_hatasi}")

# 4. Alt Bilgi Satırları (Orijinal Tasarımın)
st.markdown("""
<br><br><hr style="border-color: #132247;"><br>
<div style="text-align: center; font-family: 'Courier New', monospace; font-size: 11px; color: #475569;">
    <p>‡ Bu sistem Stanford CheXNet mimarisinden ilham alınarak oluşturulmuştur. Yalnızca akademik amaçlıdır.</p>
    <p style="margin-top: 6px; color: #64748b;">API üzerinden kullan 🚀 · Streamlit ile 🔥 · Saha ⚙️</p>
</div>
""", unsafe_allow_html=True)
