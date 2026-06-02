import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Sayfa ayarları ve Streamlit'in kendi boşluklarını sıfırlama
st.set_page_config(page_title="Medikal Radyoloji Analiz Portalı", layout="wide", initial_sidebar_state="collapsed")

# Streamlit'i tamamen senin localindeki Gradio şablonuna benzeten CSS enjeksiyonu
st.markdown("""
<style>
    /* Google Fonts üzerinden senin orijinal IBM Plex Mono fontunu çekiyoruz */
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');

    /* Tüm sayfayı kapsayan genel renkler ve yazı tipi */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"], [data-testid="stMainBlockContainer"] {
        background-color: #030a16 !important;
        color: #f8fafc !important;
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    /* Sayfa kenar boşluklarını daraltıp localdeki gibi kompakt yapıyoruz */
    [data-testid="stMainBlockContainer"] {
        padding-top: 1.5rem !important;
        padding-bottom: 1.5rem !important;
        padding-left: 4rem !important;
        padding-right: 4rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Streamlit'in kendi üst menü ve logolarını tamamen gizliyoruz */
    #MainMenu, header, footer {visibility: hidden;}
    
    /* Üst Karşılama Paneli */
    .portal-header {
        background-color: #091224;
        border: 1px solid #132247;
        padding: 20px;
        border-radius: 8px;
        margin-bottom: 20px;
    }
    
    /* İstatistik Kartları Satırı */
    .metric-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    
    .metric-card {
        flex: 1;
        background-color: #091224;
        border: 1px solid #132247;
        padding: 15px;
        border-radius: 8px;
        text-align: center;
    }
    
    /* [ 01 ] ve [ 02 ] Bölüm Başlıkları */
    .section-title {
        font-size: 12px;
        color: #334155;
        letter-spacing: 2px;
        margin-bottom: 12px;
        font-weight: 700;
    }
    
    /* Localdeki Orijinal Rapor Bekleme Kutusu (Birebir CSS Kodun) */
    .gradio-report-box {
        font-family: 'IBM Plex Mono', monospace !important;
        text-align: center;
        padding: 80px 20px;
        border: 1px dashed #1e293b;
        border-radius: 8px;
        background: #0a1525;
        min-height: 270px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
    }
    
    /* Streamlit Dosya Yükleyicisini Karartma ve Kareleştirme */
    [data-testid="stFileUploader"] {
        background-color: #0a1525 !important;
        border: 1px dashed #1e293b !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
    
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
        padding: 0 !important;
    }
    
    /* Buton Tasarımları (Gradio Esnekliği) */
    div.stButton > button {
        width: 100% !important;
        font-family: 'IBM Plex Mono', monospace !important;
        font-weight: bold !important;
        border-radius: 6px !important;
        padding: 10px 15px !important;
        transition: 0.2s;
        letter-spacing: 1px;
    }
    
    /* ANALİZİ BAŞLAT Butonu */
    div.stButton > button[key="start_btn"] {
        background-color: #00a3c4 !important;
        color: white !important;
        border: none !important;
    }
    
    /* Temizle Butonu */
    div.stButton > button[key="clear_btn"] {
        background-color: #1e293b !important;
        color: #cbd5e1 !important;
        border: 1px solid #334155 !important;
    }
    
    /* Orijinal Alt Bilgi Şeridi */
    .portal-footer {
        margin-top: 35px;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 10px;
        color: #334155;
        text-align: center;
        letter-spacing: 1px;
        padding-top: 15px;
        border-top: 1px solid #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# 1. BAŞLIK PANELİ
st.markdown("""
<div class="portal-header">
    <p style="color: #38bdf8; margin: 0; font-size: 11px; letter-spacing: 2px;">○ Yapay Zeka Destekli Görüntü Analizi</p>
    <h1 style="color: white; margin: 5px 0 8px 0; font-size: 24px; font-weight: bold; letter-spacing: 1px;">Medikal Radyoloji Analiz Portalı</h1>
    <p style="color: #64748b; margin: 0; font-size: 12px;">Göğüs Röntgeni → Derin Öğrenme Modeli → Pnömoni Teşhisi</p>
</div>
""", unsafe_allow_html=True)

# 2. ÜST METRİK KARTLARI
st.markdown("""
<div class="metric-container">
    <div class="metric-card">
        <h2 style="color: white; margin: 0; font-size: 22px; font-weight: bold;">5.863</h2>
        <p style="color: #475569; margin: 3px 0 0 0; font-size: 10px; letter-spacing: 1px;">EĞİTİM GÖRÜNTÜSÜ</p>
    </div>
    <div class="metric-card">
        <h2 style="color: white; margin: 0; font-size: 22px; font-weight: bold;">%92,6</h2>
        <p style="color: #475569; margin: 3px 0 0 0; font-size: 10px; letter-spacing: 1px;">MODEL DOĞRULUĞU</p>
    </div>
    <div class="metric-card">
        <h2 style="color: white; margin: 0; font-size: 22px; font-weight: bold;">CNN</h2>
        <p style="color: #475569; margin: 3px 0 0 0; font-size: 10px; letter-spacing: 1px;">MİMARİSİ</p>
    </div>
</div>
""", unsafe_allow_html=True)

# MODELİ ÖNBELLEKTEN YÜKLEME
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model('zaturre_modeli.h5', compile=False)

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Model yükleme hatası: {e}")

# 3. İKİ SÜTUNLU GRİD YAN YANA YAPI
col1, col2 = st.columns(2)

with col1:
    st.markdown('<p class="section-title">[ 01 ] GÖRÜNTÜ GİRİŞİ</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Röntgen Görüntüsü", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    st.write("")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        start_analiz = st.button("○ ANALİZİ BAŞLAT", key="start_btn")
    with btn_col2:
        clear_page = st.button("↻ Temizle", key="clear_btn")

with col2:
    st.markdown('<p class="section-title">[ 02 ] YAPAY ZEKA RAPORU</p>', unsafe_allow_html=True)
    report_placeholder = st.empty()

# Resim seçildiyse hemen yükleme kutusunun altına şık bir şekilde basılıyor
if uploaded_file:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption='Yüklenen Röntgen Resmi', use_container_width=True)

# İlk Açılış veya Temizleme Durumu (Orijinal Kodundaki Boşluklar ve Sembollerle)
if not uploaded_file or clear_page:
    report_placeholder.markdown("""
    <div class="gradio-report-box">
        <div style="font-size:32px; margin-bottom:12px; opacity:0.3; color:#334155;">⬡</div>
        <div style="font-size:11px; color:#334155; letter-spacing:3px; line-height:1.8;">
            RAPOR İÇİN BEKLENİYOR<br>
            <span style="opacity:0.5; font-size:10px; letter-spacing:1px;">Görüntü yükleyip analiz başlatın</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Analiz Başlatıldığında Tetiklenen Orijinal Mantık
if uploaded_file and start_analiz and not clear_page:
    with col2:
        report_placeholder.markdown("""
        <div class="gradio-report-box">
            <div style="font-size:12px; color:#00a3c4; letter-spacing:2px; font-weight:bold;">🔄 MODEL GÖRÜNTÜYÜ ANALİZ EDİYOR...</div>
        </div>
        """, unsafe_allow_html=True)
        
    try:
        # Localdeki orijinal resim işleme mantığının birebir aynısı
        resim_gri = image.convert('L')
        resim_boyutlu = resim_gri.resize((150, 150))
        resim_dizisi = np.array(resim_boyutlu) / 255.0
        resim_dizisi = np.expand_dims(resim_dizisi, axis=(0, -1))
        
        tahmin = model.predict(resim_dizisi)[0][0]
        
        if tahmin > 0.5:
            olasilik = tahmin * 100
            severity = "KRİTİK" if olasilik > 85 else "ORTA"
            color = "#f87171" if severity == "KRİTİK" else "#fbbf24"
            
            report_placeholder.markdown(f"""
            <div class="gradio-report-box" style="border: 1px solid {color} !important;">
                <div style="font-size:13px; color:{color}; font-weight:bold; letter-spacing:2px; margin-bottom:12px;">🚨 TESPİT EDİLDİ</div>
                <div style="font-size:16px; color:#f8fafc; font-weight:bold; margin-bottom:6px;">ZATÜRRE (Pneumonia) Şüphesi</div>
                <div style="font-size:12px; color:#94a3b8; margin-bottom:15px;">Güven Oranı: %{olasilik:.2f}</div>
                <div style="background:{color}; color:#030a16; padding:5px 14px; border-radius:4px; font-size:11px; font-weight:bold; letter-spacing:1px;">
                    DURUM: {severity}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            olasilik = (1 - tahmin) * 100
            report_placeholder.markdown(f"""
            <div class="gradio-report-box" style="border: 1px solid #34d399 !important;">
                <div style="font-size:13px; color:#34d399; font-weight:bold; letter-spacing:2px; margin-bottom:12px;">✅ TEMİZ</div>
                <div style="font-size:16px; color:#f8fafc; font-weight:bold; margin-bottom:6px;">SAĞLIKLI (Normal) Akciğer</div>
                <div style="font-size:12px; color:#94a3b8;">Güven Oranı: %{olasilik:.2f}</div>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        report_placeholder.error(f"Tahmin işlemi esnasında hata oluştu: {e}")

# 4. ORİJİNAL ALT BİLGİ ŞERİDİ
st.markdown("""
<div class="portal-footer">
  ⚕ Bu sistem Stanford CheXNet mimarisinden ilham alınarak geliştirilmiştir. Yalnızca akademik amaçlıdır.
</div>
""", unsafe_allow_html=True)
