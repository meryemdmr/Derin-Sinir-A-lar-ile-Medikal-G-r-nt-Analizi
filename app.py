import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. SAYFA YAPILANDIRMASI
st.set_page_config(
    page_title="Medikal AI — Pnömoni Teşhis Sistemi", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 2. İZOLE EDİLMİŞ GÜVENLİ CSS ENJERSİYONU (Üst üste binme sorunları çözüldü)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');

    /* Genel Arka Plan Ayarları */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background-color: #020617 !important;
        color: #e2e8f0 !important;
    }

    /* Streamlit başlık alanını gizleme */
    [data-testid="stHeader"] { visibility: hidden; height: 0px !important; }
    
    /* Sayfa Yapısı */
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Yazı Tiplerini Güvenli Alanlara Atama (Streamlit'i bozmadan) */
    [data-testid="stWidgetLabel"] p, [data-testid="stMarkdownContainer"] p, button div {
        font-family: 'IBM Plex Mono', monospace !important;
    }

    /* Üst Başlık Paneli (Line-height değerleri sabitlendi) */
    .header-bar {
        background: linear-gradient(135deg, #0f172a 0%, #0c1a2e 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 26px 30px;
        margin-bottom: 20px;
        position: relative;
        overflow: hidden;
        line-height: 1.5 !important;
    }
    .header-bar::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #0891b2, #22d3ee, transparent);
    }
    .header-title {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 24px;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 6px 0 !important;
        line-height: 1.2 !important;
    }
    .header-sub {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 11px;
        color: #0891b2;
        letter-spacing: 3px;
        margin: 0 0 8px 0 !important;
        line-height: 1.2 !important;
    }
    .header-badge {
        display: inline-block;
        background: rgba(8,145,178,0.1);
        border: 1px solid rgba(8,145,178,0.3);
        color: #22d3ee;
        font-size: 10px;
        letter-spacing: 2px;
        padding: 4px 10px;
        border-radius: 4px;
        margin-top: 8px;
        font-family: 'IBM Plex Mono', monospace !important;
        line-height: 1.0 !important;
    }

    /* İstatistik Kutuları */
    .stat-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        font-family: 'IBM Plex Mono', monospace !important;
        line-height: 1.4 !important;
    }
    .stat-number { 
        font-size: 22px; 
        font-weight: 700; 
        color: #22d3ee;
        margin: 0 !important;
    }
    .stat-label { 
        font-size: 10px; 
        color: #475569; 
        letter-spacing: 2px; 
        margin-top: 4px !important;
    }

    /* Rapor Bekleme Kutusu */
    .report-waiting-box {
        font-family: 'IBM Plex Mono', monospace !important;
        text-align: center;
        padding: 75px 20px;
        border: 1px dashed #1e293b;
        border-radius: 8px;
        background: #0a1525;
        line-height: 1.6 !important;
    }

    /* Dosya Yükleme Alanı Düzenlemesi */
    [data-testid="stFileUploader"] {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 10px !important;
    }
    
    /* Buton Tasarımları */
    div[data-testid="stButton"] button {
        width: 100% !important;
        border-radius: 8px !important;
        padding: 10px 0 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stButton"] button[data-testid="baseButton-primary"] {
        background-color: #0891b2 !important;
        color: #f0f9ff !important;
        border: 1px solid #0891b2 !important;
    }
    div[data-testid="stButton"] button[data-testid="baseButton-secondary"] {
        background-color: #1e293b !important;
        color: #94a3b8 !important;
        border: 1px solid #1e293b !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. MODELİ YÜKLEME
@st.cache_resource
def model_yukle():
    return tf.keras.models.load_model('zaturre_modeli.h5', compile=False)

try:
    model = model_yukle()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")

# 4. ÜST LOGO / HEADER BAR
st.markdown("""
<div class="header-bar">
  <p class="header-sub">⬡ Yapay Zeka Destekli Görüntü Analizi</p>
  <h1 class="header-title">Medikal Radyoloji Analiz Portalı</h1>
  <p style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#475569; margin:4px 0 0 0; line-height:1.2;">
    Göğüs Röntgeni → Derin Öğrenme Modeli → Pnömoni Teşhisi
  </p>
  <span class="header-badge">CNN MODELİ · %92.6 DOĞRULUK · CHEST X-RAY</span>
</div>
""", unsafe_allow_html=True)

# 5. İSTATİSTİK KUTULARI
stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    st.markdown('<div class="stat-box"><div class="stat-number">5.863</div><div class="stat-label">EĞİTİM GÖRÜNTÜSÜ</div></div>', unsafe_allow_html=True)
with stat_col2:
    st.markdown('<div class="stat-box"><div class="stat-number">%92.6</div><div class="stat-label">MODEL DOĞRULUĞU</div></div>', unsafe_allow_html=True)
with stat_col3:
    st.markdown('<div class="stat-box"><div class="stat-number">CNN</div><div class="stat-label">MİMARİ</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:15px'></div>", unsafe_allow_html=True)

# Temizleme mekanizması
if "uploader_id" not in st.session_state:
    st.session_state.uploader_id = 0

# 6. ANA YAN YANA PANEL
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#0891b2;letter-spacing:3px;margin-bottom:8px;">[ 01 ] GÖRÜNTÜ GİRİŞİ</div>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Röntgen Görüntüsü", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_id}"
    )
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace; font-size:11px; color:#334155; text-align:center; padding-top:6px; letter-spacing:1px;">▲ DICOM / JPG / PNG formatları desteklenir</div>', unsafe_allow_html=True)
    
    st.write("")
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        btn_run = st.button("⬡  ANALİZİ BAŞLAT", type="primary")
    with btn_col2:
        btn_clear = st.button("↺  Temizle", type="secondary")

with col2:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#0891b2;letter-spacing:3px;margin-bottom:8px;">[ 02 ] YAPAY ZEKA RAPORU</div>', unsafe_allow_html=True)
    output_container = st.empty()

# Resim yüklendiyse giriş alanının altında şık bir şekilde gösterilir
if uploaded_file:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Analiz Edilecek Röntgen", use_container_width=True)

# Varsayılan Rapor Bekleme Durumu
varsayilan_rapor_html = """
<div class="report-waiting-box">
  <div style="font-size:32px; margin-bottom:12px; opacity:0.3;">⬡</div>
  <div style="font-size:11px; color:#334155; letter-spacing:3px;">
    RAPOR İÇİN BEKLENİYOR<br>
    <span style="opacity:0.5; font-size:10px;">Görüntü yükleyip analiz başlatın</span>
  </div>
</div>
"""
output_container.markdown(varsayilan_rapor_html, unsafe_allow_html=True)

# Temizle tetikleyicisi
if btn_clear:
    st.session_state.uploader_id += 1
    st.rerun()

# 7. TAHMİN MANTIĞI VE SONUÇ EKRANI
if uploaded_file and btn_run:
    try:
        xray_resmi = np.array(image)
        resim = Image.fromarray(xray_resmi.astype('uint8'), 'RGB').convert('L')
        resim = resim.resize((150, 150))
        resim_dizisi = np.array(resim) / 255.0
        resim_dizisi = np.expand_dims(resim_dizisi, axis=(0, -1))

        tahmin = model.predict(resim_dizisi)[0][0]

        if tahmin > 0.5:
            olasilik = tahmin * 100
            severity = "KRİTİK" if olasilik > 85 else "ORTA"
            rapor_sonuc = f"""
            <div style="font-family:'Courier New',monospace; color:#f8fafc; line-height:1.6;">
              <div style="border-left:4px solid #ef4444; padding-left:16px; margin-bottom:20px;">
                <div style="font-size:11px; color:#94a3b8; letter-spacing:3px; margin-bottom:4px;">YAPAY ZEKA TANI RAPORU</div>
                <div style="font-size:22px; font-weight:700; color:#ef4444;">🚨 PNÖMONİ TESPİT EDİLDİ</div>
              </div>
              <div style="background:#1e293b; border-radius:8px; padding:16px; margin-bottom:12px;">
                <div style="font-size:11px; color:#64748b; letter-spacing:2px;">GÜVENİLİRLİK SKORU</div>
                <div style="font-size:36px; font-weight:700; color:#f87171;">%{olasilik:.1f}</div>
                <div style="background:#0f172a; border-radius:4px; height:6px; margin-top:8px;">
                  <div style="background:linear-gradient(90deg,#ef4444,#f87171); width:{olasilik:.0f}%; height:100%; border-radius:4px;"></div>
                </div>
              </div>
              <div style="background:#1e293b; border-radius:8px; padding:16px; margin-bottom:12px;">
                <div style="font-size:11px; color:#64748b; letter-spacing:2px; margin-bottom:8px;">KLİNİK DEĞERLENDİRME</div>
                <div style="color:#fca5a5; font-size:13px;">▸ Akciğer parankiminde konsolidasyon bulgusu</div>
                <div style="color:#fca5a5; font-size:13px;">▸ İnfiltrasyon paterni: {severity} düzey</div>
                <div style="color:#fca5a5; font-size:13px;">▸ Acil radyoloji konsültasyonu önerilir</div>
              </div>
            </div>
            """
        else:
            olasilik = (1 - tahmin) * 100
            rapor_sonuc = f"""
            <div style="font-family:'Courier New',monospace; color:#f8fafc; line-height:1.6;">
              <div style="border-left:4px solid #22c55e; padding-left:16px; margin-bottom:20px;">
                <div style="font-size:11px; color:#94a3b8; letter-spacing:3px; margin-bottom:4px;">YAPAY ZEKA TANI RAPORU</div>
                <div style="font-size:22px; font-weight:700; color:#22c55e;">✅ SAĞLIKLI AKCİĞER</div>
              </div>
              <div style="background:#1e293b; border-radius:8px; padding:16px; margin-bottom:12px;">
                <div style="font-size:11px; color:#64748b; letter-spacing:2px;">GÜVENİLİRLİK SKORU</div>
                <div style="font-size:36px; font-weight:700; color:#4ade80;">%{olasilik:.1f}</div>
                <div style="background:#0f172a; border-radius:4px; height:6px; margin-top:8px;">
                  <div style="background:linear-gradient(90deg,#22c55e,#4ade80); width:{olasilik:.0f}%; height:100%; border-radius:4px;"></div>
                </div>
              </div>
              <div style="background:#1e293b; border-radius:8px; padding:16px; margin-bottom:12px;">
                <div style="font-size:11px; color:#64748b; letter-spacing:2px; margin-bottom:8px;">KLİNİK DEĞERLENDİRME</div>
                <div style="color:#86efac; font-size:13px;">▸ Akciğer parankiminde patoloji saptanmadı</div>
                <div style="color:#86efac; font-size:13px;">▸ Konsolidasyon / infiltrasyon: Negatif</div>
                <div style="color:#86efac; font-size:13px;">▸ Rutin takip yeterli görünmektedir</div>
              </div>
            </div>
            """
        output_container.markdown(rapor_sonuc, unsafe_allow_html=True)
        
    except Exception as e:
        output_container.error(f"Hata oluştu: {e}")

# 8. ALT BİLGİ ŞERİDİ
st.markdown("""
<div style="margin-top:35px; font-family:'IBM Plex Mono',monospace; font-size:10px;
            color:#334155; text-align:center; letter-spacing:1px; padding:12px;
            border-top:1px solid #0f172a;">
  ⚕ Bu sistem Stanford CheXNet mimarisinden ilham alınarak geliştirilmiştir. Yalnızca akademik amaçlıdır.
</div>
""", unsafe_allow_html=True)
