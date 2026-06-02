import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. SAYFA YAPILANDIRMASI (Gradio genişlik ve arka plan ayarları için)
st.set_page_config(
    page_title="Medikal AI — Pnömoni Teşhis Sistemi", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# 2. LOCALDEKİ ORİJİNAL GRADIOTHEME VE CSS KODLARININ BİREBİR ENJEKSİYONU
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;700&display=swap');

    /* Tüm sayfayı ve yazı tiplerini localdeki Gradio temana eşitliyoruz */
    * {
        font-family: 'IBM Plex Mono', monospace !important;
    }
    
    html, body, [data-testid="stAppViewContainer"], [data-testid="stMainBlockContainer"] {
        background-color: #020617 !important;
        color: #e2e8f0 !important;
    }

    /* Streamlit'in varsayılan üst menü ve boşluklarını sıfırlama */
    [data-testid="stHeader"] { visibility: hidden; height: 0px !important; }
    [data-testid="stMainBlockContainer"] {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px;
        margin: 0 auto;
    }

    /* Orijinal Gradio CSS Kodların (Aynen Korundu) */
    body { background: #020617 !important; }
    .header-bar {
        background: linear-gradient(135deg, #0f172a 0%, #0c1a2e 100%);
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 28px 32px;
        margin-bottom: 8px;
        position: relative;
        overflow: hidden;
    }
    .header-bar::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #0891b2, #22d3ee, transparent);
    }
    .header-title {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: #e2e8f0;
        margin: 0 0 6px 0;
    }
    .header-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 12px;
        color: #0891b2;
        letter-spacing: 3px;
        margin: 0 0 6px 0;
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
        margin-top: 12px;
        font-family: 'IBM Plex Mono', monospace;
    }
    .stat-box {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 18px;
        text-align: center;
        font-family: 'IBM Plex Mono', monospace;
        width: 100%;
    }
    .stat-number { font-size: 20px; font-weight: 700; color: #22d3ee; }
    .stat-label  { font-size: 10px; color: #475569; letter-spacing: 2px; margin-top: 2px; text-transform: uppercase; }
    .upload-hint {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 11px;
        color: #334155;
        text-align: center;
        padding: 8px;
        letter-spacing: 1px;
    }

    /* Streamlit Dosya Yükleyicisini Gradio Kutusuna Benzetme */
    [data-testid="stFileUploader"] {
        background-color: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 12px !important;
        padding: 15px !important;
    }
    [data-testid="stFileUploader"] section {
        background-color: #020617 !important;
        border: 1px dashed #1e293b !important;
        border-radius: 8px !important;
    }
    
    /* Buton Tasarımlarını Gradio Birincil/İkincil Temasına Eşitleme */
    button[data-testid="baseButton-primary"] {
        background-color: #0891b2 !important;
        color: #f0f9ff !important;
        border: 1px solid #0891b2 !important;
        border-radius: 8px !important;
        width: 100% !important;
        font-weight: 700 !important;
        padding: 10px 0 !important;
    }
    button[data-testid="baseButton-primary"]:hover {
        background-color: #0e7490 !important;
        border-color: #0e7490 !important;
    }
    button[data-testid="baseButton-secondary"] {
        background-color: #1e293b !important;
        color: #94a3b8 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        width: 100% !important;
        font-weight: 700 !important;
        padding: 10px 0 !important;
    }
    button[data-testid="baseButton-secondary"]:hover {
        background-color: #334155 !important;
        border-color: #334155 !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. MODEL YÜKLEME
@st.cache_resource
def model_yukle():
    return tf.keras.models.load_model('zaturre_modeli.h5', compile=False)

try:
    model = model_yukle()
except Exception as e:
    st.error(f"Model yüklenirken hata oluştu: {e}")

# 4. ÜST LOGO / HEADER BAR (Orijinal HTML)
st.markdown("""
<div class="header-bar">
  <p class="header-sub">⬡ Yapay Zeka Destekli Görüntü Analizi</p>
  <h1 class="header-title">Medikal Radyoloji Analiz Portalı</h1>
  <p style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#475569; margin:6px 0 0 0;">
    Göğüs Röntgeni → Derin Öğrenme Modeli → Pnömoni Teşhisi
  </p>
  <span class="header-badge">CNN MODELİ · %92.6 DOĞRULUK · CHEST X-RAY</span>
</div>
""", unsafe_allow_html=True)

# 5. İSTATİSTİK KUTULARI ROW YAPISI (Orijinal HTML & CSS Sınıfları)
stat_col1, stat_col2, stat_col3 = st.columns(3)
with stat_col1:
    st.markdown('<div class="stat-box"><div class="stat-number">5.863</div><div class="stat-label">EĞİTİM GÖRÜNTÜSÜ</div></div>', unsafe_allow_html=True)
with stat_col2:
    st.markdown('<div class="stat-box"><div class="stat-number">%92.6</div><div class="stat-label">MODEL DOĞRULUĞU</div></div>', unsafe_allow_html=True)
with stat_col3:
    st.markdown('<div class="stat-box"><div class="stat-number">CNN</div><div class="stat-label">MİMARİ</div></div>', unsafe_allow_html=True)

st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

# Temizleme mekanizması için session state kontrolü
if "uploader_id" not in st.session_state:
    st.session_state.uploader_id = 0

# 6. ANA YAN YANA PANEL (Giriş ve Rapor Çıktısı)
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#0891b2;letter-spacing:3px;margin-bottom:8px;">[ 01 ] GÖRÜNTÜ GİRİŞİ</div>', unsafe_allow_html=True)
    
    # Gradio'daki gr.Image alanını simüle eden dosya yükleyici
    uploaded_file = st.file_uploader(
        "Röntgen Görüntüsü", 
        type=["jpg", "jpeg", "png"], 
        label_visibility="collapsed",
        key=f"uploader_{st.session_state.uploader_id}"
    )
    st.markdown('<div class="upload-hint">▲ DICOM / JPG / PNG formatları desteklenir</div>', unsafe_allow_html=True)
    
    # Butonlar yan yana
    btn_col1, btn_col2 = st.columns(2)
    with btn_col1:
        btn_run = st.button("⬡  ANALİZİ BAŞLAT", type="primary")
    with btn_col2:
        btn_clear = st.button("↺  Temizle", type="secondary")

with col2:
    st.markdown('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#0891b2;letter-spacing:3px;margin-bottom:8px;">[ 02 ] YAPAY ZEKA RAPORU</div>', unsafe_allow_html=True)
    output_container = st.empty()

# Resim yüklendiyse giriş sütununun altında önizleme gösterilir
if uploaded_file:
    image = Image.open(uploaded_file)
    with col1:
        st.image(image, caption="Analiz Edilecek Röntgen", use_container_width=True)

# Varsayılan Rapor Bekleme Durumu (Orijinal Gradio HTML Tasarımın)
varsayilan_rapor_html = """
<div style="font-family:'IBM Plex Mono',monospace;text-align:center;
            padding:80px 20px;border:1px dashed #1e293b;
            border-radius:8px;background:#0a1525;">
  <div style="font-size:32px;margin-bottom:12px;opacity:0.3;">⬡</div>
  <div style="font-size:11px;color:#334155;letter-spacing:3px;">
    RAPOR İÇİN BEKLENİYOR<br>
    <span style="opacity:0.5">Görüntü yükleyip analiz başlatın</span>
  </div>
</div>
"""
output_container.markdown(varsayilan_rapor_html, unsafe_allow_html=True)

# Temizle Butonuna Basıldığında Reset Atma
if btn_clear:
    st.session_state.uploader_id += 1
    st.rerun()

# 7. TAHMİN VE ANALİZ MANTIĞI (Orijinal Çıktı Şablonların)
if uploaded_file and btn_run:
    try:
        # Gradio resim matrisi simülasyonu ve ön işleme adımların
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
            <div style="font-family:'Courier New',monospace; color:#f8fafc; line-height:1.8;">
              <div style="border-left:4px solid #ef4444; padding-left:16px; margin-bottom:20px;">
                <div style="font-size:11px; color:#94a3b8; letter-spacing:3px; margin-bottom:4px;">YAPAY ZEKA TANI RAPORU</div>
                <div style="font-size:22px; font-weight:700; color:#ef4444;">🚨 PNÖMONİ (ZATÜRRE) TESPİT EDİLDİ</div>
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
                <div style="color:#fca5a5;">▸ Akciğer parankiminde konsolidasyon bulgusu</div>
                <div style="color:#fca5a5;">▸ İnfiltrasyon paterni: {severity} düzey</div>
                <div style="color:#fca5a5;">▸ Acil radyoloji konsültasyonu önerilir</div>
              </div>
              <div style="font-size:10px; color:#475569; border-top:1px solid #1e293b; padding-top:12px;">
                ⚕ Bu rapor yalnızca karar destek amaçlıdır. Kesin tanı için uzman hekim onayı zorunludur.
              </div>
            </div>
            """
        else:
            olasilik = (1 - tahmin) * 100
            rapor_sonuc = f"""
            <div style="font-family:'Courier New',monospace; color:#f8fafc; line-height:1.8;">
              <div style="border-left:4px solid #22c55e; padding-left:16px; margin-bottom:20px;">
                <div style="font-size:11px; color:#94a3b8; letter-spacing:3px; margin-bottom:4px;">YAPAY ZEKA TANI RAPORU</div>
                <div style="font-size:22px; font-weight:700; color:#22c55e;">✅ SAĞLIKLI AKCİĞER TESPİT EDİLDİ</div>
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
                <div style="color:#86efac;">▸ Akciğer parankiminde patoloji saptanmadı</div>
                <div style="color:#86efac;">▸ Konsolidasyon / infiltrasyon: Negatif</div>
                <div style="color:#86efac;">▸ Rutin takip yeterli görünmektedir</div>
              </div>
              <div style="font-size:10px; color:#475569; border-top:1px solid #1e293b; padding-top:12px;">
                ⚕ Bu rapor yalnızca karar destek amaçlıdır. Kesin tanı için uzman hekim onayı zorunludur.
              </div>
            </div>
            """
        output_container.markdown(rapor_sonuc, unsafe_allow_html=True)
        
    except Exception as e:
        output_container.error(f"Hata oluştu: {e}")

# 8. ALT BİLGİ ŞERİDİ
st.markdown("""
<div style="margin-top:24px;font-family:'IBM Plex Mono',monospace;font-size:10px;
            color:#334155;text-align:center;letter-spacing:1px;padding:12px;
            border-top:1px solid #0f172a;">
  ⚕ Bu sistem Stanford CheXNet mimarisinden ilham alınarak geliştirilmiştir.
  Yalnızca akademik amaçlıdır.
</div>
""", unsafe_allow_html=True)
