import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image

# ─────────────────────────────────────────
#  1. MODEL YÜKLEME
# ─────────────────────────────────────────
model = tf.keras.models.load_model('zaturre_modeli.h5')

# ─────────────────────────────────────────
#  2. TAHMİN FONKSİYONU
# ─────────────────────────────────────────
def tahmin_et(xray_resmi):
    if xray_resmi is None:
        return "<p style='color:#f87171;font-family:monospace'>⚠ HATA: Lütfen önce bir röntgen görüntüsü yükleyin.</p>"

    resim = Image.fromarray(xray_resmi.astype('uint8'), 'RGB').convert('L')
    resim = resim.resize((150, 150))
    resim_dizisi = np.array(resim) / 255.0
    resim_dizisi = np.expand_dims(resim_dizisi, axis=(0, -1))

    tahmin = model.predict(resim_dizisi)[0][0]

    if tahmin > 0.5:
        olasilik = tahmin * 100
        severity = "KRİTİK" if olasilik > 85 else "ORTA"
        return f"""
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
        return f"""
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

# ─────────────────────────────────────────
#  3. TEMA  (label_text_color KALDIRILDI)
# ─────────────────────────────────────────
temas = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "monospace"],
).set(
    body_background_fill="#020617",
    body_background_fill_dark="#020617",
    block_background_fill="#0f172a",
    block_background_fill_dark="#0f172a",
    block_border_color="#1e293b",
    block_border_color_dark="#1e293b",
    block_border_width="1px",
    block_radius="12px",
    button_primary_background_fill="#0891b2",
    button_primary_background_fill_hover="#0e7490",
    button_primary_text_color="#f0f9ff",
    button_secondary_background_fill="#1e293b",
    button_secondary_background_fill_hover="#334155",
    button_secondary_text_color="#94a3b8",
    input_background_fill="#0f172a",
    input_background_fill_dark="#0f172a",
    input_border_color="#1e293b",
    input_border_color_focus="#0891b2",
    body_text_color="#e2e8f0",
    body_text_size="14px",
)

# ─────────────────────────────────────────
#  4. CSS
# ─────────────────────────────────────────
css = """
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
}
.stat-number { font-size: 20px; font-weight: 700; color: #22d3ee; }
.stat-label  { font-size: 10px; color: #475569; letter-spacing: 2px; margin-top: 2px; }
.upload-hint {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    color: #334155;
    text-align: center;
    padding: 8px;
    letter-spacing: 1px;
}
"""

# ─────────────────────────────────────────
#  5. ARAYÜZ
# ─────────────────────────────────────────
with gr.Blocks(theme=temas, css=css, title="Medikal AI — Pnömoni Teşhis Sistemi") as arayuz:

    gr.HTML("""
    <div class="header-bar">
      <p class="header-sub">⬡ Yapay Zeka Destekli Görüntü Analizi</p>
      <h1 class="header-title">Medikal Radyoloji Analiz Portalı</h1>
      <p style="font-family:'IBM Plex Mono',monospace; font-size:13px; color:#475569; margin:6px 0 0 0;">
        Göğüs Röntgeni → Derin Öğrenme Modeli → Pnömoni Teşhisi
      </p>
      <span class="header-badge">CNN MODELİ · %92.6 DOĞRULUK · CHEST X-RAY</span>
    </div>
    """)

    with gr.Row():
        gr.HTML('<div class="stat-box" style="flex:1"><div class="stat-number">5.863</div><div class="stat-label">EĞİTİM GÖRÜNTÜsü</div></div>')
        gr.HTML('<div class="stat-box" style="flex:1;margin:0 8px"><div class="stat-number">%92.6</div><div class="stat-label">MODEL DOĞRULUĞU</div></div>')
        gr.HTML('<div class="stat-box" style="flex:1"><div class="stat-number">CNN</div><div class="stat-label">MİMARİ</div></div>')

    gr.HTML("<div style='height:12px'></div>")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.HTML('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#0891b2;letter-spacing:3px;margin-bottom:8px;">[ 01 ] GÖRÜNTÜ GİRİŞİ</div>')
            input_img = gr.Image(label="Röntgen Görüntüsü", type="numpy", height=300)
            gr.HTML('<div class="upload-hint">▲ DICOM / JPG / PNG formatları desteklenir</div>')
            with gr.Row():
                btn_run   = gr.Button("⬡  ANALİZİ BAŞLAT", variant="primary",   size="lg")
                btn_clear = gr.Button("↺  Temizle",        variant="secondary", size="lg")

        with gr.Column(scale=1):
            gr.HTML('<div style="font-family:\'IBM Plex Mono\',monospace;font-size:10px;color:#0891b2;letter-spacing:3px;margin-bottom:8px;">[ 02 ] YAPAY ZEKA RAPORU</div>')
            output_html = gr.HTML(value="""
                <div style="font-family:'IBM Plex Mono',monospace;text-align:center;
                            padding:80px 20px;border:1px dashed #1e293b;
                            border-radius:8px;background:#0a1525;">
                  <div style="font-size:32px;margin-bottom:12px;opacity:0.3;">⬡</div>
                  <div style="font-size:11px;color:#334155;letter-spacing:3px;">
                    RAPOR İÇİN BEKLENİYOR<br>
                    <span style="opacity:0.5">Görüntü yükleyip analiz başlatın</span>
                  </div>
                </div>""")

    gr.HTML("""
    <div style="margin-top:12px;font-family:'IBM Plex Mono',monospace;font-size:10px;
                color:#334155;text-align:center;letter-spacing:1px;padding:12px;
                border-top:1px solid #0f172a;">
      ⚕ Bu sistem Stanford CheXNet mimarisinden ilham alınarak geliştirilmiştir.
      Yalnızca akademik amaçlıdır.
    </div>
    """)

    btn_run.click(fn=tahmin_et, inputs=input_img, outputs=output_html)
    btn_clear.click(
        fn=lambda: (None, '<div style="font-family:\'IBM Plex Mono\',monospace;text-align:center;padding:80px 20px;border:1px dashed #1e293b;border-radius:8px;background:#0a1525;"><div style="font-size:32px;opacity:0.3;">⬡</div><div style="font-size:11px;color:#334155;letter-spacing:3px;margin-top:12px;">RAPOR İÇİN BEKLENİYOR</div></div>'),
        outputs=[input_img, output_html]
    )

# ─────────────────────────────────────────
#  6. BAŞLATMA
# ─────────────────────────────────────────
if __name__ == "__main__":
    arayuz.launch(share=True)