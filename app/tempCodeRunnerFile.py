"""
Accident Detection System — Streamlit Application
===================================================

Launch with:
    streamlit run app/streamlit_app.py

Features:
    1. Image-based accident detection  (MobileNetV2 / TensorFlow)
    2. Video-based accident detection   (R3D-18 / PyTorch)
    3. Severity assessment              (YOLOv8 / Ultralytics)
"""

import sys
import os
import tempfile
import random

import streamlit as st

# Ensure project root is on sys.path so `src.*` imports work
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.config import (
    PAGE_TITLE,
    PAGE_ICON,
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    IMAGE_MODEL_PATH,
    VIDEO_MODEL_PATH,
    SEVERITY_MODEL_PATH,
    FAST2SMS_API_KEY,           # ← ADD THIS
    SMS_CONFIDENCE_THRESHOLD,   # ← ADD THIS
)
from app.components import (
    render_image_result,
    render_video_result,
    render_severity_result,
    render_sms_alert_widget,    # ← ADD THIS
    render_sms_result,          # ← ADD THIS
)

# ── ADD THIS LINE ──
from src.services.alert_service import send_accident_sms


DEMO_MODE = os.getenv("DEMO_MODE", "0").strip().lower() in {"1", "true", "yes"}


def _fake_prediction(num_classes: int) -> dict:
    rng = random.Random()
    cls_idx = rng.randrange(0, num_classes)
    score = rng.uniform(0.6, 0.98)
    return {"class_index": cls_idx, "score": score}

# ── Page configuration ────────────────────────────────────────────────────
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")




# ── CUSTOM CSS + ANIMATED BACKGROUND ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');



/* ── BACKGROUND ── */
.stApp {
    background: #05050a !important;
}
[data-testid="stSidebar"] {
    background: #0d0d15 !important;
    border-right: 1px solid rgba(255,45,45,0.2) !important;
}
[data-testid="stSidebar"] h1 {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 26px !important;
    letter-spacing: 4px !important;
    color: #eeeef8 !important;
    text-shadow: 0 0 20px rgba(255,45,45,0.4) !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stRadio label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #7070a0 !important;
}
[data-testid="stSidebar"] .stMarkdown p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2.5px !important;
    text-transform: uppercase !important;
    color: #383850 !important;
}
h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 5px !important;
    color: #eeeef8 !important;
    line-height: 0.95 !important;
}
h1 { font-size: 56px !important; }
h2 { font-size: 40px !important; }
h3 { font-size: 22px !important; color: #ff2d2d !important; }
p, .stMarkdown p {
    font-family: 'Outfit', sans-serif !important;
    color: #7070a0 !important;
    font-weight: 300 !important;
    line-height: 1.75 !important;
}
[data-testid="stFileUploader"] {
    background: #161622 !important;
    border: 1px dashed rgba(255,45,45,0.3) !important;
}
.stButton > button {
    background: #ff2d2d !important;
    color: #ffffff !important;
    border: none !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 12px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 12px 32px !important;
    border-radius: 0 !important;
    box-shadow: 0 0 30px rgba(255,45,45,0.25) !important;
    clip-path: polygon(8px 0%,100% 0%,100% calc(100% - 8px),calc(100% - 8px) 100%,0% 100%,0% 8px) !important;
}
.stButton > button:hover {
    background: #ff6b35 !important;
    box-shadow: 0 0 50px rgba(255,45,45,0.45) !important;
    transform: translateY(-2px) !important;
}
.stTextInput input {
    background: #161622 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    color: #eeeef8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 0 !important;
}
.stTextInput input:focus {
    border-color: rgba(255,45,45,0.4) !important;
}
.stToggle label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 10px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #7070a0 !important;
}
div[data-testid="stSuccess"] {
    background: rgba(57,255,20,0.05) !important;
    border-left: 3px solid #39ff14 !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stError"] {
    background: rgba(255,45,45,0.05) !important;
    border-left: 3px solid #ff2d2d !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stWarning"] {
    background: rgba(255,214,0,0.05) !important;
    border-left: 3px solid #ffd600 !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
div[data-testid="stInfo"] {
    background: rgba(0,229,255,0.04) !important;
    border-left: 3px solid #00e5ff !important;
    border-radius: 0 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stMetric"] {
    background: #161622 !important;
    border: 1px solid rgba(255,255,255,0.05) !important;
    border-left: 2px solid #ff2d2d !important;
    padding: 16px !important;
    border-radius: 0 !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 3px !important;
    text-transform: uppercase !important;
    color: #383850 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 40px !important;
    letter-spacing: 2px !important;
    color: #eeeef8 !important;
}
[data-testid="stImage"] img {
    border: 1px solid rgba(255,45,45,0.15) !important;
}
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg,transparent,rgba(255,45,45,0.2),transparent) !important;
}
::-webkit-scrollbar { width: 4px !important; }
::-webkit-scrollbar-track { background: #05050a !important; }
::-webkit-scrollbar-thumb { background: #1e1e2e !important; }
::-webkit-scrollbar-thumb:hover { background: #ff2d2d !important; }
thead tr th {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 9px !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    color: #383850 !important;
    background: #161622 !important;
    border-bottom: 1px solid rgba(255,45,45,0.15) !important;
}
tbody tr td {
    font-family: 'Outfit', sans-serif !important;
    font-size: 13px !important;
    color: #7070a0 !important;
    background: #0d0d15 !important;
    border-bottom: 1px solid rgba(255,255,255,0.04) !important;
}
</style>

<!-- ANIMATED BACKGROUND -->
<canvas id="bg-canvas" style="position:fixed;top:0;left:0;width:100%;height:100%;z-index:0;pointer-events:none;opacity:0.6"></canvas>

<!-- CURSOR -->
<div id="custom-cursor"></div>
<div id="custom-cursor-ring"></div>

<script>
const canvas = document.getElementById('bg-canvas');
const ctx    = canvas.getContext('2d');
function resize(){canvas.width=window.innerWidth;canvas.height=window.innerHeight;}
resize();
window.addEventListener('resize',resize);
function drawGrid(){
  ctx.strokeStyle='rgba(255,45,45,0.04)';ctx.lineWidth=1;const size=60;
  for(let x=0;x<canvas.width;x+=size){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,canvas.height);ctx.stroke();}
  for(let y=0;y<canvas.height;y+=size){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(canvas.width,y);ctx.stroke();}
}
const particles=Array.from({length:60},()=>({
  x:Math.random()*window.innerWidth,y:Math.random()*window.innerHeight,
  r:Math.random()*1.5+0.3,vx:(Math.random()-.5)*0.4,vy:(Math.random()-.5)*0.4,
  alpha:Math.random()*0.5+0.1,color:Math.random()>0.7?'0,229,255':'255,45,45'
}));
const streaks=Array.from({length:8},()=>({
  x:Math.random()*window.innerWidth,y:-200,speed:Math.random()*1.5+0.5,
  len:Math.random()*120+60,alpha:Math.random()*0.4+0.1,width:Math.random()*0.8+0.2
}));
let frame=0;
function animate(){
  ctx.clearRect(0,0,canvas.width,canvas.height);
  const grd=ctx.createRadialGradient(canvas.width*.5,0,0,canvas.width*.5,0,canvas.height*.7);
  grd.addColorStop(0,'rgba(255,45,45,0.06)');grd.addColorStop(1,'rgba(0,0,0,0)');
  ctx.fillStyle=grd;ctx.fillRect(0,0,canvas.width,canvas.height);
  drawGrid();
  ctx.strokeStyle='rgba(255,45,45,0.06)';ctx.lineWidth=1;
  ctx.beginPath();ctx.moveTo(canvas.width/2,0);ctx.lineTo(canvas.width/2,canvas.height);ctx.stroke();
  ctx.beginPath();ctx.moveTo(0,canvas.height/2);ctx.lineTo(canvas.width,canvas.height/2);ctx.stroke();
  [{r:200,a:0.03},{r:350,a:0.02},{r:500,a:0.015}].forEach(c=>{
    ctx.beginPath();ctx.arc(canvas.width/2,canvas.height/2,c.r,0,Math.PI*2);
    ctx.strokeStyle=`rgba(255,45,45,${c.a})`;ctx.lineWidth=1;ctx.stroke();
  });
  particles.forEach(p=>{
    p.x+=p.vx;p.y+=p.vy;
    if(p.x<0||p.x>canvas.width)p.vx*=-1;
    if(p.y<0||p.y>canvas.height)p.vy*=-1;
    ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);
    ctx.fillStyle=`rgba(${p.color},${p.alpha})`;ctx.fill();
    ctx.beginPath();ctx.arc(p.x,p.y,p.r*4,0,Math.PI*2);
    ctx.fillStyle=`rgba(${p.color},${p.alpha*0.15})`;ctx.fill();
  });
  for(let i=0;i<particles.length;i++){
    for(let j=i+1;j<particles.length;j++){
      const dx=particles[i].x-particles[j].x,dy=particles[i].y-particles[j].y;
      const dist=Math.sqrt(dx*dx+dy*dy);
      if(dist<120){
        ctx.beginPath();ctx.moveTo(particles[i].x,particles[i].y);
        ctx.lineTo(particles[j].x,particles[j].y);
        ctx.strokeStyle=`rgba(255,45,45,${0.06*(1-dist/120)})`;
        ctx.lineWidth=0.5;ctx.stroke();
      }
    }
  }
  streaks.forEach(s=>{
    s.y+=s.speed;
    if(s.y>canvas.height+s.len){s.y=-s.len;s.x=Math.random()*canvas.width;}
    const grad=ctx.createLinearGradient(s.x,s.y-s.len,s.x,s.y);
    grad.addColorStop(0,'rgba(255,45,45,0)');grad.addColorStop(1,`rgba(255,45,45,${s.alpha})`);
    ctx.beginPath();ctx.moveTo(s.x,s.y-s.len);ctx.lineTo(s.x,s.y);
    ctx.strokeStyle=grad;ctx.lineWidth=s.width;ctx.stroke();
  });
  const cSize=40;ctx.strokeStyle='rgba(255,45,45,0.15)';ctx.lineWidth=1.5;
  ctx.beginPath();ctx.moveTo(20,20+cSize);ctx.lineTo(20,20);ctx.lineTo(20+cSize,20);ctx.stroke();
  ctx.beginPath();ctx.moveTo(canvas.width-20-cSize,20);ctx.lineTo(canvas.width-20,20);ctx.lineTo(canvas.width-20,20+cSize);ctx.stroke();
  ctx.beginPath();ctx.moveTo(20,canvas.height-20-cSize);ctx.lineTo(20,canvas.height-20);ctx.lineTo(20+cSize,canvas.height-20);ctx.stroke();
  ctx.beginPath();ctx.moveTo(canvas.width-20-cSize,canvas.height-20);ctx.lineTo(canvas.width-20,canvas.height-20);ctx.lineTo(canvas.width-20,canvas.height-20-cSize);ctx.stroke();
  const scanY=(frame*0.5)%canvas.height;
  const scanGrad=ctx.createLinearGradient(0,scanY-30,0,scanY+30);
  scanGrad.addColorStop(0,'rgba(255,45,45,0)');scanGrad.addColorStop(0.5,'rgba(255,45,45,0.04)');scanGrad.addColorStop(1,'rgba(255,45,45,0)');
  ctx.fillStyle=scanGrad;ctx.fillRect(0,scanY-30,canvas.width,60);
  frame++;requestAnimationFrame(animate);
}
animate();
const cur=document.getElementById('custom-cursor');
const ring=document.getElementById('custom-cursor-ring');
let mx=0,my=0,rx=0,ry=0;
document.addEventListener('mousemove',e=>{mx=e.clientX;my=e.clientY;cur.style.left=mx+'px';cur.style.top=my+'px';});
setInterval(()=>{rx+=(mx-rx)*.18;ry+=(my-ry)*.18;ring.style.left=rx+'px';ring.style.top=ry+'px';},10);
document.addEventListener('mouseover',e=>{if(e.target.matches('button,a,input,select,label')){cur.style.transform='translate(-50%,-50%) scale(2.5)';ring.style.transform='translate(-50%,-50%) scale(0.5)';}});
document.addEventListener('mouseout',e=>{if(e.target.matches('button,a,input,select,label')){cur.style.transform='translate(-50%,-50%) scale(1)';ring.style.transform='translate(-50%,-50%) scale(1)';}});
</script>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────



if DEMO_MODE:
    st.info("Demo mode is on...")
# ───────────────────────────────────────────────────

if DEMO_MODE:
    st.info("Demo mode is on. Predictions are simulated and ML models are not loaded.")


# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title(f"{PAGE_ICON} {PAGE_TITLE}")
mode = st.sidebar.radio(
    "Select Mode",
    ["Image Detection", "Video Detection", "Severity Assessment",
     "Location Risk Predictor","World Traffic Map", "About"],
)

st.sidebar.markdown("---")
st.sidebar.caption("AI-Powered Real-Time Accident Information System")
# ── ADD THIS LINE HERE ──
sms_settings = render_sms_alert_widget()
# ────────────────────────

# ── Helper: save uploaded file to temp path ───────────────────────────────
def _save_upload(uploaded_file, suffix: str) -> str:
    """Write an UploadedFile to a temp path and return that path."""
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(uploaded_file.read())
    tmp.close()
    return tmp.name


# ═══════════════════════════════════════════════════════════════════════════
# MODE 1 — Image Detection
# ═══════════════════════════════════════════════════════════════════════════
if mode == "Image Detection":
    st.header("Image Accident Detection")
    st.write("Upload an image to classify it as **Accident** or **Non-Accident**.")

    uploaded = st.file_uploader(
        "Choose an image", type=IMAGE_EXTENSIONS, key="img_upload"
    )

    if uploaded is not None:
        tmp_path = _save_upload(uploaded, f".{uploaded.name.split('.')[-1]}")
        with st.spinner("Analysing image..."):
            try:
                if DEMO_MODE:
                    result = _fake_prediction(num_classes=2)
                else:
                    from src.services.image_service import predict_image

                    result = predict_image(tmp_path)
                render_image_result(tmp_path, result)

                # ── ADD FROM HERE ──
                if (result["class_index"] == 0 and
                    result["score"] >= SMS_CONFIDENCE_THRESHOLD  and
                    sms_settings["enabled"] and
                    sms_settings["phone"]):

                    sms_result = send_accident_sms(
                        to_number  = sms_settings["phone"],
                        confidence = result["score"],
                        api_key    = FAST2SMS_API_KEY
                    )
                    render_sms_result(sms_result)
                # ── TO HERE ──

            except FileNotFoundError:
                st.error(
                    f"Image model weights not found at `{IMAGE_MODEL_PATH}`. "
                    "Please train the model first using `python scripts/train_image.py`."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        os.unlink(tmp_path)


# ═══════════════════════════════════════════════════════════════════════════
# MODE 2 — Video Detection
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Video Detection":
    st.header("Video Accident Detection")
    st.write("Upload a video clip to classify it as **Accident** or **Non-Accident**.")

    uploaded = st.file_uploader(
        "Choose a video", type=VIDEO_EXTENSIONS, key="vid_upload"
    )

    if uploaded is not None:
        tmp_path = _save_upload(uploaded, f".{uploaded.name.split('.')[-1]}")
        with st.spinner("Analysing video (sampling frames)..."):
            try:
                if DEMO_MODE:
                    result = _fake_prediction(num_classes=2)
                else:
                    from src.services.video_service import predict_video

                    result = predict_video(tmp_path)
                render_video_result(tmp_path, result)
            except FileNotFoundError:
                st.error(
                    f"Video model weights not found at `{VIDEO_MODEL_PATH}`. "
                    "Please train the model first using `python scripts/train_video.py`."
                )
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        os.unlink(tmp_path)

# ═══════════════════════════════════════════════════════════════════════════
# MODE 3 — severity assessment
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Severity Assessment":
    st.header("Accident Severity Assessment")
    st.write(
        "Upload an image of an accident scene to assess the severity "
        "using a YOLOv8-based model."
    )

    uploaded = st.file_uploader(
        "Choose an image", type=IMAGE_EXTENSIONS, key="sev_upload"
    )

    if uploaded is not None:
        tmp_path = _save_upload(uploaded, f".{uploaded.name.split('.')[-1]}")
        with st.spinner("Running severity analysis..."):
            try:
                if DEMO_MODE:
                    result = _fake_prediction(num_classes=3)
                else:
                    from src.services.severity_service import predict_severity

                    result = predict_severity(tmp_path)
                render_severity_result(tmp_path, result)

                # ── ADD FROM HERE ──
                if (sms_settings["enabled"] and
                    sms_settings["phone"]):

                    sms_result = send_accident_sms(
                        to_number  = sms_settings["phone"],
                        confidence = result["score"],
                        severity   = result.get("label", "Unknown"),
                        api_key    = FAST2SMS_API_KEY
                    )
                    render_sms_result(sms_result)
                # ── TO HERE ──

            except FileNotFoundError:
                st.error(
                    f"Severity model weights not found at `{SEVERITY_MODEL_PATH}`. "
                    "Please train the model first using `python scripts/train_severity.py`."
                )
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Prediction failed: {e}")
        os.unlink(tmp_path)
        
        
        
        # ═══════════════════════════════════════════════════════════════════════════
# MODE 4 — Location Risk Predictor
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "Location Risk Predictor":
    from src.services.location_service import predict_location_risk

    st.header("Location Risk Predictor")
    st.write("Enter any location to get a real-time accident risk prediction.")

    col1, col2 = st.columns([3, 1])
    with col1:
        place = st.text_input(
            "Enter Location",
            placeholder="e.g. NH-48 Delhi, MG Road Bangalore, Bandra Mumbai"
        )
    with col2:
        predict_btn = st.button("PREDICT RISK", use_container_width=True)

    # Quick location buttons
    st.markdown("**Quick Select:**")
    qcol1, qcol2, qcol3, qcol4 = st.columns(4)
    quick_place = None
    with qcol1:
        if st.button("NH-48 Delhi"):      quick_place = "NH-48 Delhi"
    with qcol2:
        if st.button("MG Road Bangalore"): quick_place = "MG Road Bangalore"
    with qcol3:
        if st.button("Marine Drive Mumbai"): quick_place = "Marine Drive Mumbai"
    with qcol4:
        if st.button("Anna Salai Chennai"): quick_place = "Anna Salai Chennai"

    search_place = quick_place or (place if predict_btn and place else None)

    if search_place:
        with st.spinner(f"Analysing risk for {search_place}..."):
            result = predict_location_risk(search_place)

        if not result["success"]:
            st.error(f"Could not find location: {result['error']}")
        else:
            score   = result["score"]
            level   = result["level"]
            color   = result["color"]
            weather = result["weather"]
            time    = result["time"]

            # ── Risk Score Banner ──
            score_color = (
                "#39ff14" if level == "LOW" else
                "#ffd600" if level == "MODERATE" else
                "#ff2d2d"
            )
            st.markdown(f"""
            <div style="background:#0d0d15;border:1px solid {score_color}33;
                        border-left:4px solid {score_color};padding:24px 28px;
                        margin:16px 0;display:flex;align-items:center;gap:28px">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:80px;
                            color:{score_color};line-height:1;
                            text-shadow:0 0 30px {score_color}55">{score}</div>
                <div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:10px;
                                color:#383850;letter-spacing:3px;text-transform:uppercase;
                                margin-bottom:6px">Accident Risk Score / 100</div>
                    <div style="font-family:'JetBrains Mono',monospace;font-size:13px;
                                color:{score_color};letter-spacing:3px;
                                border:1px solid {score_color};padding:4px 16px;
                                display:inline-block">{color} {level} RISK</div>
                    <div style="font-family:'Outfit',sans-serif;font-size:12px;
                                color:#7070a0;margin-top:8px;font-weight:300">
                        📍 {result['address'][:80]}...</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Factor Breakdown ──
            st.markdown("### Risk Breakdown")
            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    label = "Weather Risk",
                    value = f"{result['weather_risk']}%",
                    delta = weather.get("condition", "N/A")
                )
                st.caption(f"🌡 {weather.get('temperature','?')}°C  "
                          f"💨 {weather.get('windspeed','?')} km/h  "
                          f"🌧 {weather.get('precipitation','?')} mm")

            with c2:
                st.metric(
                    label = "Time Risk",
                    value = f"{result['time_risk']}%",
                    delta = time.get("label", "N/A")
                )
                st.caption(f"🕐 Current time: {time.get('hour','?')}:00 hrs")

            with c3:
                st.metric(
                    label = "Location Risk",
                    value = f"{result['hist_risk']}%",
                    delta = "Historical Data"
                )
                factors = result["historical"].get("factors", [])
                if factors:
                    st.caption("⚠ " + " · ".join(factors[:2]))

            # ── Weather Details ──
            st.markdown("### Current Weather at Location")
            wc1, wc2, wc3, wc4 = st.columns(4)
            with wc1: st.metric("Condition",     weather.get("condition",    "N/A"))
            with wc2: st.metric("Temperature",   f"{weather.get('temperature','?')}°C")
            with wc3: st.metric("Wind Speed",    f"{weather.get('windspeed','?')} km/h")
            with wc4: st.metric("Visibility",    f"{weather.get('visibility','?')} m")

            # ── Alert Message ──
            if level == "LOW":
                st.success(f"✓ Low accident risk at this location. Safe to travel.")
            elif level == "MODERATE":
                st.warning(f"⚠ Moderate risk detected. Drive carefully and reduce speed.")
            elif level == "HIGH":
                st.error(f"✕ High accident risk! Avoid if possible or take extreme caution.")
            else:
                st.error(f"🚨 CRITICAL RISK — Do not travel to this location right now!")

            # ── SMS Alert ──
            if level in ["HIGH", "CRITICAL"] and sms_settings["enabled"] and sms_settings["phone"]:
                from src.services.alert_service import send_accident_sms
                sms_result = send_accident_sms(
                    to_number  = sms_settings["phone"],
                    confidence = score / 100,
                    severity   = f"{level} RISK at {search_place}",
                    api_key    = FAST2SMS_API_KEY
                )
                render_sms_result(sms_result)
# ═══════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════
# MODE 5 — World Traffic Map
# ═══════════════════════════════════════════════════════════════════════════
elif mode == "World Traffic Map":
    from src.services.map_service import build_world_map, GLOBAL_ACCIDENT_DATA
    from streamlit_folium import st_folium
    import pandas as pd

    st.header("World Traffic Risk Map")
    st.write("Live accident risk heatmap across major global cities.")

    # ── Controls ──
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        filter_level = st.selectbox(
            "Filter by Risk Level",
            ["All", "Critical (80+)", "High (60-79)",
             "Moderate (40-59)", "Low (<40)"]
        )
    with col2:
        view = st.selectbox(
            "Map View",
            ["World View", "India Focus"]
        )
    with col3:
        st.markdown("<br>", unsafe_allow_html=True)
        refresh = st.button("REFRESH", use_container_width=True)

    # ── Stats row ──
    total    = len(GLOBAL_ACCIDENT_DATA)
    critical = len([d for d in GLOBAL_ACCIDENT_DATA if d["risk"] >= 80])
    high     = len([d for d in GLOBAL_ACCIDENT_DATA if 60 <= d["risk"] < 80])
    low      = len([d for d in GLOBAL_ACCIDENT_DATA if d["risk"] < 40])

    s1, s2, s3, s4 = st.columns(4)
    with s1: st.metric("Total Cities Monitored", total)
    with s2: st.metric("Critical Risk Cities",   critical, delta="score 80+")
    with s3: st.metric("High Risk Cities",        high,     delta="score 60-79")
    with s4: st.metric("Safe Cities",             low,      delta="score <40")

    # ── Map ──
    with st.spinner("Loading map..."):
        m = build_world_map(
            filter_level = filter_level,
            center_india = (view == "India Focus")
        )
        st_folium(m, width=None, height=550, returned_objects=[])

    # ── Legend ──
    st.markdown("""
    <div style="display:flex;gap:20px;margin-top:12px;
                font-family:'JetBrains Mono',monospace;font-size:10px;
                letter-spacing:1.5px">
        <span style="color:#ff2d2d">⬤ CRITICAL (80+)</span>
        <span style="color:#ffd600">⬤ HIGH (60-79)</span>
        <span style="color:#ff6b35">⬤ MODERATE (40-59)</span>
        <span style="color:#39ff14">⬤ LOW (&lt;40)</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Data Table ──
    st.markdown("### City Risk Data")
    df = pd.DataFrame(GLOBAL_ACCIDENT_DATA)
    df = df.rename(columns={
        "city"       : "City",
        "country"    : "Country",
        "risk"       : "Risk Score",
        "deaths_year": "Deaths/Year"
    })
    df = df.sort_values("Risk Score", ascending=False)
    df = df[["City", "Country", "Risk Score", "Deaths/Year"]]

    if filter_level == "Critical (80+)":
        df = df[df["Risk Score"] >= 80]
    elif filter_level == "High (60-79)":
        df = df[(df["Risk Score"] >= 60) & (df["Risk Score"] < 80)]
    elif filter_level == "Moderate (40-59)":
        df = df[(df["Risk Score"] >= 40) & (df["Risk Score"] < 60)]
    elif filter_level == "Low (<40)":
        df = df[df["Risk Score"] < 40]

    st.dataframe(df, use_container_width=True, hide_index=True)
# ABOUT
# ═══════════════════════════════════════════════════════════════════════════
else:
    st.header("About")
    st.markdown(
        """
        ## AI-Powered Real-Time Accident Information System

        This application demonstrates a supervised machine-learning system that
        classifies images and videos as **Accident** or **Non-Accident**, and
        optionally assesses the **severity** of detected accidents.

        ### Models Used

        | Pipeline | Model | Framework |
        |----------|-------|-----------|
        | Image Classification | MobileNetV2 (transfer learning) | TensorFlow / Keras |
        | Video Classification | R3D-18 (3D ResNet) | PyTorch |
        | Severity Assessment | YOLOv8 | Ultralytics |

        ### How It Works

        1. **Image Detection** — A single image is resized to 224×224 and passed
           through a MobileNetV2 backbone with a custom classification head.
        2. **Video Detection** — 16 evenly-spaced frames are sampled from the
           video, resized to 112×112, and fed into a 3D ResNet (R3D-18) that
           captures both spatial and temporal features.
        3. **Severity Assessment** — A YOLOv8 model analyses the accident image
           to classify the severity or detect relevant objects.

        ### Project Structure

        ```
        src/          — Core ML library (models, datasets, services)
        scripts/      — CLI scripts for training, evaluation, inference
        preprocessing/— Data preparation utilities
        configs/      — YAML configuration files
        models/       — Trained model weights
        app/          — This Streamlit application
        tests/        — Test suite
        ```

        ---
        *Developed as a college project for supervised machine learning.*
        """
    )