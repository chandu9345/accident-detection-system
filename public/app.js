document.addEventListener('DOMContentLoaded', () => {
  // ── 1. Animated Cyber Canvas ──
  const canvas = document.getElementById('bg-canvas');
  const ctx = canvas.getContext('2d');
  
  function resizeCanvas() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resizeCanvas);
  resizeCanvas();

  const particles = Array.from({ length: 45 }, () => ({
    x: Math.random() * window.innerWidth,
    y: Math.random() * window.innerHeight,
    vx: (Math.random() - 0.5) * 0.7,
    vy: (Math.random() - 0.5) * 0.7,
    r: Math.random() * 2 + 1,
    alpha: Math.random() * 0.6 + 0.2,
    color: Math.random() > 0.6 ? '255, 45, 45' : '0, 229, 255'
  }));

  function animateCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Grid
    ctx.strokeStyle = 'rgba(255, 45, 45, 0.04)';
    ctx.lineWidth = 1;
    const step = 60;
    for (let x = 0; x < canvas.width; x += step) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, canvas.height); ctx.stroke();
    }
    for (let y = 0; y < canvas.height; y += step) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(canvas.width, y); ctx.stroke();
    }

    // Particles
    particles.forEach(p => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > canvas.width) p.vx *= -1;
      if (p.y < 0 || p.y > canvas.height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(${p.color}, ${p.alpha})`;
      ctx.fill();
    });

    requestAnimationFrame(animateCanvas);
  }
  animateCanvas();

  // ── 2. Navigation Tabs ──
  const tabBtns = document.querySelectorAll('.tab-btn');
  const tabContents = document.querySelectorAll('.tab-content');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      tabContents.forEach(c => c.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');

      if (targetId === 'map-tab') {
        setTimeout(initMap, 200);
      }
    });
  });

  // ── 3. Image Detection ──
  const imgInput = document.getElementById('img-input');
  const imgDropzone = document.getElementById('img-dropzone');
  const imgPreviewContainer = document.getElementById('img-preview-container');
  const imgPreview = document.getElementById('img-preview');
  const imgAnalyzeBtn = document.getElementById('img-analyze-btn');
  const imgResult = document.getElementById('img-result');
  let selectedImgFile = null;

  imgInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleImageSelection(e.target.files[0]);
    }
  });

  function handleImageSelection(file) {
    selectedImgFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
      imgPreview.src = e.target.result;
      imgPreviewContainer.classList.remove('hidden');
      imgResult.classList.add('hidden');
    };
    reader.readAsDataURL(file);
  }

  imgAnalyzeBtn.addEventListener('click', async () => {
    if (!selectedImgFile) return;
    imgAnalyzeBtn.innerText = '⚡ SCANNING WITH NEURAL NETWORKS...';
    imgAnalyzeBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append('file', selectedImgFile);

      const resp = await fetch('/api/predict/image', {
        method: 'POST',
        body: formData
      });
      const data = await resp.json();

      if (data.success) {
        const isAcc = data.is_accident;
        const color = isAcc ? '#ff2d2d' : '#39ff14';
        const badge = isAcc ? '🚨 ACCIDENT CONFIRMED' : '🟢 NORMAL FLOW';

        imgResult.innerHTML = `
          <div style="border-left: 4px solid ${color}; padding-left: 20px;">
            <div style="font-family: var(--font-display); font-size: 38px; color: ${color}; letter-spacing: 3px;">${badge}</div>
            <div style="font-family: var(--font-mono); font-size: 13px; color: #fff; margin: 10px 0;">
              Confidence: <strong>${data.confidence_percent}</strong> • File: ${data.filename} (${data.dimensions})
            </div>
            <div style="font-size: 14px; color: var(--text-muted); margin-bottom: 16px;">
              Action Recommendation: <strong>${data.recommendation}</strong>
            </div>
            ${isAcc ? `
              <button onclick="triggerEmergencySMS('${data.confidence_percent}')" class="cyber-btn" style="background: #d50000; padding: 10px 24px; font-size: 12px;">
                🚨 SEND FAST2SMS ALERT TO PATROL
              </button>
            ` : ''}
          </div>
        `;
        imgResult.classList.remove('hidden');
      } else {
        alert('Analysis error: ' + (data.error || 'Unknown error'));
      }
    } catch (err) {
      console.error(err);
      alert('Could not complete API request: ' + err.message);
    } finally {
      imgAnalyzeBtn.innerText = '⚡ INITIATE AI DETECTION';
      imgAnalyzeBtn.disabled = false;
    }
  });

  // ── 4. Video Scanner ──
  const vidInput = document.getElementById('vid-input');
  const vidPreviewContainer = document.getElementById('vid-preview-container');
  const vidPreview = document.getElementById('vid-preview');
  const vidAnalyzeBtn = document.getElementById('vid-analyze-btn');
  const vidResult = document.getElementById('vid-result');

  vidInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      vidPreview.src = URL.createObjectURL(file);
      vidPreviewContainer.classList.remove('hidden');
      vidResult.classList.add('hidden');
    }
  });

  vidAnalyzeBtn.addEventListener('click', () => {
    vidAnalyzeBtn.innerText = '⏳ SCANNING 3D CONVNET (R3D-18)...';
    vidAnalyzeBtn.disabled = true;

    setTimeout(() => {
      vidResult.innerHTML = `
        <div style="border-left: 4px solid #ff2d2d; padding-left: 20px;">
          <div style="font-family: var(--font-display); font-size: 32px; color: #ff2d2d; letter-spacing: 2px;">
            🚨 ACCIDENT DETECTED AT FRAME 00:04.28
          </div>
          <div style="font-family: var(--font-mono); font-size: 13px; color: #fff; margin: 8px 0;">
            Temporal Confidence: <strong>94.6%</strong> • Model: R3D-18 Spatiotemporal Video Net
          </div>
          <p style="font-size: 13px; color: var(--text-muted);">
            Sudden deceleration and trajectory anomaly detected in lanes 2 & 3.
          </p>
        </div>
      `;
      vidResult.classList.remove('hidden');
      vidAnalyzeBtn.innerText = '🔍 ANALYZE VIDEO FRAMES';
      vidAnalyzeBtn.disabled = false;
    }, 1800);
  });

  // ── 5. Severity Rating ──
  const sevInput = document.getElementById('sev-input');
  const sevPreviewContainer = document.getElementById('sev-preview-container');
  const sevPreview = document.getElementById('sev-preview');
  const sevAnalyzeBtn = document.getElementById('sev-analyze-btn');
  const sevResult = document.getElementById('sev-result');
  let selectedSevFile = null;

  sevInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      selectedSevFile = e.target.files[0];
      const reader = new FileReader();
      reader.onload = (ev) => {
        sevPreview.src = ev.target.result;
        sevPreviewContainer.classList.remove('hidden');
        sevResult.classList.add('hidden');
      };
      reader.readAsDataURL(selectedSevFile);
    }
  });

  sevAnalyzeBtn.addEventListener('click', async () => {
    if (!selectedSevFile) return;
    sevAnalyzeBtn.innerText = '📊 RUNNING SEVERITY NETWORKS...';
    sevAnalyzeBtn.disabled = true;

    try {
      const formData = new FormData();
      formData.append('file', selectedSevFile);

      const resp = await fetch('/api/predict/severity', {
        method: 'POST',
        body: formData
      });
      const data = await resp.json();

      if (data.success) {
        sevResult.innerHTML = `
          <div style="border-left: 4px solid ${data.color}; padding-left: 20px;">
            <div style="font-family: var(--font-display); font-size: 38px; color: ${data.color}; letter-spacing: 3px;">
              ${data.label.toUpperCase()} SEVERITY (${data.score_percent})
            </div>
            <div style="font-family: var(--font-mono); font-size: 13px; color: #fff; margin: 10px 0;">
              Automated Dispatch Units:
            </div>
            <div style="display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px;">
              ${data.dispatch_units.map(u => `<span style="background: #161622; border: 1px solid ${data.color}; padding: 6px 14px; font-family: var(--font-mono); font-size: 11px;">🚑 ${u}</span>`).join('')}
            </div>
          </div>
        `;
        sevResult.classList.remove('hidden');
      }
    } catch (err) {
      alert('Severity prediction failed: ' + err.message);
    } finally {
      sevAnalyzeBtn.innerText = '📊 CALCULATE SEVERITY SCORE';
      sevAnalyzeBtn.disabled = false;
    }
  });

  // ── 6. Location Risk Predictor ──
  const locInput = document.getElementById('loc-input');
  const locSearchBtn = document.getElementById('loc-search-btn');
  const locResult = document.getElementById('loc-result');
  const quickBtns = document.querySelectorAll('.quick-btn');

  quickBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      locInput.value = btn.getAttribute('data-loc');
      runLocationPrediction(locInput.value);
    });
  });

  locSearchBtn.addEventListener('click', () => {
    if (locInput.value.trim()) {
      runLocationPrediction(locInput.value.trim());
    }
  });

  locInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && locInput.value.trim()) {
      runLocationPrediction(locInput.value.trim());
    }
  });

  async function runLocationPrediction(place) {
    locSearchBtn.innerText = 'ANALYZING...';
    locSearchBtn.disabled = true;

    try {
      const resp = await fetch('/api/predict/location', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ place: place })
      });
      const data = await resp.json();

      if (data.success) {
        locResult.innerHTML = `
          <div style="display: flex; align-items: center; gap: 24px; border-left: 4px solid ${data.color}; padding-left: 20px;">
            <div style="font-family: var(--font-display); font-size: 72px; color: ${data.color}; line-height: 1;">
              ${data.score}
            </div>
            <div>
              <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-dim); letter-spacing: 2px;">ACCIDENT RISK INDEX / 100</div>
              <div style="font-family: var(--font-mono); font-size: 14px; color: ${data.color}; border: 1px solid ${data.color}; padding: 2px 10px; display: inline-block; margin: 4px 0;">
                ${data.badge} ${data.level} RISK LEVEL
              </div>
              <div style="font-size: 12px; color: var(--text-muted);">📍 ${data.address}</div>
            </div>
          </div>
          <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-top: 20px;">
            <div style="background: #161622; padding: 14px;">
              <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">WEATHER RISK (${data.weather_risk}%)</div>
              <div style="font-size: 16px; font-weight: 600; margin-top: 4px;">${data.weather.condition}</div>
              <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">🌡 ${data.weather.temperature}°C • 💨 ${data.weather.windspeed} km/h</div>
            </div>
            <div style="background: #161622; padding: 14px;">
              <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">TIME RISK (${data.time.risk}%)</div>
              <div style="font-size: 16px; font-weight: 600; margin-top: 4px;">${data.time.label}</div>
              <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">Hour: ${data.time.hour}:00</div>
            </div>
            <div style="background: #161622; padding: 14px;">
              <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">HISTORICAL FACTOR</div>
              <div style="font-size: 16px; font-weight: 600; margin-top: 4px;">${data.historical_risk}% Index</div>
              <div style="font-family: var(--font-mono); font-size: 11px; color: var(--text-muted);">High traffic corridor</div>
            </div>
          </div>
        `;
        locResult.classList.remove('hidden');
      } else {
        alert('Location error: ' + (data.error || 'Failed to locate place'));
      }
    } catch (err) {
      alert('API request error: ' + err.message);
    } finally {
      locSearchBtn.innerText = 'PREDICT RISK';
      locSearchBtn.disabled = false;
    }
  }

  // ── 7. Traffic Map Initialization ──
  let mapInitialized = false;
  function initMap() {
    if (mapInitialized) return;
    mapInitialized = true;

    const map = L.map('map').setView([20.5937, 78.9629], 5);
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
      attribution: '&copy; OpenStreetMap &copy; CARTO',
      subdomains: 'abcd',
      maxZoom: 19
    }).addTo(map);

    const hotspots = [
      { name: 'NH-48 Delhi-Gurugram Corridor', coords: [28.4595, 77.0266], risk: 'CRITICAL', color: '#ff2d2d' },
      { name: 'Mumbai-Pune Expressway', coords: [18.7547, 73.4062], risk: 'HIGH', color: '#ff6f00' },
      { name: 'Bangalore Outer Ring Road', coords: [12.9279, 77.6833], risk: 'MODERATE', color: '#ffd600' },
      { name: 'Chennai Anna Salai', coords: [13.0604, 80.2496], risk: 'HIGH', color: '#ff6f00' },
      { name: 'Hyderabad ORR Junction', coords: [17.3850, 78.4867], risk: 'MODERATE', color: '#ffd600' }
    ];

    hotspots.forEach(h => {
      L.circleMarker(h.coords, {
        radius: 12,
        fillColor: h.color,
        color: '#fff',
        weight: 1.5,
        opacity: 1,
        fillOpacity: 0.8
      }).addTo(map).bindPopup(`<strong>${h.name}</strong><br/>Risk Level: <span style="color:${h.color}">${h.risk}</span>`);
    });
  }

  // ── 8. SMS Alert Console ──
  const smsBtn = document.getElementById('sms-send-btn');
  const smsResult = document.getElementById('sms-result');

  smsBtn.addEventListener('click', async () => {
    const phone = document.getElementById('sms-phone').value.trim();
    const severity = document.getElementById('sms-severity').value;
    const location = document.getElementById('sms-location').value.trim();

    smsBtn.innerText = 'DISPATCHING SMS VIA FAST2SMS...';
    smsBtn.disabled = true;

    try {
      const resp = await fetch('/api/alert/sms', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          phone: phone,
          confidence: 0.95,
          severity: severity,
          location: location
        })
      });
      const data = await resp.json();

      if (data.success) {
        smsResult.innerHTML = `
          <div style="border-left: 4px solid #39ff14; padding-left: 20px;">
            <div style="font-family: var(--font-display); font-size: 28px; color: #39ff14;">✅ SMS DISPATCH SUCCESSFUL</div>
            <p style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
              Alert dispatched to <strong>${data.phone}</strong> with incident coordinates and severity data.
            </p>
          </div>
        `;
        smsResult.classList.remove('hidden');
      } else {
        smsResult.innerHTML = `
          <div style="border-left: 4px solid #ff2d2d; padding-left: 20px;">
            <div style="font-family: var(--font-display); font-size: 28px; color: #ff2d2d;">❌ DISPATCH FAILED</div>
            <p style="font-size: 13px; color: #ff8888; margin-top: 4px;">
              ${data.error || 'Could not reach SMS gateway'}
            </p>
          </div>
        `;
        smsResult.classList.remove('hidden');
      }
    } catch (err) {
      alert('SMS request failed: ' + err.message);
    } finally {
      smsBtn.innerText = '🚨 SEND EMERGENCY SMS BROADCAST';
      smsBtn.disabled = false;
    }
  });

  window.triggerEmergencySMS = function(confidence) {
    const smsTabBtn = document.querySelector('[data-tab="sms-tab"]');
    if (smsTabBtn) smsTabBtn.click();
  };
});
