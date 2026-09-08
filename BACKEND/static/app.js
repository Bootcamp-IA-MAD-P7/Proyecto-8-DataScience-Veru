const form = document.getElementById('prediction-form');
const result = document.getElementById('result');
const needle = document.getElementById('needle');
const gaugeFill = document.getElementById('gauge-fill');
const probValue = document.getElementById('prob-value');
const verdict = document.getElementById('verdict');

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const payload = {
    age: Number(document.getElementById('age').value),
    avg_glucose_level: Number(document.getElementById('avg_glucose_level').value),
    bmi: Number(document.getElementById('bmi').value),
    gender: document.getElementById('gender').value,
    ever_married: document.getElementById('ever_married').value,
    work_type: document.getElementById('work_type').value,
    residence_type: document.getElementById('residence_type').value,
    smoking_status: document.getElementById('smoking_status').value,
    hypertension: document.getElementById('hypertension').checked ? 1 : 0,
    heart_disease: document.getElementById('heart_disease').checked ? 1 : 0,
  };

  const btn = form.querySelector('button');
  btn.disabled = true;
  btn.textContent = '…';
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();
    const pct = Math.round(data.probabilidad_ictus * 100);
    const deg = pct * 1.8; // 0-100% -> 0-180°
    needle.style.transform = `translateX(-50%) rotate(${deg}deg)`;
    gaugeFill.style.transform = `rotate(${deg}deg)`;
    probValue.textContent = pct + '%';
    verdict.textContent = data.riesgo === 'ALTO' ? '⚠️ RIESGO ALTO' : '✅ RIESGO BAJO';
    verdict.className = 'verdict ' + (data.riesgo === 'ALTO' ? 'alto' : 'bajo');
    result.hidden = false;
    loadHistory();
  } catch (err) {
    alert('Error en la predicción: ' + err.message);
  } finally {
    btn.disabled = false;
    btn.textContent = '🔮 Predecir riesgo';
  }
});

async function loadHistory() {
  const tbody = document.getElementById('history-body');
  try {
    const r = await fetch('/predictions');
    if (!r.ok) throw new Error();
    const rows = await r.json();
    if (!rows.length) {
      tbody.innerHTML = '<tr><td colspan="8">Aún no hay predicciones.</td></tr>';
      return;
    }
    tbody.innerHTML = rows.map((p) => `
      <tr>
        <td>${p.id}</td>
        <td>${p.age}</td>
        <td>${p.gender}</td>
        <td>${p.avg_glucose_level}</td>
        <td>${p.bmi}</td>
        <td>${(p.probabilidad * 100).toFixed(1)}%</td>
        <td><span class="risk-badge ${p.riesgo === 'ALTO' ? 'alto' : 'bajo'}">${p.riesgo}</span></td>
        <td>${formatDate(p.created_at)}</td>
      </tr>`).join('');
  } catch {
    tbody.innerHTML = '<tr><td colspan="8">No se pudo cargar el historial (¿BD disponible?).</td></tr>';
  }
}

function formatDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString('es-ES', {
    day: '2-digit', month: '2-digit', year: '2-digit',
    hour: '2-digit', minute: '2-digit',
  });
}

document.getElementById('refresh').addEventListener('click', loadHistory);
loadHistory();