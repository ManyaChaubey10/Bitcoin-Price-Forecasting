"""Small browser interface for the forecasting API."""

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

web_router = APIRouter()


@web_router.get("/", response_class=HTMLResponse, include_in_schema=False)
def forecasting_home() -> str:
    """Serve the lightweight project UI without mixing presentation into APIs."""
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <title>BTC Forecasting Lab</title>
  <style>
    :root{font-family:Inter,Segoe UI,Arial,sans-serif;color:#18212f;background:#f3f6fa}
    body{margin:0}.wrap{max-width:1120px;margin:36px auto;padding:0 22px}
    header{background:#10243c;color:white;border-radius:18px;padding:28px 32px}
    h1{margin:0 0 8px;font-size:30px}header p{margin:0;color:#d2deea}
    section{background:#fff;border:1px solid #dfe6ee;border-radius:14px;padding:22px;margin:18px 0}
    .controls{display:flex;gap:14px;flex-wrap:wrap;align-items:end}
    label{display:grid;gap:6px;font-weight:600;font-size:13px}select,input,button{font:inherit;padding:10px 12px;border:1px solid #c7d1dd;border-radius:8px;background:white}
    button{background:#1769aa;border-color:#1769aa;color:white;font-weight:700;cursor:pointer}button:disabled{opacity:.55;cursor:wait}
    .muted{color:#66778a}.reason{border-left:4px solid #1769aa;padding:10px 14px;background:#f2f8fd}
    .grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(225px,1fr));gap:12px}.card{border:1px solid #dfe6ee;border-radius:10px;padding:14px}
    .card h3{margin:0 0 8px;text-transform:capitalize}.price{font-size:22px;font-weight:700}.pill{font-size:12px;padding:3px 8px;border-radius:99px;background:#eaf2fa}
    table{border-collapse:collapse;width:100%;font-size:14px}th,td{text-align:left;padding:9px;border-bottom:1px solid #e7ecf1}pre{white-space:pre-wrap;overflow-wrap:anywhere}
    .error{color:#a12525}.status{margin-left:8px}
  </style>
</head>
<body><div class="wrap">
  <header><h1>BTC Forecasting Lab</h1><p>Compare independent forecasters and route by objective using saved 2024 validation results.</p></header>
  <section><div class="controls">
    <label>Forecast objective<select id="objective"><option value="price_rmse">Minimize price RMSE</option><option value="price_mae">Minimize price MAE</option><option value="directional_accuracy">Maximize directional accuracy</option></select></label>
    <label>Horizon (days)<input id="horizon" type="number" min="1" max="90" value="1"></label>
    <label>Model<select id="model"><option value="auto">Automatic selection</option><option value="persistence">Persistence baseline</option><option value="arima">ARIMA</option><option value="xgboost">XGBoost</option><option value="lstm">LSTM</option></select></label>
    <button id="run">Compare forecasts</button><span id="status" class="status muted"></span>
  </div></section>
  <section><h2>Selected forecast</h2><div id="selection" class="muted">Choose an objective and run a comparison.</div></section>
  <section><h2>Individual forecasts</h2><div id="forecasts" class="grid"></div><p id="ensemble" class="muted"></p></section>
  <section><h2>Validation performance</h2><p class="muted">Scores below come from 2024 validation. They are not 2026 test scores.</p><div id="performance" class="muted">Loading…</div></section>
  <section><h2>Model readiness</h2><div id="readiness" class="grid"></div></section>
</div>
<script>
const $=id=>document.getElementById(id);
function money(x){return Number(x).toLocaleString(undefined,{style:'currency',currency:'USD',maximumFractionDigits:2})}
async function loadInfo(){
  try{
    const [p,m]=await Promise.all([fetch(`/api/v1/performance?horizon_days=${Number($('horizon').value)}`),fetch('/api/v1/models')]);
    if(!p.ok||!m.ok)throw new Error('API did not return model information');
    const perf=await p.json(),models=await m.json();
    const rows=perf.records.filter(r=>r.objective===$('objective').value);
    $('performance').innerHTML=rows.length?'<table><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>Direction</th><th>Validation</th></tr></thead><tbody>'+rows.map(r=>`<tr><td>${r.model}</td><td>${money(r.mae)}</td><td>${money(r.rmse)}</td><td>${(100*r.directional_accuracy).toFixed(2)}%</td><td>${r.validation_period}</td></tr>`).join('')+'</tbody></table>':'No validation record for this horizon.';
    $('readiness').innerHTML=models.models.map(m=>`<div class="card"><h3>${m.name}</h3><span class="pill">${m.ready?'Ready':'Unavailable'}</span><p>${m.detail}</p><small>Validated horizons: ${m.validated_horizons.join(', ')||'none'}</small></div>`).join('');
  }catch(e){$('performance').textContent=e.message;$('readiness').textContent=e.message}
}
async function run(){
  const button=$('run');button.disabled=true;$('status').textContent='Forecasting…';$('selection').textContent='';$('forecasts').replaceChildren();
  try{
    const body={symbol:'BTC-USD',model:$('model').value,objective:$('objective').value,horizon_days:Number($('horizon').value)};
    const response=await fetch('/api/v1/forecast/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
    const data=await response.json();if(!response.ok)throw new Error(data.detail||'Forecast request failed');
    $('status').textContent=`Origin: ${data.history_end}`;
    $('selection').innerHTML=data.selection?`<div class="reason"><strong>${data.selected_model.toUpperCase()}</strong><br>${data.selection.reason}</div>`:'No model has validation evidence for this objective and horizon. Manual forecasts are shown where the model supports them.';
    $('forecasts').innerHTML=data.models.map(m=>{const last=m.forecasts.length?m.forecasts[m.forecasts.length-1]:null;const first=m.forecasts.length?m.forecasts[0]:null;return `<div class="card"><h3>${m.model}</h3><span class="pill">${m.supported?'Forecast available':m.ready?'Not supported':'Unavailable'}</span><p class="price">${first?money(first.predicted_close_usd):'—'}</p><small>${first?first.date+' forecast'+(last&&last.date!==first.date?' … '+last.date:''):m.detail}</small>${m.validation_metrics?`<p class="muted">2024 RMSE ${money(m.validation_metrics.rmse)}</p>`:''}</div>`}).join('');
    $('ensemble').textContent=data.combined_forecast?`Validated combined forecast available.`:`Combined forecast disabled: ${data.combined_forecast_reason}`;
    await loadInfo();
  }catch(e){$('status').textContent=e.message;$('status').className='status error'}finally{button.disabled=false}
}
$('run').addEventListener('click',run);$('objective').addEventListener('change',loadInfo);$('horizon').addEventListener('change',loadInfo);loadInfo();
</script></body></html>"""
