import { useEffect, useMemo, useState } from 'react'

const money = value => value == null ? '—' : new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(value)
const pct = value => value == null ? '—' : `${(Number(value) * 100).toFixed(2)}%`
const pretty = value => String(value || '').replaceAll('_', ' ')

function LineChart({ rows, series, height = 220, xKey = 'date', format = money }) {
  const width = 760, pad = 28
  const values = rows.flatMap(row => series.map(s => Number(row[s.key])).filter(Number.isFinite))
  if (!rows.length || !values.length) return <div className="chart-empty">No observations available.</div>
  let min = Math.min(...values), max = Math.max(...values)
  if (max === min) max = min + 1
  const x = i => pad + i * (width - 2 * pad) / Math.max(1, rows.length - 1)
  const y = v => height - pad - (v - min) * (height - 2 * pad) / (max - min)
  return <div className="chart-scroll"><svg viewBox={`0 0 ${width} ${height}`} className="line-chart" role="img">
    {[0, .5, 1].map(f => <line key={f} x1={pad} x2={width-pad} y1={pad + f*(height-2*pad)} y2={pad + f*(height-2*pad)} className="gridline" />)}
    {series.map(s => <polyline key={s.key} fill="none" stroke={s.color} strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" points={rows.map((r,i)=>`${x(i)},${y(Number(r[s.key]))}`).filter(p=>p.includes('NaN')===false).join(' ')} />)}
    <text x={pad} y={height-5} className="axis-label">{rows[0][xKey]}</text><text x={width-pad} y={height-5} textAnchor="end" className="axis-label">{rows[rows.length-1][xKey]}</text>
    <text x={pad} y={15} className="axis-label">{format(max)}</text><text x={pad} y={height-30} className="axis-label">{format(min)}</text>
  </svg><div className="legend">{series.map(s=><span key={s.key}><i style={{background:s.color}} />{s.label}</span>)}</div></div>
}

function MetricCard({ label, value, sub }) { return <article className="metric-card"><span>{label}</span><strong>{value}</strong>{sub && <small>{sub}</small>}</article> }
function Panel({ id, title, subtitle, children, action }) { return <section id={id} className="panel"><div className="panel-heading"><div><h2>{title}</h2>{subtitle && <p>{subtitle}</p>}</div>{action}</div>{children}</section> }

export default function App() {
  const [market, setMarket] = useState(null), [performance, setPerformance] = useState(null)
  const [evaluation, setEvaluation] = useState(null), [explain, setExplain] = useState(null)
  const [compare, setCompare] = useState(null), [objective, setObjective] = useState('price_rmse')
  const [model, setModel] = useState('auto'), [horizon, setHorizon] = useState(1)
  const [busy, setBusy] = useState(false), [error, setError] = useState('')
  const [period, setPeriod] = useState(365)

  async function get(url) { const r=await fetch(url); const j=await r.json(); if(!r.ok) throw new Error(j.detail||`Request failed: ${url}`); return j }
  async function refresh(){
    setError('')
    try { const [m,p,e,x]=await Promise.all([get(`/api/v1/history?days=${period}`),get(`/api/v1/performance?horizon_days=${horizon}`),get('/api/v1/evaluation'),get('/api/v1/explainability')]); setMarket(m);setPerformance(p);setEvaluation(e);setExplain(x) }
    catch(e){setError(e.message)}
  }
  useEffect(()=>{refresh()},[period,horizon])
  useEffect(()=>{if(!compare)runForecast()},[])

  async function runForecast(){
    setBusy(true);setError('')
    try { const r=await fetch('/api/v1/forecast/compare',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({symbol:'BTC-USD',model,objective,horizon_days:Number(horizon)})}); const j=await r.json(); if(!r.ok) throw new Error(j.detail||'Forecast request failed');setCompare(j) }
    catch(e){setError(e.message)} finally {setBusy(false)}
  }

  const evalMetrics = evaluation?.metrics || []
  const testRows = evaluation?.rows || []
  const supportedForecasts = compare?.models?.filter(m=>m.supported) || []
  const forecastChartRows = useMemo(()=>{
    const byDate = new Map()
    for (const modelRow of supportedForecasts) {
      for (const point of modelRow.forecasts || []) {
        const row = byDate.get(point.date) || {date: point.date}
        row[modelRow.model] = point.predicted_close_usd
        byDate.set(point.date, row)
      }
    }
    return [...byDate.values()].sort((a,b)=>a.date.localeCompare(b.date))
  },[compare])
  const scoreRows = useMemo(()=>performance?.records?.filter(r=>r.objective===objective) || [],[performance,objective])
  const evalChartRows = useMemo(()=>testRows.map(r=>({date:r.Target_Date,actual:r.actual_close_usd,arima:r.arima_prediction_usd,xgboost:r.xgboost_prediction_usd,lstm:r.lstm_prediction_usd})),[testRows])

  return <div className="app-shell">
    <aside className="sidebar"><div className="brand-mark">₿</div><div className="brand"><b>Forecast Lab</b><span>BTC · USD</span></div><nav><a className="active" href="#overview">Overview</a><a href="#models">Model routing</a><a href="#evaluation">Evaluation</a><a href="#explain">Explainability</a></nav><div className="side-note">Training 2018–2023<br/>Validation 2024<br/>Final fit through 2025<br/>2026 held out</div><a className="api-link" href="/docs" target="_blank">FastAPI docs ↗</a></aside>
    <main className="main-content"><header className="topbar"><div><span className="eyebrow">MODEL-SPECIALIZED FORECASTING</span><h1>Bitcoin Forecasting Lab</h1><p>Compare independent models. Route by objective and horizon using validation evidence.</p></div><div className="top-actions"><label>History<select value={period} onChange={e=>setPeriod(Number(e.target.value))}><option value="90">90 days</option><option value="365">1 year</option><option value="1095">3 years</option></select></label><span className="status-chip">{market?.latest_date ? `Data through ${market.latest_date}` : 'Loading data'}</span></div></header>
    {error&&<div className="error-banner">{error}</div>}
    <div id="overview" className="metric-grid"><MetricCard label="Latest BTC close" value={money(market?.latest_close_usd)} sub={market?.latest_date}/><MetricCard label="Daily return" value={pct(market?.latest_return)} sub="Latest observed candle"/><MetricCard label="30-day annualized volatility" value={pct(market?.latest_annualized_volatility)} sub="Historical estimate"/><MetricCard label="Volume" value={market?.latest_volume?`${(market.latest_volume/1e9).toFixed(2)}B`: '—'} sub="Latest candle · BTC units"/></div>
    <div className="dashboard-grid">
      <Panel title="BTC price history" subtitle="Daily closing price from the configured historical dataset."><LineChart rows={market?.points||[]} series={[{key:'close_usd',label:'Close',color:'#e5a834'}]} /></Panel>
      <Panel title="Trading volume" subtitle="Daily BTC volume as recorded in the market data file."><LineChart rows={market?.points||[]} format={v=>Number(v).toLocaleString(undefined,{notation:'compact',maximumFractionDigits:1})} series={[{key:'volume',label:'Volume',color:'#37b7a5'}]} /></Panel>
      <Panel title="Returns and volatility" subtitle="Daily percentage returns and trailing 30-day annualized volatility."><LineChart rows={market?.points||[]} format={pct} series={[{key:'return_1d',label:'Daily return',color:'#e5a834'},{key:'annualized_volatility_30d',label:'30d volatility',color:'#8b7cf6'}]} /></Panel>
    </div>
    <Panel title="Forecast router" subtitle="Automatic routing uses saved 2024 validation records only. Manual selection is always available.">
      <div id="models" className="form-grid"><label>Objective<select value={objective} onChange={e=>setObjective(e.target.value)}><option value="price_rmse">Minimize price RMSE</option><option value="price_mae">Minimize price MAE</option><option value="directional_accuracy">Maximize directional accuracy</option></select></label><label>Horizon<select value={horizon} onChange={e=>setHorizon(Number(e.target.value))}><option value="1">1 day</option><option value="7">7 days</option><option value="14">14 days</option><option value="30">30 days</option></select></label><label>Model<select value={model} onChange={e=>setModel(e.target.value)}><option value="auto">Automatic</option><option value="persistence">Persistence baseline</option><option value="arima">ARIMA</option><option value="xgboost">XGBoost</option><option value="lstm">LSTM</option></select></label><button className="primary-button" disabled={busy} onClick={runForecast}>{busy?'Working…':'Compare forecasts'}</button></div>
      {compare?.selection&&<div className="selection-callout"><div className="selected-model">{compare.selected_model}</div><div><b>Why this model</b><p>{compare.selection.reason}</p><small>Decision metric: {compare.selection.metric||'manual'} · 2024 validation · {compare.horizon_days}-day horizon</small></div></div>}
      {!compare?.selection&&compare&&<div className="neutral-callout">No automatic route exists for {objective} at {horizon} days because this horizon has not been validated. Manual forecasts are shown where the model supports them.</div>}
      <div className="forecast-cards">{(compare?.models||[]).map(row=>{const first=row.forecasts?.[0],last=row.forecasts?.at(-1);return <article className="forecast-card" key={row.model}><div className="forecast-title"><h3>{row.model}</h3><span className={row.supported?'tag good':'tag'}>{row.supported?'Ready':'Unavailable'}</span></div><strong>{first?money(first.predicted_close_usd):'—'}</strong><small>{first?`${first.date}${last?.date!==first.date?` to ${last?.date}`:''}`:row.detail}</small>{row.validation_metrics&&<span className="muted-line">2024 RMSE {money(row.validation_metrics.rmse)}</span>}</article>})}</div>
      {forecastChartRows.length>0&&<div className="chart-block"><h3>Forecast comparison from the same origin</h3><LineChart rows={forecastChartRows} series={[{key:'persistence',label:'Persistence',color:'#9aa8b5'},{key:'arima',label:'ARIMA',color:'#37b7a5'},{key:'xgboost',label:'XGBoost',color:'#8b7cf6'},{key:'lstm',label:'LSTM',color:'#e5a834'}]} /> </div>}
      <div className="ensemble-note"><b>Combined forecast</b><span>{compare?.combined_forecast? 'Validated ensemble enabled.' : compare?.combined_forecast_reason || 'Waiting for model comparison.'}</span></div>
    </Panel>
    <Panel title="Validation performance by horizon" subtitle="No 7/14/30-day validation records are currently saved. The router will not extrapolate a one-day winner.">
      <div className="table-wrap"><table><thead><tr><th>Model</th><th>Horizon</th><th>MAE</th><th>RMSE</th><th>Direction</th><th>Validation period</th></tr></thead><tbody>{scoreRows.length?scoreRows.map((r,i)=><tr key={`${r.model}-${i}`}><td className="capitalize">{r.model}</td><td>{r.horizon_days} day</td><td>{money(r.mae)}</td><td>{money(r.rmse)}</td><td>{pct(r.directional_accuracy)}</td><td>{r.validation_period}</td></tr>):<tr><td colSpan="6" className="empty-cell">No validation results recorded for this objective and horizon.</td></tr>}</tbody></table></div>
    </Panel>
    <Panel id="evaluation" title="Final out-of-sample evaluation" subtitle="Frozen saved 2026 results for reporting only. These results are not inputs to the router.">
      {evaluation?.available?<><div className="test-banner">Final test: {evaluation.period?.[0]} through {evaluation.period?.[1]} · {testRows.length} saved forecasts</div><div className="metric-grid test-metrics">{evalMetrics.map(m=><MetricCard key={m.model} label={`${m.model} · MAE`} value={money(m.mae)} sub={`RMSE ${money(m.rmse)} · Direction ${pct(m.directional_accuracy)}`}/>)}</div><LineChart rows={evalChartRows} xKey="Target_Date" series={[{key:'actual',label:'Actual',color:'#142a43'},{key:'arima',label:'ARIMA',color:'#37b7a5'},{key:'xgboost',label:'XGBoost',color:'#8b7cf6'},{key:'lstm',label:'LSTM',color:'#e5a834'}]}/></>:<div className="chart-empty">{evaluation?.reason||'Saved evaluation artifacts unavailable.'}</div>}
    </Panel>
    <Panel id="explain" title="Model explainability" subtitle="Settings and diagnostics are read from saved model runs; no values are generated for display.">
      <div className="explain-grid">{Object.entries(explain?.models||{}).map(([name,details])=><article className="explain-card" key={name}><h3>{name}</h3>{details.available?<><p className="muted-line">Artifact run: {details.run_id}</p><p><b>Configuration</b></p><pre>{JSON.stringify(details.parameters,null,2)}</pre><details><summary>Input columns ({details.feature_columns?.length||0})</summary><p>{details.feature_columns?.join(', ')}</p></details>{details.stationarity_tests&&<details open><summary>Stationarity diagnostics</summary><pre>{JSON.stringify(details.stationarity_tests,null,2)}</pre></details>}{details.fitted_coefficients&&<details><summary>Fitted coefficients</summary><pre>{JSON.stringify(details.fitted_coefficients,null,2)}</pre></details>}{details.residual_diagnostics_image&&<details open><summary>Saved residual diagnostics</summary><img className="diagnostic-image" src={details.residual_diagnostics_image} alt="Saved ARIMA residual diagnostic plots" /></details>}{details.feature_importance&&<details open><summary>XGBoost feature importance</summary><pre>{JSON.stringify(details.feature_importance.slice(0,12),null,2)}</pre></details>}{details.training_loss_per_epoch&&<details open><summary>LSTM training log</summary><pre>{JSON.stringify(details.training_loss_per_epoch.slice(-8),null,2)}</pre></details>}</>:<p>{details.reason}</p>}</article>)}</div>
    </Panel>
    <footer>Research interface · forecasts are model outputs, not investment advice · validation and final evaluation remain separate.</footer>
    </main>
  </div>
}
