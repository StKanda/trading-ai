"use strict";
const $ = id => document.getElementById(id);
const state = {symbol:"SPY", timeframe:5, minute:270, leverage:1, mode:"demo", data:null, stocks:[], serial:0, scanSerial:0, player:null};
const money = v => v == null ? "—" : new Intl.NumberFormat("fr-FR", {minimumFractionDigits:2, maximumFractionDigits:2}).format(v);
const pct = v => v == null ? "—" : `${v >= 0 ? "+" : ""}${(v * 100).toFixed(2).replace(".", ",")} %`;
const nyTime = t => new Intl.DateTimeFormat("fr-FR", {timeZone:"America/New_York", hour:"2-digit", minute:"2-digit"}).format(new Date(t));
const names = {SPY:"S&P 500 ETF", QQQ:"Nasdaq 100 ETF", AAPL:"Apple", MSFT:"Microsoft", NVDA:"NVIDIA", TSLA:"Tesla"};
const colors = {teal:"#56c7ab", coral:"#e77f86", amber:"#dbc085", violet:"#a497e4", blue:"#74b4e9", muted:"#6d8094"};
function error(message) { $("error-banner").textContent = message || ""; $("error-banner").hidden = !message; }
async function api(path, options) {
  const response = await fetch(path, options);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || `Erreur HTTP ${response.status}`);
  return data;
}
function query() { const q = new URLSearchParams({symbol:state.symbol, timeframe:state.timeframe, leverage:state.leverage}); if (state.mode === "demo") q.set("minute", state.minute); return q.toString(); }
function node(tag, className, text) { const el = document.createElement(tag); if (className) el.className = className; if (text != null) el.textContent = text; return el; }
function setChange(el, value) { el.textContent = pct(value); el.className = value >= 0 ? "positive" : "negative"; }
function stocksView() {
  const list = $("stock-list"); list.replaceChildren();
  const select = $("symbol-select"); select.replaceChildren();
  for (const item of state.stocks) {
    const opt = node("option", "", `${item.symbol} · ${item.selection_score == null ? "indisponible" : item.selection_score + "/100"}`);
    opt.value = item.symbol; opt.selected = item.symbol === state.symbol; select.append(opt);
    const row = node("button", "stock-row" + (item.symbol === state.symbol ? " selected" : ""));
    row.dataset.symbol = item.symbol; row.setAttribute("aria-label", `Afficher ${item.symbol}`);
    const left = node("span", "stock-name"); left.append(node("strong", "", item.symbol));
    const change = node("small"); if (item.error) {change.textContent = "Données absentes"; row.title = item.error;} else setChange(change, item.session_change); left.append(change);
    const right = node("span", "stock-score", item.selection_score == null ? "—" : String(item.selection_score));
    const meter = node("span", "score-bar"), fill = node("i"); fill.style.width = `${item.selection_score || 0}%`; meter.append(fill); right.append(meter);
    row.append(left, right); row.addEventListener("click", () => selectSymbol(item.symbol)); list.append(row);
  }
}
function selectSymbol(symbol) { state.symbol = symbol; stocksView(); loadAnalysis(); }
async function scan() {
  const serial = ++state.scanSerial;
  $("refresh-button").disabled = true;
  try {
    const result = await api(`/api/scan?${query()}`);
    if (serial !== state.scanSerial) return;
    state.stocks = result.items; stocksView();
  } catch (e) { if (serial === state.scanSerial) error(e.message); }
  finally { if (serial === state.scanSerial) $("refresh-button").disabled = false; }
}
async function loadAnalysis() {
  const serial = ++state.serial;
  error(""); $("chart-loading").hidden = false; $("chart-loading").textContent = "Calcul des bougies et des niveaux…";
  $("model-value").textContent = "Non calculé"; $("model-button").disabled = false; $("model-button").textContent = "Calculer le score RF ↗";
  try {
    const data = await api(`/api/analysis?${query()}`);
    if (serial !== state.serial) return;
    state.data = data; updateView(data); $("chart-loading").hidden = true;
  } catch (e) { if (serial === state.serial) {error(e.message); $("chart-loading").textContent = "Analyse indisponible · vérifier la source de données";} }
}
function updateView(data) {
  const s = data.scanner;
  $("symbol-title").textContent = data.symbol; $("symbol-description").textContent = names[data.symbol] || "Action / ETF";
  $("last-price").textContent = money(s.price); setChange($("session-change"), s.session_change);
  $("session-label").textContent = `Séance du ${new Date(data.session + "T12:00:00Z").toLocaleDateString("fr-FR")} · ${nyTime(data.asof)} NY`;
  $("clock-label").textContent = state.mode === "demo" ? "Prix fabriqués · rejeu de démonstration" : state.mode === "yahoo" ? "Yahoo · délai non garanti · interrogation chaque minute" : "Snapshot CSV · dernière bougie du fichier";
  $("bar-count").textContent = `${data.bars.length} bougies terminées · ${state.timeframe} min`;
  $("chart-status").textContent = s.missing_bars || s.data_lag_minutes ? `Données à vérifier · ${s.missing_bars} trous` : "Bougies terminées";
  $("trend-value").textContent = s.trend; $("trend-value").className = s.trend === "Haussière" ? "positive" : s.trend === "Baissière" ? "negative" : "";
  $("relative-volume").textContent = s.relative_volume == null ? "Non disponible" : `${s.relative_volume.toFixed(2).replace(".", ",")}×`;
  const lastPattern = data.patterns.at(-1);
  $("pattern-value").textContent = lastPattern ? `${lastPattern.pattern} · ${Math.round(lastPattern.quality_score)}/100` : "Aucune figure";
  $("pattern-detail").textContent = lastPattern ? lastPattern.status : "Deux bougies pour confirmer un pivot";
  const plan = data.plan;
  $("plan-entry").textContent = plan ? money(plan.entry_estimate) : "—";
  $("plan-stop").textContent = plan ? money(plan.stop) : "—";
  $("plan-target").textContent = plan ? money(plan.target) : "—";
  $("plan-shares").textContent = plan ? `${plan.shares} actions · ${money(plan.notional)} USD` : "Taille indisponible";
  $("plan-leverage").textContent = plan ? `${plan.leverage.toFixed(2)}× effectif` : "—";
  $("plan-note").textContent = !plan ? "Historique insuffisant pour le scénario." : !plan.data_usable ? "Scénario suspendu : vérifier les données." : !plan.entry_window_open ? "Fenêtre d'entrée fermée pour un horizon de 30 min. Niveaux affichés à titre descriptif." : "Niveaux à recalculer à l'ouverture suivante. Aucun signal de trading validé.";
  $("analysis-text").textContent = data.commentary;
  // Une nouvelle valeur/séance change le contexte : les réponses précédentes ne sont pas réutilisées.
  [...$("chat-messages").children].slice(1).forEach(el => el.remove());
  chart.draw();
}
class PriceChart {
  constructor(canvas) {
    this.canvas = canvas; this.hover = -1;
    new ResizeObserver(() => this.draw()).observe(canvas.parentElement);
    canvas.addEventListener("mousemove", e => { const r = canvas.getBoundingClientRect(); const bars = state.data?.bars || []; this.hover = Math.floor((e.clientX - r.left - 16) / Math.max(1, r.width - 100) * bars.length); this.draw(); this.tooltip(e.clientX - r.left); });
    canvas.addEventListener("mouseleave", () => {this.hover = -1; $("chart-tooltip").hidden = true; this.draw();});
  }
  tooltip() {
    const b = state.data?.bars[this.hover], tip = $("chart-tooltip"); tip.hidden = !b;
    if (b) tip.textContent = `Fin ${nyTime(b.time)} NY · ${state.symbol}\nO ${money(b.open)}   H ${money(b.high)}\nL ${money(b.low)}   C ${money(b.close)}\nVolume ${Math.round(b.volume).toLocaleString("fr-FR")}`;
  }
  draw() {
    const canvas = this.canvas, rect = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    const w = rect.width, h = rect.height; if (!w || !h) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h);
    const data = state.data, bars = data?.bars || []; if (!bars.length) return;
    const left = 16, right = w - 84, top = 28, bottom = h - 104, volumeTop = h - 79, volumeBottom = h - 29;
    let low = Math.min(...bars.map(b => b.low)), high = Math.max(...bars.map(b => b.high));
    const span = Math.max(high - low, high * .001);
    const levels = $("levels-toggle").checked ? data.levels.filter(l => l.price > low - span && l.price < high + span) : [];
    if ($("plan-toggle").checked && data.plan) levels.push({name:"Entrée", price:data.plan.entry_estimate,color:"blue"}, {name:"Stop",price:data.plan.stop,color:"coral"},{name:"Objectif",price:data.plan.target,color:"teal"});
    for (const l of levels) {low = Math.min(low,l.price); high = Math.max(high,l.price);}
    low -= span * .13; high += span * .13;
    const step = (right - left) / bars.length;
    const x = i => left + (i + .5) * step;
    const y = v => bottom - (v - low) / Math.max(high - low, .001) * (bottom - top);
    ctx.font = "10px ui-sans-serif, system-ui"; ctx.lineWidth = 1;
    for (let i=0;i<=5;i++) {const gy = top + (bottom-top)*i/5;ctx.strokeStyle="#202a34";ctx.beginPath();ctx.moveTo(left,gy);ctx.lineTo(right,gy);ctx.stroke();ctx.fillStyle="#788897";ctx.fillText(money(high-(high-low)*i/5),right+10,gy+3);}
    const xtick = Math.max(1,Math.ceil(bars.length/6));
    for (let i=0;i<bars.length;i+=xtick) {ctx.strokeStyle="#1c252e";ctx.beginPath();ctx.moveTo(x(i),top);ctx.lineTo(x(i),volumeBottom);ctx.stroke();ctx.fillStyle="#758391";ctx.textAlign="center";ctx.fillText(nyTime(bars[i].time),x(i),h-10);}
    ctx.textAlign="left";
    const maxVolume = Math.max(...bars.map(b=>b.volume),1);
    bars.forEach((b,i)=>{const up=b.close>=b.open,col=up?colors.teal:colors.coral;ctx.strokeStyle=col;ctx.fillStyle=col;ctx.beginPath();ctx.moveTo(x(i),y(b.high));ctx.lineTo(x(i),y(b.low));ctx.stroke();const cw=Math.max(2,Math.min(9,step*.64));ctx.fillRect(x(i)-cw/2,Math.min(y(b.open),y(b.close)),cw,Math.max(1.2,Math.abs(y(b.open)-y(b.close))));ctx.globalAlpha=.32;ctx.fillRect(x(i)-cw/2,volumeBottom-b.volume/maxVolume*(volumeBottom-volumeTop),cw,b.volume/maxVolume*(volumeBottom-volumeTop));ctx.globalAlpha=1;});
    if ($("vwap-toggle").checked) {ctx.strokeStyle=colors.violet;ctx.lineWidth=1.4;ctx.beginPath();let first=true;bars.forEach((b,i)=>{if(b.vwap==null)return;if(first){ctx.moveTo(x(i),y(b.vwap));first=false;}else ctx.lineTo(x(i),y(b.vwap));});ctx.stroke();ctx.lineWidth=1;}
    let labelYs=[];
    for(const l of levels){const ly=y(l.price);ctx.strokeStyle=colors[l.color];ctx.setLineDash([4,5]);ctx.globalAlpha=.65;ctx.beginPath();ctx.moveTo(left,ly);ctx.lineTo(right,ly);ctx.stroke();ctx.setLineDash([]);ctx.globalAlpha=1;let labelY=ly-5;while(labelYs.some(v=>Math.abs(v-labelY)<14))labelY-=14;labelYs.push(labelY);if(labelY<15)continue;ctx.fillStyle=colors[l.color];ctx.font="9px ui-sans-serif, system-ui";const title=`${l.name} · ${money(l.price)}`;ctx.fillStyle="#11171ee8";ctx.fillRect(left+3,labelY-10,ctx.measureText(title).width+9,13);ctx.fillStyle=colors[l.color];ctx.fillText(title,left+7,labelY);}
    if ($("patterns-toggle").checked) for(const pattern of data.patterns.slice(-3)){const points=pattern.points.map(([t,p])=>[bars.findIndex(b=>b.time===t),p]).filter(([i])=>i>=0);if(points.length!==3)continue;ctx.strokeStyle=pattern.pattern==="W"?colors.teal:colors.coral;ctx.lineWidth=2;ctx.setLineDash(pattern.status==="cassure confirmée"?[]:[3,4]);ctx.beginPath();points.forEach(([i,p],j)=>j?ctx.lineTo(x(i),y(p)):ctx.moveTo(x(i),y(p)));ctx.stroke();ctx.setLineDash([]);ctx.lineWidth=1;ctx.fillStyle=ctx.strokeStyle;ctx.fillText(`${pattern.pattern} ${Math.round(pattern.quality_score)}/100`,x(points[1][0])+5,y(points[1][1])-8);}
    const last=bars.at(-1), py=y(last.close);ctx.strokeStyle="#a8b5c4";ctx.setLineDash([2,4]);ctx.beginPath();ctx.moveTo(x(bars.length-1),py);ctx.lineTo(right,py);ctx.stroke();ctx.setLineDash([]);ctx.fillStyle=last.close>=last.open?colors.teal:colors.coral;ctx.fillRect(right+4,py-10,73,20);ctx.fillStyle="#081713";ctx.font="10px ui-sans-serif, system-ui";ctx.fillText(money(last.close),right+10,py+4);
    if(this.hover>=0&&this.hover<bars.length){ctx.strokeStyle="#556777";ctx.setLineDash([3,4]);ctx.beginPath();ctx.moveTo(x(this.hover),top);ctx.lineTo(x(this.hover),volumeBottom);ctx.stroke();ctx.setLineDash([]);}
    ctx.fillStyle="#637282";ctx.font="9px ui-sans-serif, system-ui";ctx.fillText("VOLUME",left,volumeTop-7);
    if(state.mode==="demo"){ctx.save();ctx.font="600 23px ui-sans-serif, system-ui";ctx.fillStyle="#76818b12";ctx.textAlign="center";ctx.fillText("DÉMONSTRATION SYNTHÉTIQUE",(right+left)/2,(top+bottom)/2);ctx.restore();}
  }
}
const chart = new PriceChart($("price-chart"));
for (const id of ["levels-toggle","patterns-toggle","vwap-toggle","plan-toggle"]) $(id).addEventListener("change",()=>chart.draw());
document.querySelectorAll("[data-timeframe]").forEach(button=>button.addEventListener("click",()=>{state.timeframe=Number(button.dataset.timeframe);document.querySelectorAll("[data-timeframe]").forEach(b=>{const active=Number(b.dataset.timeframe)===state.timeframe;b.classList.toggle("active",active);b.setAttribute("aria-pressed",String(active));});loadAnalysis();scan();}));
$("symbol-select").addEventListener("change",e=>selectSymbol(e.target.value));
$("leverage-select").addEventListener("change",e=>{state.leverage=Number(e.target.value);loadAnalysis();});
$("refresh-button").addEventListener("click",()=>{loadAnalysis();scan();});
function setMinute(value){state.minute=Number(value);$("replay-slider").value=state.minute;const total=570+state.minute;$("replay-time").textContent=`${String(Math.floor(total/60)).padStart(2,"0")}:${String(total%60).padStart(2,"0")}`;}
$("replay-slider").addEventListener("input",e=>setMinute(e.target.value));
$("replay-slider").addEventListener("change",()=>{loadAnalysis();scan();});
$("play-button").addEventListener("click",()=>{if(state.player){clearInterval(state.player);state.player=null;$("play-button").textContent="▶";return;}$("play-button").textContent="Ⅱ";state.player=setInterval(()=>{if(state.minute>=390){clearInterval(state.player);state.player=null;$("play-button").textContent="▶";return;}if(!$("chart-loading").hidden)return;setMinute(state.minute+5);loadAnalysis();if(state.minute%30===0)scan();},2200);});
$("model-button").addEventListener("click",async()=>{const serial=state.serial;$("model-button").disabled=true;$("model-button").textContent="Entraînement sur le passé…";try{const model=await api(`/api/model?${query()}`);if(serial!==state.serial)return;$("model-value").textContent=`${(model.score*100).toFixed(1).replace(".",",")} % · non calibré`;$("model-value").title=`${model.note} Train : ${model.audit.train_rows} lignes, dernière target connue ${model.audit.train_label_end_max}.`;$("model-button").textContent=`RF_A · ${model.audit.train_rows} lignes train`;}catch(e){if(serial===state.serial){error(e.message);$("model-value").textContent="Non disponible";$("model-button").textContent="Réessayer";}}finally{if(serial===state.serial)$("model-button").disabled=false;}});
async function ask(question){if(!question.trim()||!state.data)return;const serial=state.serial;const messages=$("chat-messages");messages.append(node("div","message user-message",question));$("chat-input").value="";$("chat-submit").disabled=true;try{const payload={question,symbol:state.symbol,timeframe:state.timeframe,leverage:state.leverage};if(state.mode==="demo")payload.minute=state.minute;const response=await api("/api/chat",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(payload)});if(serial!==state.serial)return;messages.append(node("div","message assistant-message",response.answer));messages.scrollTop=messages.scrollHeight;}catch(e){error(e.message);}finally{$("chat-submit").disabled=false;}}
$("chat-form").addEventListener("submit",e=>{e.preventDefault();ask($("chat-input").value);});
document.querySelectorAll(".quick-questions button").forEach(b=>b.addEventListener("click",()=>ask(b.textContent)));
$("method-button").addEventListener("click",()=>$("method-dialog").showModal());$("close-dialog").addEventListener("click",()=>$("method-dialog").close());
async function init(){try{const setup=await api("/api/config");state.mode=setup.mode;state.symbol=setup.config.symbols[0];state.leverage=setup.config.risk.max_leverage;$("leverage-select").value=String(state.leverage);$("source-badge").textContent=state.mode==="demo"?"DÉMO SYNTHÉTIQUE":state.mode==="yahoo"?"YAHOO · DÉLAI NON GARANTI":"SNAPSHOT CSV";$("replay-controls").hidden=state.mode!=="demo";$("capital-value").textContent=`${money(setup.config.risk.initial_capital)} USD`;$("trade-risk").textContent=`${money(setup.config.risk.initial_capital*setup.config.risk.risk_per_trade)} USD`;$("daily-risk").textContent=`${money(setup.config.risk.initial_capital*setup.config.risk.daily_loss_limit)} USD`;$("cost-note").textContent=`Hypothèses : frais ${setup.config.risk.fee_bps} pb/côté · glissement ${setup.config.risk.slippage_bps} pb/côté`;state.stocks=setup.config.symbols.map(symbol=>({symbol,selection_score:null}));stocksView();await Promise.all([loadAnalysis(),scan()]);if(state.mode==="yahoo")setInterval(()=>{if(document.visibilityState==="visible"){loadAnalysis();scan();}},60000);}catch(e){error(e.message);}}
init();
