const root = document.getElementById('scene');
const manifest = window.PROJECT_MANIFEST;
const metricMount = document.getElementById('metrics');
const signalMount = document.getElementById('signals');
function metricCard(item){return `<article class="panel"><strong>${item.value}</strong><span>${item.label}</span><p>${item.note}</p></article>`;}
function signalCard(item){return `<article class="panel tilt"><h3>${item.name}</h3><div class="badge ${item.status.toLowerCase()}">${item.status}</div><p>${item.detail}</p></article>`;}
metricMount.innerHTML = manifest.kpis.map(metricCard).join('');
fetch('/api/signals').then(r=>r.json()).then(data=>{signalMount.innerHTML = data.map(signalCard).join('');});
root.addEventListener('pointermove', (event)=>{const x=(event.clientX/window.innerWidth)-0.5; const y=(event.clientY/window.innerHeight)-0.5; root.style.setProperty('--rx', `${y*-10}deg`); root.style.setProperty('--ry', `${x*12}deg`);});
