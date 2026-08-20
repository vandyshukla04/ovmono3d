"""Build the AeroView Flight Deck: a single self-contained HTML replay of real survey flights.

    python tools/aeroview/build_flightdeck.py \
        --root /mnt/d/3DBOX/papersubdata \
        --detany3d /home/shuklva/DetAny3D \
        --seg "rhin2/DJI_20250225165243_0018_D/seg2:Rhino orbit" \
        --seg "elep3/DJI_20260227083814_0001_V/seg5:Elephant herd" \
        --out flightdeck.html

PUBLIC RELEASE
--------------
The output is ONE static HTML file with the data embedded. It has no server, no Claude dependency and no
build step for the viewer: open it in any browser, attach it to a paper as supplementary material, or host
it on GitHub Pages. The only network fetch is Google Fonts, with system fallbacks, so it renders offline too.

WHAT IT SHOWS (the north star, made visible)
--------------------------------------------
Per animal, a ring of 24 sectors around its OWN body frame -- amber where the camera has, at any point up to
the scrubber position, observed that side. TWO CONCENTRIC RINGS record the viewing SPHERE, not just the
circle: the outer ring is side views (camera elevation < 35 deg above the animal's plane -- the views that
show a flank and are usable for identification), the inner ring is overhead views (>= 35 deg -- the drone
passing over the top, which shows the back and is low-value for ID). So "the drone flew over the animal" is
recorded, but separately from the coverage that monitoring actually needs.

Headings are label-free (motion-derived from the tracks); when the trained model's predicted alpha replaces
them, the same page becomes the live monitoring console. Positions are in the reconstruction's relative
units; every angle is exact.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def export_segment(seg_dir: Path, name: str, species: str, *, step: int = 2) -> dict:
    from tools.heading.papersub import load_segment  # resolved via --detany3d on sys.path

    s = load_segment(seg_dir)
    cams = dict(sorted(s.cameras.items()))
    tracks = list(s.tracks.values()) if hasattr(s.tracks, "values") else list(s.tracks)

    up = getattr(s, "up_unsigned", None)
    up = np.array([0, 0, 1.0]) if up is None else np.asarray(up, float)
    up = up / np.linalg.norm(up)
    seed = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 0, 1.0])
    e1 = seed - (seed @ up) * up
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    P = lambda X: [round(float(X @ e1), 4), round(float(X @ e2), 4)]
    az = lambda v: float(np.arctan2(v @ e2, v @ e1))

    # per-track heading: smoothed velocity, hold-last when slow
    T = {}
    for t in tracks:
        C = np.asarray(t.centers)
        F = np.asarray(t.frames).astype(int)
        head = np.full(len(F), np.nan)
        spd = np.zeros(len(F))
        for i in range(len(F)):
            j0, j1 = max(0, i - 6), min(len(F) - 1, i + 6)
            d = C[j1] - C[j0]
            d = d - (d @ up) * up
            n = np.linalg.norm(d)
            spd[i] = n / max(F[j1] - F[j0], 1)
            if n > 1e-6:
                head[i] = az(d / n)
        thr = np.percentile(spd[spd > 0], 35) if (spd > 0).any() else 0
        conf = spd >= thr
        last = np.nan
        for i in range(len(F)):
            if conf[i]:
                last = head[i]
            elif not np.isnan(last):
                head[i] = last
                conf[i] = False
        T[getattr(t, "track_id", id(t))] = dict(C=C, F=F, head=head, conf=conf)

    frames = []
    for f in sorted(cams)[::step]:
        cam = cams[f]
        cc = np.asarray(cam.center)
        look = cam.extrinsic[:, :3].T @ np.array([0, 0, 1.0])
        lookg = look - (look @ up) * up
        rows = []
        for tid, tr in T.items():
            m = np.flatnonzero(tr["F"] == f)
            if not len(m):
                continue
            i = int(m[0])
            c = tr["C"][i]
            v = cc - c
            vg = v - (v @ up) * up
            dist = float(np.linalg.norm(v))
            horiz = float(np.linalg.norm(vg))
            elev = float(np.degrees(np.arctan2(max(v @ up, 0.0), max(horiz, 1e-9))))
            h = tr["head"][i]
            ra = None if np.isnan(h) else round(float(((az(vg / max(horiz, 1e-9)) - h) + np.pi) % (2 * np.pi) - np.pi), 3)
            rows.append({"id": int(tid) if isinstance(tid, (int, np.integer)) else 0,
                         "p": P(c), "h": None if np.isnan(h) else round(float(h), 3),
                         "cf": bool(tr["conf"][i]), "ra": ra,
                         "d": round(dist, 3), "el": round(elev, 1)})
        frames.append({"f": int(f), "cam": P(cc),
                       "look": round(az(lookg / max(np.linalg.norm(lookg), 1e-9)), 3), "a": rows})
    return {"name": name, "species": species, "path": str(seg_dir), "frames": frames}


TEMPLATE = r'''<title>AeroView Flight Deck</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,300..900&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --bg:#FAF6ED; --panel:#F1EADC; --panel2:#E9E0CC; --line:#D8CCAF;
  --ink:#241D10; --muted:#6E6046; --faint:#9A8C6E;
  --drone:#0B7D6D; --drone-soft:rgba(11,125,109,.14);
  --cov:#A96A10; --cov-soft:rgba(169,106,16,.18);
  --animal:#5D4F33; --sel:#241D10;
  --grid:rgba(36,29,16,.07); --wedge:rgba(11,125,109,.08);
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --bg:#171209; --panel:#211A0F; --panel2:#2A2213; --line:#3A2F1D;
    --ink:#EDE4D2; --muted:#A6987C; --faint:#7A6C50;
    --drone:#3FC9B4; --drone-soft:rgba(63,201,180,.16);
    --cov:#E5A13D; --cov-soft:rgba(229,161,61,.20);
    --animal:#D8C9A3; --sel:#F5EDDA;
    --grid:rgba(237,228,210,.06); --wedge:rgba(63,201,180,.07);
  }
}
:root[data-theme="dark"]{
  --bg:#171209; --panel:#211A0F; --panel2:#2A2213; --line:#3A2F1D;
  --ink:#EDE4D2; --muted:#A6987C; --faint:#7A6C50;
  --drone:#3FC9B4; --drone-soft:rgba(63,201,180,.16);
  --cov:#E5A13D; --cov-soft:rgba(229,161,61,.20);
  --animal:#D8C9A3; --sel:#F5EDDA;
  --grid:rgba(237,228,210,.06); --wedge:rgba(63,201,180,.07);
}
*{box-sizing:border-box;margin:0}
html,body{height:100%}
body{background:var(--bg);color:var(--ink);font:15px/1.5 Archivo,system-ui,sans-serif;
  display:flex;flex-direction:column;overflow:hidden}
header{display:flex;align-items:baseline;gap:20px;padding:14px 20px 10px;border-bottom:1px solid var(--line)}
h1{font-size:19px;font-weight:800;letter-spacing:-.01em;font-stretch:112%}
h1 span{color:var(--drone)}
.sub{color:var(--muted);font-size:12.5px;max-width:52ch}
nav{margin-left:auto;display:flex;gap:6px}
nav button,#trk{font:600 12.5px Archivo;padding:7px 13px;border:1px solid var(--line);border-radius:99px;
  background:transparent;color:var(--muted);cursor:pointer}
nav button[aria-pressed="true"],#trk[aria-pressed="true"]{background:var(--ink);color:var(--bg);border-color:var(--ink)}
nav button:focus-visible,#trk:focus-visible,.tr button:focus-visible,input:focus-visible{outline:2px solid var(--drone);outline-offset:2px}
main{flex:1;display:flex;min-height:0}
#stage{flex:1;position:relative;min-width:0}
canvas{position:absolute;inset:0;width:100%;height:100%}
#hud{position:absolute;top:12px;left:14px;display:flex;gap:8px;pointer-events:none}
.chip{font:500 11px "IBM Plex Mono",monospace;color:var(--muted);background:var(--panel);
  border:1px solid var(--line);border-radius:6px;padding:4px 9px}
.chip b{color:var(--ink);font-weight:500}
#tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);
  border-radius:8px;padding:7px 10px;font:500 11.5px "IBM Plex Mono",monospace;color:var(--ink);
  display:none;box-shadow:0 4px 16px rgba(0,0,0,.18);white-space:nowrap;z-index:3}
#tip small{display:block;color:var(--muted)}
aside{width:272px;border-left:1px solid var(--line);background:var(--panel);
  overflow-y:auto;padding:12px;display:flex;flex-direction:column;gap:8px}
.lbl{font:600 10.5px Archivo;letter-spacing:.08em;text-transform:uppercase;color:var(--faint);padding:2px 2px 0}
.card{display:flex;align-items:center;gap:11px;background:var(--bg);border:1px solid var(--line);
  border-radius:10px;padding:8px 10px;cursor:pointer}
.card[data-sel="1"]{border-color:var(--cov);box-shadow:inset 0 0 0 1px var(--cov)}
.card svg{flex:none}
.card .nm{font:600 13px Archivo}
.card .dg{font:500 11px "IBM Plex Mono",monospace;color:var(--muted)}
.card .pc{margin-left:auto;font:600 15px Archivo;color:var(--cov);font-variant-numeric:tabular-nums}
#group{background:var(--panel2);border-radius:10px;padding:10px 12px}
#group .big{font:800 26px Archivo;color:var(--cov);font-variant-numeric:tabular-nums;line-height:1.1}
#group .cap{font-size:11.5px;color:var(--muted)}
.tr{display:flex;align-items:center;gap:14px;padding:10px 20px;border-top:1px solid var(--line);background:var(--panel)}
.tr button.pp{width:38px;height:38px;border-radius:50%;border:1px solid var(--line);background:var(--bg);
  color:var(--ink);font-size:15px;cursor:pointer;flex:none}
input[type=range]{flex:1;accent-color:var(--drone);height:4px}
#t{font:500 12px "IBM Plex Mono",monospace;color:var(--muted);min-width:110px;text-align:right;font-variant-numeric:tabular-nums}
.foot{font-size:10.5px;color:var(--faint);padding:0 20px 8px;background:var(--panel)}
@media (max-width:760px){aside{display:none}}
</style>

<header>
  <h1>AeroView <span>Flight Deck</span></h1>
  <p class="sub">Survey replay — each animal's heading, and how much of its viewing sphere the camera has covered: outer ring = side views, inner ring = overhead.</p>
  <nav id="tabs" aria-label="Segment"></nav>
</header>
<main>
  <div id="stage">
    <canvas id="cv"></canvas>
    <div id="hud"></div>
    <div id="tip"></div>
  </div>
  <aside id="rail"></aside>
</main>
<div class="tr">
  <button class="pp" id="play" aria-label="Play or pause">▶</button>
  <input id="scrub" type="range" min="0" value="0" step="1" aria-label="Timeline">
  <button id="trk" aria-pressed="false" title="Show the drone's flight track">Track</button>
  <div id="t"></div>
</div>
<p class="foot">Real WildBox segments · positions in reconstruction units · headings are label-free (motion-derived) · ring sectors turn amber once the camera has observed that side of the animal — outer ring: side views (&lt;35° elevation, the ID-usable ones), inner ring: overhead (≥35°) · space = play, ←/→ = step</p>

<script>
const DATA = __DATA__;
const TAU = Math.PI*2, SEC = 24, SW = TAU/SEC, ELEV_SPLIT = 35;
let seg = 0, t = 0, playing = false, selected = null, hover = null, showTrack = false;

const S = DATA.map(d => {
  const ids = [];
  const idOf = raw => { let i = ids.indexOf(raw); if (i<0){ ids.push(raw); i = ids.length-1; } return i; };
  const frames = d.frames.map(fr => ({...fr, a: fr.a.map(a => ({...a, id: idOf(a.id)}))}));
  const n = ids.length;
  const cumO = [], cumI = [];
  let mo = new Array(n).fill(0), mi = new Array(n).fill(0);
  for (const fr of frames){
    mo = mo.slice(); mi = mi.slice();
    for (const a of fr.a) if (a.ra!=null){
      const k = ((Math.floor((a.ra+Math.PI)/SW))%SEC+SEC)%SEC;
      if (a.el < ELEV_SPLIT) mo[a.id] |= (1<<k); else mi[a.id] |= (1<<k);
    }
    cumO.push(mo); cumI.push(mi);
  }
  const pts=[]; frames.forEach(fr=>{pts.push(fr.cam); fr.a.forEach(a=>pts.push(a.p));});
  const xs=pts.map(p=>p[0]), ys=pts.map(p=>p[1]);
  return {...d, frames, n, cumO, cumI,
    bx:[Math.min(...xs),Math.max(...xs)], by:[Math.min(...ys),Math.max(...ys)]};
});
const pop = m => { let c=0; while(m){ c+=m&1; m>>=1; } return c; };

let C = {};
const readTokens = () => { const s = getComputedStyle(document.documentElement);
  for (const k of ["bg","ink","muted","faint","drone","drone-soft","cov","cov-soft","animal","sel","grid","wedge","line"])
    C[k] = s.getPropertyValue("--"+k).trim(); };
readTokens();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", ()=>{readTokens(); draw();});
new MutationObserver(()=>{readTokens(); draw();}).observe(document.documentElement,{attributes:true,attributeFilter:["data-theme"]});

const cv = document.getElementById("cv"), ctx = cv.getContext("2d");
let W=0,H=0,DPR=1, view={s:1,ox:0,oy:0};
function fit(){
  const st=document.getElementById("stage"); DPR=devicePixelRatio||1;
  W=st.clientWidth; H=st.clientHeight; cv.width=W*DPR; cv.height=H*DPR;
  const d=S[seg], pad=64;
  const sx=(W-2*pad)/Math.max(d.bx[1]-d.bx[0],1e-9), sy=(H-2*pad)/Math.max(d.by[1]-d.by[0],1e-9);
  view.s=Math.min(sx,sy);
  view.ox=W/2 - view.s*(d.bx[0]+d.bx[1])/2;
  view.oy=H/2 + view.s*(d.by[0]+d.by[1])/2;
}
const px = p => [view.s*p[0]+view.ox, -view.s*p[1]+view.oy];
const sa = a => -a;

function ring(pt, rr, mask, h, lwOn, lwOff){
  for(let k=0;k<SEC;k++){
    const a0=(h??0)+(-Math.PI+k*SW)+SW*0.12, a1=(h??0)+(-Math.PI+(k+1)*SW)-SW*0.12;
    ctx.beginPath(); ctx.arc(pt[0],pt[1],rr,sa(a1),sa(a0));
    ctx.strokeStyle=(mask>>k)&1?C.cov:C.grid; ctx.lineWidth=(mask>>k)&1?lwOn:lwOff;
    ctx.stroke();
  }
}

function draw(){
  const d=S[seg], fr=d.frames[t], mo=d.cumO[t], mi=d.cumI[t];
  ctx.setTransform(DPR,0,0,DPR,0,0);
  ctx.fillStyle=C.bg; ctx.fillRect(0,0,W,H);

  ctx.fillStyle=C.grid;
  const gs=56;
  for(let x=(view.ox%gs+gs)%gs; x<W; x+=gs) for(let y=(view.oy%gs+gs)%gs; y<H; y+=gs){
    ctx.beginPath(); ctx.arc(x,y,1.1,0,TAU); ctx.fill(); }

  if(showTrack){
    ctx.strokeStyle=C.line; ctx.lineWidth=1; ctx.setLineDash([3,5]); ctx.beginPath();
    d.frames.forEach((f,i)=>{const q=px(f.cam); i?ctx.lineTo(...q):ctx.moveTo(...q);});
    ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle=C.drone; ctx.lineWidth=2.5; ctx.lineCap="round"; ctx.beginPath();
    for(let i=0;i<=t;i++){const q=px(d.frames[i].cam); i?ctx.lineTo(...q):ctx.moveTo(...q);}
    ctx.stroke();
  }

  const cp=px(fr.cam), look=sa(fr.look), spread=0.24;
  const reach = fr.a.length ? Math.max(...fr.a.map(a=>Math.hypot(px(a.p)[0]-cp[0],px(a.p)[1]-cp[1])))+40 : 180;
  ctx.fillStyle=C.wedge; ctx.beginPath(); ctx.moveTo(...cp);
  ctx.arc(cp[0],cp[1],reach,look-spread,look+spread); ctx.closePath(); ctx.fill();

  for(const a of fr.a){
    const p=px(a.p), isSel=selected===a.id, isHov=hover===a.id;
    ring(p, isSel?21:16, mo[a.id]||0, a.h, 3.2, 1.6);          /* outer: side views */
    ring(p, isSel?13.5:10, mi[a.id]||0, a.h, 2.6, 1.1);        /* inner: overhead   */
    if(a.h!=null){
      const th=sa(a.h);
      ctx.save(); ctx.translate(...p); ctx.rotate(th);
      ctx.beginPath(); ctx.moveTo(8,0); ctx.lineTo(-5.4,4.8); ctx.lineTo(-2.7,0); ctx.lineTo(-5.4,-4.8); ctx.closePath();
      ctx.fillStyle=isSel?C.sel:C.animal; ctx.globalAlpha=a.cf?1:.55; ctx.fill(); ctx.globalAlpha=1;
      ctx.restore();
    }else{
      ctx.beginPath(); ctx.arc(p[0],p[1],4.5,0,TAU);
      ctx.strokeStyle=C.animal; ctx.lineWidth=2; ctx.stroke();
    }
    if(isSel||isHov){
      ctx.font="600 11px Archivo"; ctx.fillStyle=C.ink;
      ctx.fillText("#"+(a.id+1), p[0]+24, p[1]+4);
    }
  }

  ctx.beginPath(); ctx.arc(cp[0],cp[1],7,0,TAU); ctx.fillStyle=C.drone; ctx.fill();
  ctx.beginPath(); ctx.arc(cp[0],cp[1],11,0,TAU); ctx.strokeStyle=C["drone-soft"]; ctx.lineWidth=5; ctx.stroke();
  ctx.save(); ctx.translate(...cp); ctx.rotate(look);
  ctx.beginPath(); ctx.moveTo(13,0); ctx.lineTo(22,0); ctx.strokeStyle=C.drone; ctx.lineWidth=2.5; ctx.stroke();
  ctx.restore();

  hud(fr,mo); rail(mo,mi);
}

function hud(fr,mo){
  const d=S[seg];
  const mean = d.n? Math.round([...Array(d.n).keys()].reduce((s,i)=>s+pop(mo[i]||0),0)/d.n*(360/SEC)) : 0;
  document.getElementById("hud").innerHTML =
    `<span class="chip">frame <b>${fr.f}</b></span>`+
    `<span class="chip">animals <b>${fr.a.length}</b></span>`+
    `<span class="chip">side-view coverage <b>${mean}°</b> / 360°</span>`;
}
function ringSVG(maskO,maskI,h){
  let s=`<svg width="34" height="34" viewBox="-17 -17 34 34" aria-hidden="true">`;
  const arcs=(mask,r,wOn,wOff)=>{ let o="";
    for(let k=0;k<SEC;k++){
      const a0=(h??0)+(-Math.PI+k*SW)+SW*.14, a1=(h??0)+(-Math.PI+(k+1)*SW)-SW*.14;
      o+=`<path d="M ${r*Math.cos(-a0)} ${r*Math.sin(-a0)} A ${r} ${r} 0 0 0 ${r*Math.cos(-a1)} ${r*Math.sin(-a1)}"
          fill="none" stroke="${(mask>>k)&1?'var(--cov)':'var(--line)'}" stroke-width="${(mask>>k)&1?wOn:wOff}"/>`;
    } return o; };
  return s+arcs(maskO,13,3,1.4)+arcs(maskI,8,2.4,1)+`</svg>`;
}
function rail(mo,mi){
  const d=S[seg], fr=d.frames[t], seenO=id=>pop(mo[id]||0), seenI=id=>pop(mi[id]||0);
  const live=new Map(fr.a.map(a=>[a.id,a]));
  const order=[...Array(d.n).keys()].sort((a,b)=>seenO(b)-seenO(a));
  const mean=Math.round(order.reduce((s,i)=>s+seenO(i),0)/Math.max(d.n,1)*(360/SEC));
  let html=`<div id="group"><div class="big">${mean}°</div><div class="cap">mean side-view coverage of the herd so far (outer rings)</div></div>
            <div class="lbl">Animals · sides seen</div>`;
  for(const id of order){
    const a=live.get(id), degO=Math.round(seenO(id)*(360/SEC)), degI=Math.round(seenI(id)*(360/SEC));
    html+=`<div class="card" data-id="${id}" data-sel="${selected===id?1:0}" role="button" tabindex="0"
             aria-label="animal ${id+1}, ${degO} degrees of side views">
             ${ringSVG(mo[id]||0, mi[id]||0, a?.h)}
             <div><div class="nm">#${id+1}</div><div class="dg">${a?`dist ${a.d.toFixed(2)}`:"out of frame"}${degI?` · top ${degI}°`:""}</div></div>
             <div class="pc">${degO}°</div></div>`;
  }
  const rl=document.getElementById("rail");
  rl.innerHTML=html;
  rl.querySelectorAll(".card").forEach(c=>{
    const go=()=>{selected=selected==+c.dataset.id?null:+c.dataset.id; draw();};
    c.onclick=go; c.onkeydown=e=>{if(e.key==="Enter"||e.key===" "){e.preventDefault();go();}};
  });
}

const scrub=document.getElementById("scrub"), playBtn=document.getElementById("play"), tEl=document.getElementById("t");
function setT(v){ t=Math.max(0,Math.min(S[seg].frames.length-1,v|0)); scrub.value=t;
  tEl.textContent=`sample ${String(t+1).padStart(3)} / ${S[seg].frames.length}`; draw(); }
function setSeg(i){ seg=i; selected=null; t=0;
  scrub.max=S[seg].frames.length-1; fit(); setT(0);
  document.querySelectorAll("#tabs button").forEach((b,j)=>b.setAttribute("aria-pressed",j===i));
}
let raf=null, last=0;
function loop(ts){ if(!playing) return;
  if(ts-last>70){ last=ts; if(t>=S[seg].frames.length-1){toggle(false); return;} setT(t+1); }
  raf=requestAnimationFrame(loop); }
function toggle(on){ playing=on??!playing; playBtn.textContent=playing?"⏸":"▶";
  if(playing){ if(t>=S[seg].frames.length-1) setT(0); raf=requestAnimationFrame(loop);} else cancelAnimationFrame(raf); }
playBtn.onclick=()=>toggle();
scrub.oninput=e=>{toggle(false); setT(+e.target.value);};
document.getElementById("trk").onclick=e=>{showTrack=!showTrack;
  e.currentTarget.setAttribute("aria-pressed",showTrack); draw();};
addEventListener("keydown",e=>{ if(e.target.tagName==="INPUT"&&e.key===" ")return;
  if(e.key===" "){e.preventDefault();toggle();}
  if(e.key==="ArrowRight"){toggle(false);setT(t+1);}
  if(e.key==="ArrowLeft"){toggle(false);setT(t-1);} });

const tip=document.getElementById("tip");
cv.addEventListener("pointermove",e=>{
  const r=cv.getBoundingClientRect(), x=e.clientX-r.left, y=e.clientY-r.top;
  const fr=S[seg].frames[t]; let best=null,bd=26;
  for(const a of fr.a){const p=px(a.p),dd=Math.hypot(p[0]-x,p[1]-y); if(dd<bd){bd=dd;best=a;}}
  hover=best?best.id:null;
  if(best){const dO=Math.round(pop(S[seg].cumO[t][best.id]||0)*(360/SEC)),
           dI=Math.round(pop(S[seg].cumI[t][best.id]||0)*(360/SEC));
    tip.style.display="block"; tip.style.left=(x+16)+"px"; tip.style.top=(y-10)+"px";
    tip.innerHTML=`<b>#${best.id+1}</b> · sides ${dO}° · top ${dI}°<small>heading ${best.h!=null?Math.round(best.h*180/Math.PI)+"°":"—"} · elev ${best.el}° · dist ${best.d.toFixed(2)}</small>`;
  } else tip.style.display="none";
  draw();
});
cv.addEventListener("pointerleave",()=>{hover=null;tip.style.display="none";draw();});
cv.addEventListener("click",()=>{ if(hover!=null){selected=selected===hover?null:hover; draw();} });

const tabs=document.getElementById("tabs");
DATA.forEach((d,i)=>{const b=document.createElement("button");
  b.textContent=d.name; b.onclick=()=>setSeg(i); tabs.appendChild(b);});
addEventListener("resize",()=>{fit();draw();});
setSeg(0);
if(!matchMedia("(prefers-reduced-motion: reduce)").matches) toggle(true);
</script>'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("/mnt/d/3DBOX/papersubdata"))
    ap.add_argument("--detany3d", type=Path, default=Path("/home/shuklva/DetAny3D"),
                    help="repo holding tools/heading/papersub.py")
    ap.add_argument("--seg", action="append", required=True,
                    help='"group/video/segN:Display name" (repeatable)')
    ap.add_argument("--step", type=int, default=2, help="frame decimation")
    ap.add_argument("--out", type=Path, default=Path("flightdeck.html"))
    args = ap.parse_args()

    sys.path.insert(0, str(args.detany3d))
    data = []
    for spec in args.seg:
        rel, _, name = spec.partition(":")
        species = rel.split("/")[0].rstrip("0123456789")
        d = export_segment(args.root / rel, name or rel, species, step=args.step)
        print(f"  {d['name']:16s} frames={len(d['frames'])}")
        data.append(d)

    args.out.write_text(TEMPLATE.replace("__DATA__", json.dumps(data, separators=(",", ":"))))
    print(f"wrote {args.out}  ({args.out.stat().st_size/1024:.0f} KB) — a single static file; host anywhere.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
