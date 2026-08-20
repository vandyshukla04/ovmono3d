"""Build the Phase-A human-audit kit: a local click-through page for the two checks a model must not
self-grade.  [CPU, ~2 min]

    python tools/aeroview/build_audit_kit.py \
        --train /mnt/d/aeroview/labelled/WildBox_train_paper.json \
        --val   /mnt/d/aeroview/labelled/WildBox_val_paper.json \
        --image-root /mnt/d/3DBOX/papersubdata \
        --out /mnt/d/aeroview/audit_kit

Then open  <out>/annotate.html  in any browser (double-click works; everything is local).
Keyboard only: one key per image, auto-advance. Progress is saved in the browser between sessions.
When done, press D (or the button) to download audit_results.json and hand it back.

THE TWO QUESTIONS, and why a human must answer them
----------------------------------------------------
A. FEET (~300 heading-labelled crops, stratified by species x view angle x crowding):
   "Can you see where this animal touches the ground?"   V = yes   H = no   U = can't tell
   GroundCast reads depth off the ground at the contact point; when feet are hidden the visible box
   bottom sits ABOVE the true contact, which biases depth FAR -- a systematic error, not noise. This
   measures how often that happens and on which animals.

B. FLAGGED BOXES (~60 boxes sitting far off their herd's ground plane):
   "Is this box on one standing animal?"   G = yes   B = no, it's on a fragment / between animals
                                           L = the animal is lying down
   Decides label-error masking (B) vs a real tilt output (L). A pilot of 18 said all-B; this is the
   measurement.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

rng = np.random.default_rng(7)


def ioa(a, b):
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by); x2, y2 = min(ax+aw, bx+bw), min(ay+ah, by+bh)
    return 0.0 if (x2 <= x1 or y2 <= y1) else (x2-x1)*(y2-y1)/(aw*ah)


def load(paths):
    anns, images = [], {}
    for p in paths:
        d = json.loads(Path(p).read_text())
        off = len(images)
        remap = {}
        for im in d["images"]:
            key = ("I", p, im["id"])
            images[key] = im; remap[im["id"]] = key
        for a in d["annotations"]:
            a["_img"] = remap[a["image_id"]]
            anns.append(a)
    return anns, images


def crop_save(img_root, im, bbox, out_path, pad=1.8, size=560):
    img = Image.open(Path(img_root) / im["file_path"]).convert("RGB")
    x, y, w, h = bbox
    cx, cy, s = x + w/2, y + h/2, max(w, h) * pad / 2
    X1, Y1 = int(max(0, cx-s)), int(max(0, cy-s))
    X2, Y2 = int(min(img.width, cx+s)), int(min(img.height, cy+s))
    c = img.crop((X1, Y1, X2, Y2))
    scale = size / max(c.width, c.height)
    c = c.resize((int(c.width*scale), int(c.height*scale)), Image.LANCZOS)
    d = ImageDraw.Draw(c)
    d.rectangle([(x-X1)*scale, (y-Y1)*scale, (x+w-X1)*scale, (y+h-Y1)*scale],
                outline=(57, 255, 106), width=3)
    c.save(out_path, quality=88)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--image-root", type=Path, required=True)
    ap.add_argument("--n-feet", type=int, default=300)
    ap.add_argument("--n-flagged", type=int, default=60)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    anns, images = load([args.train, args.val])
    (args.out / "imgs").mkdir(parents=True, exist_ok=True)

    by_img = defaultdict(list)
    for a in anns:
        by_img[a["_img"]].append(a)

    # ---- task A: heading-labelled crops, stratified species x |sin a| band x crowding ------------
    pool = defaultdict(list)
    for a in anns:
        if not a.get("heading_valid", 0):
            continue
        crowd = max((ioa(a["bbox"], b["bbox"]) for b in by_img[a["_img"]] if b is not a), default=0.0)
        band = "end" if abs(np.sin(a["heading_alpha"])) < 0.35 else "side"
        pool[(a["category_name"], band, crowd > 0.1)].append(a)
    quota = max(2, args.n_feet // max(len(pool), 1))
    taskA = []
    for k, lst in sorted(pool.items()):
        take = min(quota, len(lst))
        taskA += [lst[i] for i in rng.choice(len(lst), take, replace=False)]
    taskA = taskA[:args.n_feet]

    # ---- task B: boxes far off their herd's ground plane -----------------------------------------
    by_seg = defaultdict(list)
    for a in anns:
        im = images[a["_img"]]
        seg = tuple(im["file_path"].replace("\\", "/").split("/")[-3:-1])
        by_seg[seg].append(a)
    flagged = []
    for seg, lst in by_seg.items():
        if len(lst) < 30:
            continue
        up = -np.array(lst[0]["R_cam"], float)[:, 1]
        C = np.array([a["center_cam"] for a in lst]); H = np.array([a["dimensions"][1] for a in lst])
        t = (C - (H[:, None]/2)*up) @ up
        off = (t - np.median(t)) / np.maximum(H, 1e-6)
        for a, o in zip(lst, off):
            if abs(o) > 1.0:
                flagged.append((abs(float(o)), float(o), a))
    flagged.sort(key=lambda r: -r[0])
    # spread across segments: at most 4 per segment
    seen = defaultdict(int); taskB = []
    for _, o, a in flagged:
        seg = tuple(images[a["_img"]]["file_path"].replace("\\", "/").split("/")[-3:-1])
        if seen[seg] >= 4:
            continue
        seen[seg] += 1
        taskB.append((o, a))
        if len(taskB) >= args.n_flagged:
            break

    # ---- render ----------------------------------------------------------------------------------
    items = []
    for i, a in enumerate(taskA):
        f = f"A{i:03d}.jpg"
        crop_save(args.image_root, images[a["_img"]], a["bbox"], args.out / "imgs" / f)
        items.append({"id": f, "task": "A", "species": a["category_name"],
                      "file": images[a["_img"]]["file_path"], "track": a["track_id"]})
    for i, (o, a) in enumerate(taskB):
        f = f"B{i:03d}.jpg"
        crop_save(args.image_root, images[a["_img"]], a["bbox"], args.out / "imgs" / f, pad=2.2)
        items.append({"id": f, "task": "B", "species": a["category_name"], "off": round(o, 2),
                      "file": images[a["_img"]]["file_path"], "track": a["track_id"]})
    print(f"task A (feet): {sum(x['task']=='A' for x in items)}   task B (flagged): {sum(x['task']=='B' for x in items)}")

    html = HTML.replace("__ITEMS__", json.dumps(items))
    (args.out / "annotate.html").write_text(html)
    print(f"open  {args.out/'annotate.html'}  in a browser. ~45-90 min of clicking; progress auto-saves.")
    return 0


HTML = r'''<!doctype html><meta charset="utf-8"><title>AeroView Audit Kit</title>
<style>
:root{--bg:#171209;--panel:#211A0F;--line:#3A2F1D;--ink:#EDE4D2;--mut:#A6987C;--acc:#E5A13D;--ok:#3FC9B4}
*{box-sizing:border-box;margin:0}body{background:var(--bg);color:var(--ink);
font:15px/1.5 system-ui,sans-serif;display:flex;flex-direction:column;align-items:center;
min-height:100vh;padding:18px;gap:12px}
h1{font-size:17px}.q{font-size:16px;color:var(--acc);font-weight:600;text-align:center}
img{max-width:min(92vw,620px);max-height:56vh;border:1px solid var(--line);border-radius:10px}
.keys{display:flex;gap:10px;flex-wrap:wrap;justify-content:center}
.k{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:8px 14px;cursor:pointer}
.k b{color:var(--ok);margin-right:6px}
.bar{width:min(92vw,620px);height:6px;background:var(--panel);border-radius:3px;overflow:hidden}
.bar div{height:100%;background:var(--acc)}
.meta{color:var(--mut);font-size:12.5px}
button.dl{background:var(--acc);color:#171209;font-weight:700;border:0;border-radius:8px;padding:9px 16px;cursor:pointer}
.done{font-size:22px;color:var(--ok);font-weight:700}
</style>
<h1>AeroView audit — one key per image</h1>
<div class="q" id="q"></div>
<img id="im" alt="">
<div class="keys" id="keys"></div>
<div class="bar"><div id="p" style="width:0%"></div></div>
<div class="meta" id="m"></div>
<button class="dl" onclick="dl()">D — download results</button>
<script>
const ITEMS=__ITEMS__;
const QA={A:{q:"Can you see where this animal touches the ground (its feet / contact point)?",
             keys:{v:"feet visible",h:"feet hidden (grass, other animals, its own body)",u:"can't tell"}},
          B:{q:"Is the green box on ONE standing animal?",
             keys:{g:"yes — standing animal",b:"no — fragment / between animals / wrong",l:"animal is lying down"}}};
const KEY="aeroview_audit_v1";
let R=JSON.parse(localStorage.getItem(KEY)||"{}"), i=ITEMS.findIndex(x=>!(x.id in R));
if(i<0)i=ITEMS.length;
function show(){
  const done=Object.keys(R).length;
  document.getElementById("p").style.width=(100*done/ITEMS.length)+"%";
  if(i>=ITEMS.length){document.getElementById("q").textContent="";
    document.getElementById("im").style.display="none";document.getElementById("keys").innerHTML=
    '<div class="done">All '+ITEMS.length+' done — press D to download.</div>';
    document.getElementById("m").textContent="";return;}
  const it=ITEMS[i],qa=QA[it.task];
  document.getElementById("q").textContent=qa.q;
  const im=document.getElementById("im");im.style.display="";im.src="imgs/"+it.id;
  document.getElementById("keys").innerHTML=Object.entries(qa.keys).map(([k,v])=>
    `<div class="k" onclick="ans('${k}')"><b>${k.toUpperCase()}</b>${v}</div>`).join("")+
    `<div class="k" onclick="back()"><b>←</b>previous</div>`;
  document.getElementById("m").textContent=
    `${i+1} / ${ITEMS.length} · ${it.species} · task ${it.task}` + (it.off?` · off=${it.off}H`:"");
  if(i+1<ITEMS.length){const pre=new Image();pre.src="imgs/"+ITEMS[i+1].id;}
}
function ans(k){R[ITEMS[i].id]=k;localStorage.setItem(KEY,JSON.stringify(R));i++;show();}
function back(){if(i>0){i--;delete R[ITEMS[i].id];localStorage.setItem(KEY,JSON.stringify(R));show();}}
function dl(){
  const out=ITEMS.map(it=>({...it,answer:R[it.id]||null}));
  const a=document.createElement("a");
  a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:"application/json"}));
  a.download="audit_results.json";a.click();}
addEventListener("keydown",e=>{const k=e.key.toLowerCase();
  if(k==="d"){dl();return;} if(k==="arrowleft"){back();return;}
  if(i<ITEMS.length&&QA[ITEMS[i].task].keys[k])ans(k);});
show();
</script>'''


if __name__ == "__main__":
    raise SystemExit(main())
