"""
make_static_backup.py — pre-render the whole demo into ONE self-contained HTML
file (demo_backup.html) that opens in any browser, offline, with no Python,
no Streamlit, and no dependencies. A safety net for the live presentation.

Run once:
    source .pneumonia/bin/activate
    python make_static_backup.py
Then open demo_backup.html in a browser.
"""

import base64
import io
import random

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import datasets
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, roc_auc_score, confusion_matrix,
)

from src.utils import get_device
from compare_models import _load_entry, _full_inference, MODEL_DEFS, CLASS_NAMES
import app  # reuse the theme colors + chart + Grad-CAM helpers


# ── theme (mirror app.py "Calm Clinic") ────────────────────────────────────────
BG, CARD, ACCENT = app.BG, app.CARD, app.ACCENT
TEXT, TEXT_SOFT = app.TEXT, app.TEXT_SOFT
CORRECT, WRONG = app.CORRECT, app.WRONG
ORDER = app.MODEL_ORDER


def b64_png(pil_img, max_px=360):
    img = pil_img.copy()
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def b64_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, facecolor=CARD, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    device = get_device()
    print("Loading models…")
    entries = {}
    for defn in MODEL_DEFS:
        e = _load_entry(defn, device)
        entries[e["short"]] = e
    data_dir = entries["ResNet-18"]["data_dir"]

    print("Scoring full test set for each model…")
    probs_by = {}
    for name in ORDER:
        e = entries[name]
        y, p = _full_inference(e["model"], e["transform"], e["data_dir"], device)
        probs_by[name] = (np.array(y), np.array(p))
    labels = probs_by["ResNet-18"][0]           # shared label order

    samples = datasets.ImageFolder(root=data_dir + "/test").samples

    html = [_head()]

    # ── 1. comparison table ──
    html.append('<h2>Model comparison · full test set (624 images)</h2>')
    rows = ['<tr><th></th><th>Params</th><th>Threshold</th><th>Accuracy</th>'
            '<th>Macro F1</th><th>AUC</th><th>Recall (Pneumonia)</th><th>Missed</th></tr>']
    for name in ORDER:
        y, p = probs_by[name]
        thr = entries[name]["threshold"]
        pred = (p >= thr).astype(int)
        acc = accuracy_score(y, pred) * 100
        mf1 = f1_score(y, pred, average="macro")
        rec = recall_score(y, pred, pos_label=1) * 100
        auc = roc_auc_score(y, p)
        missed = int(confusion_matrix(y, pred)[1][0])
        params = "~11M" if name == "ResNet-18" else "~1M"
        rows.append(
            f'<tr><td class="mname">{name}</td><td>{params}</td><td>{thr:.2f}</td>'
            f'<td class="ok">{acc:.1f}%</td><td>{mf1:.3f}</td><td>{auc:.3f}</td>'
            f'<td class="ok">{rec:.1f}%</td><td class="bad">{missed}</td></tr>')
    html.append(f'<table class="cmp">{"".join(rows)}</table>')

    # ── 2. analysis charts per model ──
    html.append('<h2>Analysis</h2>')
    for name in ORDER:
        y, p = probs_by[name]
        thr = entries[name]["threshold"]
        t_fig = b64_fig(app._fig_threshold(y, p, thr))
        c_fig = b64_fig(app._fig_confidence(y, p, thr))
        html.append(f'<h3>{name}</h3>')
        html.append(
            '<div class="row">'
            f'<div class="chartcard"><div class="cap">Threshold explorer</div>'
            f'<img src="data:image/png;base64,{t_fig}"/></div>'
            f'<div class="chartcard"><div class="cap">Confidence distribution</div>'
            f'<img src="data:image/png;base64,{c_fig}"/></div>'
            '</div>')

    # ── 3. sample predictions (all three models) ──
    rng = random.Random(7)
    pick = rng.sample(range(len(samples)), 6)
    html.append('<h2>Sample predictions · all three models</h2>')
    cards = []
    for idx in pick:
        path, true_label = samples[idx]
        img = b64_png(Image.open(path).convert("L"))
        mrows = []
        for name in ORDER:
            _, p = probs_by[name]
            thr = entries[name]["threshold"]
            prob = float(p[idx])
            pred = 1 if prob >= thr else 0
            ok = pred == true_label
            cls = "ok" if ok else "bad"
            mark = "✓" if ok else "✗"
            mrows.append(
                f'<div class="mrow"><span class="mn">{name}</span>'
                f'<span class="{cls}">{mark} {CLASS_NAMES[pred]} · {prob*100:.0f}%</span></div>')
        cards.append(
            f'<div class="card"><img src="data:image/png;base64,{img}"/>'
            f'<div class="truth">True: {CLASS_NAMES[true_label]}</div>{"".join(mrows)}</div>')
    html.append(f'<div class="grid">{"".join(cards)}</div>')

    # ── 4. Grad-CAM (compare, all three) ──
    pneu = [i for i in range(len(samples)) if labels[i] == 1]
    norm = [i for i in range(len(samples)) if labels[i] == 0]
    gc_pick = rng.sample(pneu, 2) + rng.sample(norm, 1)
    html.append('<h2>Grad-CAM · where each model looks</h2>')
    html.append('<p class="note">Warmer = stronger influence on the prediction. '
                'Coarse, approximate focus — not a clinical map.</p>')
    for idx in gc_pick:
        path, true_label = samples[idx]
        orig = Image.open(path).convert("L").resize((224, 224))
        rgb = Image.open(path).convert("RGB")
        panels = [f'<div class="gcell"><div class="cap">X-ray · True {CLASS_NAMES[true_label]}</div>'
                  f'<img src="data:image/png;base64,{b64_png(orig)}"/></div>']
        for name in ORDER:
            e = entries[name]
            cam = app.compute_cam(e["model"], e["transform"], rgb, device)
            over = app._cam_overlay(orig, cam)
            _, p = probs_by[name]
            thr = entries[name]["threshold"]
            pred = 1 if float(p[idx]) >= thr else 0
            cls = "ok" if pred == true_label else "bad"
            mark = "✓" if pred == true_label else "✗"
            panels.append(
                f'<div class="gcell"><div class="cap">{name}</div>'
                f'<img src="data:image/png;base64,{b64_png(over)}"/>'
                f'<div class="{cls}" style="text-align:center;font-weight:700;">'
                f'{mark} {CLASS_NAMES[pred]} · {float(p[idx])*100:.0f}%</div></div>')
        html.append(f'<div class="row gcrow">{"".join(panels)}</div>')

    html.append("</div></body></html>")

    with open("demo_backup.html", "w") as f:
        f.write("\n".join(html))
    print("Wrote demo_backup.html")


def _head():
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pneumonia Detection — Demo Snapshot</title>
<style>
  body {{ background:{BG}; color:{TEXT}; margin:0;
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  .wrap {{ max-width:1180px; margin:0 auto; padding:2.4rem 1.2rem 4rem; }}
  h1 {{ font-size:2.1rem; margin:0 0 .2rem; letter-spacing:-.01em; }}
  .sub {{ color:{TEXT_SOFT}; margin:0 0 2rem; }}
  h2 {{ font-size:1.5rem; margin:2.4rem 0 .9rem; }}
  h3 {{ font-size:1.1rem; color:{ACCENT}; margin:1.4rem 0 .5rem; }}
  .note {{ color:{TEXT_SOFT}; font-size:.9rem; margin:.2rem 0 1rem; }}
  table.cmp {{ border-collapse:separate; border-spacing:6px; }}
  table.cmp th {{ color:{TEXT_SOFT}; font-weight:600; font-size:.85rem; padding:.4rem .7rem; }}
  table.cmp td {{ background:{CARD}; padding:.55rem .9rem; border-radius:8px; text-align:center;
                  box-shadow:0 1px 6px rgba(46,58,58,.06); font-size:.95rem; }}
  td.mname {{ color:{ACCENT}; font-weight:700; text-align:left; }}
  .ok {{ color:{CORRECT}; font-weight:700; }} .bad {{ color:{WRONG}; font-weight:700; }}
  .row {{ display:flex; gap:1.1rem; flex-wrap:wrap; }}
  .chartcard {{ background:{CARD}; border-radius:14px; padding:.8rem; flex:1 1 440px;
                box-shadow:0 2px 10px rgba(46,58,58,.07); }}
  .chartcard img {{ width:100%; border-radius:8px; }}
  .cap {{ font-weight:700; margin-bottom:.4rem; font-size:.95rem; }}
  .grid {{ display:flex; flex-wrap:wrap; gap:1.1rem; }}
  .card {{ background:{CARD}; border-radius:14px; padding:.7rem; width:240px;
           box-shadow:0 2px 10px rgba(46,58,58,.07); }}
  .card img {{ width:100%; border-radius:9px; }}
  .truth {{ color:{TEXT_SOFT}; font-size:.82rem; margin:.5rem 0 .3rem; }}
  .mrow {{ display:flex; justify-content:space-between; padding:.28rem .1rem;
           border-top:1px solid #EFE9DC; font-size:.85rem; }}
  .mrow .mn {{ color:{ACCENT}; font-weight:600; }}
  .gcrow .gcell {{ background:{CARD}; border-radius:12px; padding:.6rem; flex:1 1 240px;
                   box-shadow:0 2px 10px rgba(46,58,58,.07); }}
  .gcrow .gcell img {{ width:100%; border-radius:8px; }}
</style></head><body><div class="wrap">
<h1>Pneumonia Detection from Chest X-Rays</h1>
<p class="sub">Offline demo snapshot · DeeperCNN · ResNet-18 · Distilled CNN</p>"""


if __name__ == "__main__":
    main()
