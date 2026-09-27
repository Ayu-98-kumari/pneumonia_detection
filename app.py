"""
app.py — Pneumonia Detection demo interface (Streamlit).

A calm, clinical UI for showing model predictions on chest X-rays without
re-running the terminal. Reuses the exact inference helpers from
compare_models.py so on-screen numbers match the project's reported results.

Run:
    source .pneumonia/bin/activate
    streamlit run app.py
"""

import base64
import io
import random
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import datasets

from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, roc_auc_score,
)

from src.utils import get_device, set_seed
from compare_models import _load_entry, _predict, _full_inference, MODEL_DEFS, CLASS_NAMES


# ── theme ("Calm Clinic") ─────────────────────────────────────────────────────

BG        = "#EAE2D0"   # soft warm beige
CARD      = "#FAF8F2"   # warm off-white panels, lift off the beige
ACCENT    = "#5B9AA0"
TEXT      = "#2E3A3A"
TEXT_SOFT = "#556060"
CORRECT   = "#7FB09A"
WRONG     = "#C98A7D"
NEUTRAL   = "#C3D0D2"   # border for predictions with no known label (uploads)

# chart series colors — validated CVD-safe categorical hues (dataviz skill),
# paired with distinct line styles + direct labels as secondary encoding.
C_RECALL    = "#2a78d6"   # blue
C_PRECISION = "#eb6834"   # orange
C_ACCURACY  = "#1baf7a"   # aqua
C_NORMAL    = "#2a78d6"   # blue   (true NORMAL)
C_PNEU      = "#eb6834"   # orange (true PNEUMONIA)
GRIDCOL     = "#E4EBED"

MODEL_ORDER = ["DeeperCNN", "ResNet-18", "Distilled CNN"]


CSS = f"""
<style>
    .stApp {{ background: {BG}; }}
    .block-container {{ padding-top: 2.2rem; max-width: 1500px; }}
    /* hide the Deploy button in the top-right toolbar */
    [data-testid="stAppDeployButton"] {{ display: none !important; }}
    .stDeployButton {{ display: none !important; }}
    section[data-testid="stSidebar"] {{ background: {CARD}; min-width: 300px; max-width: 300px; }}
    section[data-testid="stSidebar"] > div {{ width: 300px; }}

    /* "Controls" — the biggest thing in the sidebar */
    section[data-testid="stSidebar"] h3 {{
        font-size: 1.7rem !important; font-weight: 800 !important;
        color: {TEXT}; margin: 0.2rem 0 0.4rem 0; letter-spacing: -0.01em;
    }}
    /* section labels: View / Model / Image source / Cases / Batch size */
    section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        font-size: 1.12rem !important; font-weight: 700 !important; color: {TEXT} !important;
    }}
    /* option text (Test set, Upload, Mixed, model names…) — smaller & lighter */
    section[data-testid="stSidebar"] [role="radiogroup"] p,
    section[data-testid="stSidebar"] [data-baseweb="select"] div {{
        font-size: 0.92rem !important; font-weight: 400 !important; color: {TEXT_SOFT} !important;
    }}
    /* checkbox toggle labels */
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] p {{
        font-size: 1.0rem !important; font-weight: 600 !important; color: {TEXT} !important;
    }}

    /* raised 3D section separators in the sidebar */
    hr.sep3d {{
        border: none; height: 7px; margin: 1.25rem 0; border-radius: 5px;
        background: linear-gradient(180deg, #CDBFA6 0%, #E7DFCE 55%, #FAF8F2 100%);
        box-shadow: 0 3px 6px rgba(46,58,58,0.34),
                    inset 0 1px 0 rgba(255,255,255,0.95),
                    inset 0 -2px 3px rgba(46,58,58,0.16);
    }}

    h1, h2, h3, h4, p, label, span, div {{ color: {TEXT}; }}

    .app-title {{
        font-size: 3.2rem; font-weight: 700; color: {TEXT};
        margin: 0 0 0.15rem 0; letter-spacing: -0.01em;
    }}
    .app-sub {{ font-size: 1.0rem; color: {TEXT_SOFT}; margin: 0 0 1.4rem 0; }}

    .summary {{
        background: {CARD}; border-radius: 14px; padding: 0.9rem 1.3rem;
        box-shadow: 0 2px 10px rgba(46,58,58,0.06);
        margin: 0.2rem 0 1.4rem 0; display: flex; gap: 2.6rem; flex-wrap: wrap;
    }}
    .summary .stat {{ display: flex; flex-direction: column; }}
    .summary .stat .num {{ font-size: 1.6rem; font-weight: 700; color: {ACCENT}; line-height: 1.1; }}
    .summary .stat .lab {{ font-size: 0.82rem; color: {TEXT_SOFT}; }}

    .grid {{ display: flex; flex-wrap: wrap; gap: 1.1rem; }}

    .card {{
        background: {CARD}; border-radius: 14px; padding: 0.7rem;
        box-shadow: 0 2px 10px rgba(46,58,58,0.07);
        width: 220px; box-sizing: border-box;
    }}
    .card img {{
        width: 100%; border-radius: 9px; display: block;
        border: 3px solid {NEUTRAL};
    }}
    .card.correct img {{ border-color: {CORRECT}; }}
    .card.wrong   img {{ border-color: {WRONG}; }}

    .truth {{ font-size: 0.8rem; color: {TEXT_SOFT}; margin: 0.5rem 0 0.15rem 0; }}
    .verdict {{ font-size: 1.02rem; font-weight: 700; margin: 0.1rem 0; }}
    .verdict.correct {{ color: {CORRECT}; }}
    .verdict.wrong   {{ color: {WRONG}; }}
    .verdict.plain   {{ color: {TEXT}; }}
    .conf {{ font-size: 0.82rem; color: {TEXT_SOFT}; }}

    /* compare card: one X-ray + three model rows */
    .ccard {{
        background: {CARD}; border-radius: 14px; padding: 0.7rem;
        box-shadow: 0 2px 10px rgba(46,58,58,0.07);
        width: 300px; box-sizing: border-box;
    }}
    .ccard img {{ width: 100%; border-radius: 9px; display: block; border: 3px solid {NEUTRAL}; }}
    .ccard .truth {{ text-align: center; }}
    .mrow {{
        display: flex; justify-content: space-between; align-items: baseline;
        padding: 0.32rem 0.1rem; border-top: 1px solid #EEF2F3;
    }}
    .mrow .mname {{ font-size: 0.84rem; color: {ACCENT}; font-weight: 600; }}
    .mrow .mpred {{ font-size: 0.86rem; font-weight: 700; }}
    .mrow .mpred.correct {{ color: {CORRECT}; }}
    .mrow .mpred.wrong   {{ color: {WRONG}; }}
    .mrow .mpred.plain   {{ color: {TEXT}; }}

    /* full test-set metrics */
    .fulltitle {{ font-size: 1.15rem; font-weight: 700; color: {TEXT};
                  margin: 0.6rem 0 0.7rem 0; }}
    table.cm, table.cmptable {{
        border-collapse: separate; border-spacing: 6px; margin: 0.2rem 0 1.6rem 0;
    }}
    table.cm td, table.cmptable td {{
        background: {CARD}; padding: 0.55rem 0.9rem; border-radius: 8px;
        text-align: center; font-size: 0.95rem; color: {TEXT};
        box-shadow: 0 1px 6px rgba(46,58,58,0.05);
    }}
    td.cmh {{ color: {TEXT_SOFT}; font-weight: 600; background: transparent; box-shadow: none; }}
    td.cmok  {{ color: {CORRECT}; font-weight: 700; }}
    td.cmbad {{ color: {WRONG};   font-weight: 700; }}
    table.cmptable td.mname {{ color: {ACCENT}; font-weight: 700; text-align: left; }}
</style>
"""


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    device = get_device()
    entries = {}
    for defn in MODEL_DEFS:
        entry = _load_entry(defn, device)
        entries[entry["short"]] = entry
    return entries, device


@st.cache_data(show_spinner=False)
def list_test_samples(data_dir):
    """Return [(path, label)] for the test folder, without loading images."""
    ds = datasets.ImageFolder(root=data_dir + "/test")
    return ds.samples


@st.cache_data(show_spinner="Scoring full test set…")
def full_test_probs(model_name):
    """Run each model once over the whole test set; cache (labels, probs).

    Threshold is applied afterward, so it's cached independent of threshold.
    """
    entries, device = load_models()
    entry = entries[model_name]
    labels, probs = _full_inference(
        entry["model"], entry["transform"], entry["data_dir"], device
    )
    return labels, probs


def img_to_base64(pil_img, max_px=360):
    img = pil_img.copy()
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.convert("L").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def pick_batch(samples, n, class_filter, seed):
    rng = random.Random(seed)
    pool = samples
    if class_filter == "NORMAL only":
        pool = [s for s in samples if s[1] == 0]
    elif class_filter == "PNEUMONIA only":
        pool = [s for s in samples if s[1] == 1]
    n = min(n, len(pool))
    return rng.sample(pool, n)


def verdict_class(pred, true_label):
    if true_label is None:
        return "plain"
    return "correct" if pred == true_label else "wrong"


# ── rendering ─────────────────────────────────────────────────────────────────

def render_single(entry, batch, device, threshold):
    cards = []
    correct = total = pneu_total = pneu_hit = 0

    for path, true_label in batch:
        rgb = Image.open(path).convert("RGB")
        prob = _predict(entry["model"], entry["transform"], rgb, device)
        pred = 1 if prob >= threshold else 0

        vc = verdict_class(pred, true_label)
        b64 = img_to_base64(Image.open(path))

        truth_html = (
            f'<div class="truth">True: {CLASS_NAMES[true_label]}</div>'
            if true_label is not None else '<div class="truth">Uploaded image</div>'
        )
        mark = "✓" if vc == "correct" else ("✗" if vc == "wrong" else "")
        cards.append(
            f'<div class="card {vc}">'
            f'<img src="data:image/png;base64,{b64}"/>'
            f'{truth_html}'
            f'<div class="verdict {vc}">{mark} {CLASS_NAMES[pred]}</div>'
            f'<div class="conf">confidence {prob*100:.1f}%</div>'
            f'</div>'
        )

        if true_label is not None:
            total += 1
            correct += int(pred == true_label)
            if true_label == 1:
                pneu_total += 1
                pneu_hit += int(pred == 1)

    if total:
        acc = correct / total * 100
        recall = (pneu_hit / pneu_total * 100) if pneu_total else None
        stats = [
            ("Batch accuracy", f"{acc:.1f}%"),
            ("Correct", f"{correct} / {total}"),
        ]
        if recall is not None:
            stats.append(("Pneumonia recall", f"{recall:.1f}%"))
        _render_summary(stats)

    st.markdown('<div class="grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def render_compare(entries, batch, device, thresholds):
    cards = []
    for path, true_label in batch:
        rgb = Image.open(path).convert("RGB")
        b64 = img_to_base64(Image.open(path))

        rows = []
        for name in MODEL_ORDER:
            entry = entries[name]
            prob = _predict(entry["model"], entry["transform"], rgb, device)
            pred = 1 if prob >= thresholds[name] else 0
            vc = verdict_class(pred, true_label)
            mark = "✓" if vc == "correct" else ("✗" if vc == "wrong" else "")
            rows.append(
                f'<div class="mrow"><span class="mname">{name}</span>'
                f'<span class="mpred {vc}">{mark} {CLASS_NAMES[pred]} · {prob*100:.0f}%</span></div>'
            )

        truth_html = (
            f'<div class="truth">True: {CLASS_NAMES[true_label]}</div>'
            if true_label is not None else '<div class="truth">Uploaded image</div>'
        )
        cards.append(
            f'<div class="ccard">'
            f'<img src="data:image/png;base64,{b64}"/>'
            f'{truth_html}'
            f'{"".join(rows)}'
            f'</div>'
        )

    st.markdown('<div class="grid">' + "".join(cards) + "</div>", unsafe_allow_html=True)


def _render_summary(stats):
    inner = "".join(
        f'<div class="stat"><span class="num">{v}</span><span class="lab">{k}</span></div>'
        for k, v in stats
    )
    st.markdown(f'<div class="summary">{inner}</div>', unsafe_allow_html=True)


def _confusion_html(cm):
    """cm = [[TN, FP], [FN, TP]] over labels [NORMAL, PNEUMONIA]."""
    (tn, fp), (fn, tp) = cm
    return (
        '<table class="cm">'
        '<tr><td class="cmh"></td><td class="cmh">Pred NORMAL</td><td class="cmh">Pred PNEUMONIA</td></tr>'
        f'<tr><td class="cmh">True NORMAL</td><td class="cmok">{tn}</td><td class="cmbad">{fp}</td></tr>'
        f'<tr><td class="cmh">True PNEUMONIA</td><td class="cmbad">{fn}</td><td class="cmok">{tp}</td></tr>'
        '</table>'
    )


def render_metrics_bar(labels, probs, threshold):
    """Full test-set headline metrics as a row of stat tiles."""
    preds = (probs >= threshold).astype(int)

    acc    = accuracy_score(labels, preds) * 100
    mf1    = f1_score(labels, preds, average="macro")
    recall = recall_score(labels, preds, pos_label=1) * 100
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        auc = float("nan")
    missed = int(confusion_matrix(labels, preds)[1][0])   # pneumonia predicted NORMAL

    st.markdown(f'<div class="fulltitle">Full test set · {len(labels)} images · '
                f'threshold {threshold:.2f}</div>', unsafe_allow_html=True)
    _render_summary([
        ("Accuracy", f"{acc:.1f}%"),
        ("Macro F1", f"{mf1:.3f}"),
        ("AUC (ROC)", f"{auc:.3f}"),
        ("Pneumonia recall", f"{recall:.1f}%"),
        ("Pneumonia missed", f"{missed}"),
    ])


def render_full_compare(thresholds):
    st.markdown('<div class="fulltitle">Full test set · all three models</div>',
                unsafe_allow_html=True)
    header = ('<tr><td class="cmh"></td><td class="cmh">Accuracy</td>'
              '<td class="cmh">Macro F1</td><td class="cmh">AUC</td>'
              '<td class="cmh">Recall (Pneumonia)</td><td class="cmh">Missed</td></tr>')
    rows = []
    for name in MODEL_ORDER:
        labels, probs = full_test_probs(name)
        thr = thresholds[name]
        preds = (probs >= thr).astype(int)
        acc = accuracy_score(labels, preds) * 100
        mf1 = f1_score(labels, preds, average="macro")
        rec = recall_score(labels, preds, pos_label=1) * 100
        try:
            auc = roc_auc_score(labels, probs)
        except ValueError:
            auc = float("nan")
        missed = int(confusion_matrix(labels, preds)[1][0])
        rows.append(
            f'<tr><td class="mname">{name}</td>'
            f'<td class="cmok">{acc:.1f}%</td><td>{mf1:.3f}</td><td>{auc:.3f}</td>'
            f'<td class="cmok">{rec:.1f}%</td><td class="cmbad">{missed}</td></tr>'
        )
    st.markdown(f'<table class="cmptable">{header}{"".join(rows)}</table>',
                unsafe_allow_html=True)


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

def _target_layer(model):
    """Last conv feature map for each architecture."""
    if hasattr(model, "conv5"):      # DeeperCNN / Distilled
        return model.conv5
    if hasattr(model, "layer4"):     # ResNet-18
        return model.layer4
    raise ValueError("No known Grad-CAM target layer for this model.")


def compute_cam(model, transform, rgb, device):
    """Return a HxW Grad-CAM map in [0,1] for the model's single logit."""
    target = _target_layer(model)
    store = {}
    h1 = target.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    h2 = target.register_full_backward_hook(
        lambda m, gi, go: store.__setitem__("g", go[0].detach()))
    try:
        model.zero_grad()
        x = transform(rgb).unsqueeze(0).to(device).requires_grad_(True)
        out = model(x)
        out[:, 0].sum().backward()
        acts = store["a"][0]           # [C,H,W]
        grads = store["g"][0]          # [C,H,W]
        weights = grads.mean(dim=(1, 2))
        cam = torch.relu((weights[:, None, None] * acts).sum(0))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def _cam_heat(cam, size=224):
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8))
                   .resize((size, size), Image.BILINEAR)) / 255.0
    heat = plt.get_cmap("turbo")(c)[..., :3]
    return Image.fromarray((heat * 255).astype(np.uint8))


def _cam_overlay(gray_img, cam, size=224, alpha=0.45):
    g = np.asarray(gray_img.convert("L").resize((size, size))) / 255.0
    c = np.asarray(Image.fromarray((cam * 255).astype(np.uint8))
                   .resize((size, size), Image.BILINEAR)) / 255.0
    heat = plt.get_cmap("turbo")(c)[..., :3]
    over = (1 - alpha) * np.stack([g] * 3, -1) + alpha * heat
    return Image.fromarray((over * 255).astype(np.uint8))


def _img_option(i, path, label, wrong=False):
    tag = CLASS_NAMES[label] if label is not None else "Uploaded"
    cross = "  ❌" if wrong else ""
    return f"{i+1}. {tag} — {Path(path).name}{cross}"


@st.cache_data(show_spinner=False)
def _batch_misclassified(batch, items):
    """Indices in `batch` that any of the given (model, threshold) pairs get wrong.

    Test-set images only (Uploaded images have no true label). Runs over the small
    on-screen batch, so it needs no full test-set scoring.
    """
    entries, device = load_models()
    wrong = set()
    for i, (path, true_label) in enumerate(batch):
        if true_label is None:
            continue
        rgb = Image.open(path).convert("RGB")
        for name, thr in items:
            entry = entries[name]
            prob = _predict(entry["model"], entry["transform"], rgb, device)
            if (1 if prob >= thr else 0) != true_label:
                wrong.add(i)
                break
    return wrong


def render_gradcam_single(entry, path, true_label, thr, device):
    rgb = Image.open(path).convert("RGB")
    prob = _predict(entry["model"], entry["transform"], rgb, device)
    pred = 1 if prob >= thr else 0
    cam = compute_cam(entry["model"], entry["transform"], rgb, device)
    orig = Image.open(path).convert("L").resize((224, 224))

    c1, c2, c3 = st.columns(3)
    c1.image(orig, caption="Chest X-ray", use_container_width=True)
    c2.image(_cam_heat(cam), caption="Grad-CAM focus", use_container_width=True)
    c3.image(_cam_overlay(orig, cam), caption="Overlay", use_container_width=True)

    vc = verdict_class(pred, true_label)
    mark = "✓" if vc == "correct" else ("✗" if vc == "wrong" else "")
    truth = (f'True: {CLASS_NAMES[true_label]} · ' if true_label is not None else "")
    st.markdown(
        f'<p style="margin-top:0.3rem;">{truth}'
        f'<b class="verdict {vc}" style="font-size:1.05rem;">{mark} {CLASS_NAMES[pred]}</b>'
        f' · confidence {prob*100:.1f}%</p>', unsafe_allow_html=True)


def render_gradcam_compare(entries, path, true_label, thresholds, device):
    orig = Image.open(path).convert("L").resize((224, 224))
    truth = (f" · True: {CLASS_NAMES[true_label]}" if true_label is not None else "")
    lc, _ = st.columns([1, 2])
    lc.image(orig, caption=f"Selected X-ray{truth}", use_container_width=True)

    cols = st.columns(3)
    for col, name in zip(cols, MODEL_ORDER):
        entry = entries[name]
        rgb = Image.open(path).convert("RGB")
        prob = _predict(entry["model"], entry["transform"], rgb, device)
        pred = 1 if prob >= thresholds[name] else 0
        cam = compute_cam(entry["model"], entry["transform"], rgb, device)
        col.image(_cam_overlay(orig, cam), use_container_width=True)
        vc = verdict_class(pred, true_label)
        mark = "✓" if vc == "correct" else ("✗" if vc == "wrong" else "")
        col.markdown(
            f'<p style="text-align:center;"><b style="color:{ACCENT};">{name}</b><br>'
            f'<span class="verdict {vc}">{mark} {CLASS_NAMES[pred]}</span> · {prob*100:.0f}%</p>',
            unsafe_allow_html=True)


# ── analysis view ─────────────────────────────────────────────────────────────

def _style_chart(ax):
    ax.set_facecolor(CARD)
    ax.tick_params(colors=TEXT_SOFT, labelsize=8)
    ax.grid(True, color=GRIDCOL, linewidth=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRIDCOL)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _fig_threshold(labels, probs, current_thr):
    ts = np.linspace(0.02, 0.98, 49)
    rec, prec, acc = [], [], []
    for t in ts:
        p = (probs >= t).astype(int)
        rec.append(recall_score(labels, p, pos_label=1, zero_division=0) * 100)
        prec.append(precision_score(labels, p, pos_label=1, zero_division=0) * 100)
        acc.append(accuracy_score(labels, p) * 100)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    fig.patch.set_facecolor(CARD)
    _style_chart(ax)

    for name, vals, col, ls in [
        ("Recall",    rec,  C_RECALL,    "-"),
        ("Precision", prec, C_PRECISION, "--"),
        ("Accuracy",  acc,  C_ACCURACY,  "-."),
    ]:
        ax.plot(ts, vals, color=col, linestyle=ls, linewidth=2, label=name)
        ax.text(0.995, vals[-1], f"  {name}", color=col, fontsize=8.5,
                va="center", fontweight="bold")

    ax.axvline(current_thr, color=TEXT_SOFT, linestyle=":", linewidth=1.5)
    ax.text(current_thr, 2, f" thr {current_thr:.2f}", color=TEXT_SOFT,
            fontsize=8, rotation=90, va="bottom", ha="right")

    ax.set_xlim(0, 1.16)
    ax.set_ylim(0, 103)
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel("Score (%)")
    fig.tight_layout()
    return fig


def _fig_confidence(labels, probs, current_thr):
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    fig.patch.set_facecolor(CARD)
    _style_chart(ax)

    bins = np.linspace(0, 1, 31)
    ax.hist(probs[labels == 0], bins=bins, color=C_NORMAL, alpha=0.55,
            label="NORMAL (true)")
    ax.hist(probs[labels == 1], bins=bins, color=C_PNEU, alpha=0.55,
            label="PNEUMONIA (true)")

    ax.axvline(current_thr, color=TEXT_SOFT, linestyle=":", linewidth=1.5)
    ax.text(current_thr, ax.get_ylim()[1] * 0.96, f" threshold {current_thr:.2f}",
            color=TEXT_SOFT, fontsize=8, va="top")

    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability of PNEUMONIA")
    ax.set_ylabel("Number of X-rays")
    leg = ax.legend(loc="upper center", frameon=False, fontsize=9)
    for txt in leg.get_texts():
        txt.set_color(TEXT)
    fig.tight_layout()
    return fig


def _fig_confusion(labels, probs, current_thr):
    """2x2 confusion matrix recomputed at the current threshold.
    Diagonal (correct) tinted sage, off-diagonal (errors) tinted clay."""
    preds = (probs >= current_thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    total = max(int(tn + fp + fn + tp), 1)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))   # match the other two charts
    fig.patch.set_facecolor(CARD)
    ax.set_facecolor(CARD)

    # (row, col): row 0 = True NORMAL (top), row 1 = True PNEUMONIA (bottom)
    #             col 0 = Pred NORMAL (left), col 1 = Pred PNEUMONIA (right)
    cells = {
        (0, 0): (tn, "True Negatives",  True),
        (0, 1): (fp, "False Positives", False),
        (1, 0): (fn, "False Negatives", False),
        (1, 1): (tp, "True Positives",  True),
    }
    for (r, c), (val, lab, ok) in cells.items():
        x, y = c, 1 - r                       # flip so True NORMAL sits on top
        face = CORRECT if ok else WRONG
        ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=face, alpha=0.22,
                                   edgecolor=CARD, linewidth=4))
        ax.text(x + 0.5, y + 0.60, f"{int(val)}", ha="center", va="center",
                fontsize=23, fontweight="bold", color=TEXT)
        ax.text(x + 0.5, y + 0.31, lab, ha="center", va="center",
                fontsize=8.5, color=TEXT_SOFT)
        ax.text(x + 0.5, y + 0.15, f"{val / total * 100:.1f}%", ha="center",
                va="center", fontsize=8, color=TEXT_SOFT)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5])
    ax.set_yticks([0.5, 1.5])
    ax.set_xticklabels(["NORMAL", "PNEUMONIA"], color=TEXT, fontsize=9)
    ax.set_yticklabels(["PNEUMONIA", "NORMAL"], color=TEXT, fontsize=9)
    ax.set_xlabel("Predicted", color=TEXT, fontsize=9.5, fontweight="bold")
    ax.set_ylabel("True label", color=TEXT, fontsize=9.5, fontweight="bold")
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    # no set_aspect("equal") — cells fill the 7.4x3.8 frame so this chart
    # renders the same size as the threshold and confidence charts.
    fig.tight_layout()
    return fig


def render_charts_row(labels, probs, threshold):
    """Three analysis charts side by side, in one row."""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown('<div class="fulltitle">Threshold explorer</div>',
                    unsafe_allow_html=True)
        fig = _fig_threshold(labels, probs, threshold)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Recall, precision and accuracy as the decision threshold moves. "
                   "Raising it lifts precision but can drop recall — more missed "
                   "pneumonia. Use 'Override threshold' to move the dotted line.")
    with c2:
        st.markdown('<div class="fulltitle">Confidence distribution</div>',
                    unsafe_allow_html=True)
        fig = _fig_confidence(labels, probs, threshold)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Predicted pneumonia probability per true class. Clear separation "
                   "means confident predictions; overlap near the threshold is where "
                   "mistakes happen.")
    with c3:
        st.markdown('<div class="fulltitle">Confusion matrix</div>',
                    unsafe_allow_html=True)
        fig = _fig_confusion(labels, probs, threshold)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
        st.caption("Actual counts at the current threshold. Green = correct, "
                   "clay = errors. False Negatives are missed pneumonia — the "
                   "costly mistake. Moving the threshold updates every cell.")


# ── app ───────────────────────────────────────────────────────────────────────

def main():
    st.set_page_config(page_title="Pneumonia Detection", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)

    entries, device = load_models()
    data_dir = entries["ResNet-18"]["data_dir"]

    st.markdown('<div class="app-title">Pneumonia Detection from Chest X-Rays</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="app-sub">Deep learning models predicting NORMAL vs. '
                'PNEUMONIA on chest radiographs.</div>', unsafe_allow_html=True)

    # ── sidebar controls ──
    SEP = '<hr class="sep3d">'

    sb = st.sidebar
    sb.markdown("### Controls")
    mode = sb.radio("View", ["Single model", "Compare all three"])

    if mode == "Single model":
        model_name = sb.selectbox("Model", MODEL_ORDER)

    sb.markdown(SEP, unsafe_allow_html=True)

    source = sb.radio("Image source", ["Test set", "Upload"])

    if source == "Test set":
        class_filter = sb.selectbox("Cases", ["Mixed", "NORMAL only", "PNEUMONIA only"])
        batch_size = sb.select_slider(
            "Batch size",
            [4, 8, 16, 32, 64, 128, 256, 512, 624],
            value=8,
        )
        sb.caption("Scales from a small sample up to the full 624-image test set.")

    sb.markdown(SEP, unsafe_allow_html=True)

    show_analysis = sb.checkbox("Show analysis", value=False)
    sb.caption("Full test-set metrics and charts above the predictions "
               "(scores all 624 images — takes a moment the first time, then cached).")

    sb.markdown(SEP, unsafe_allow_html=True)

    gradcam = sb.checkbox("Grad-CAM inspector")
    sb.caption("See where a model focuses on one chosen X-ray.")

    sb.markdown(SEP, unsafe_allow_html=True)

    custom_thr = sb.checkbox("Override threshold")
    if custom_thr:
        thr_val = sb.slider("Threshold", 0.05, 0.95, 0.50, 0.05)
        sb.caption("Higher → predicts PNEUMONIA only when more confident "
                   "(fewer missed cases matter most in diagnosis).")

    # thresholds per model (embedded best, unless overridden)
    def thr_for(name):
        return thr_val if custom_thr else entries[name]["threshold"]

    # ── image batch ──
    if source == "Test set":
        if "seed" not in st.session_state:
            st.session_state.seed = 42
        if sb.button("New batch 🔄"):
            st.session_state.seed = random.randint(0, 10_000)
        samples = list_test_samples(data_dir)
        batch = pick_batch(samples, batch_size, class_filter, st.session_state.seed)
    else:
        uploaded = sb.file_uploader(
            "Upload chest X-ray(s)", type=["png", "jpg", "jpeg"],
            accept_multiple_files=True,
        )
        batch = []
        if uploaded:
            tmp = Path(st.session_state.get("tmpdir", "/tmp"))
            for uf in uploaded:
                p = tmp / uf.name
                p.write_bytes(uf.getbuffer())
                batch.append((str(p), None))   # label unknown

    # ── render ──
    if not batch:
        st.info("Upload one or more chest X-ray images from the sidebar to see predictions.")
        return

    if mode == "Single model":
        entry = entries[model_name]
        thr = thr_for(model_name)
        st.markdown(
            f'<p style="color:{TEXT_SOFT};margin-top:-0.4rem;">'
            f'<b style="color:{ACCENT};">{model_name}</b> · '
            f'{"~11M" if model_name=="ResNet-18" else "~1M"} params · '
            f'threshold {thr:.2f}</p>', unsafe_allow_html=True)

        # Grad-CAM inspector (focused single image)
        if gradcam:
            st.markdown('<div class="fulltitle">Grad-CAM — where the model looks</div>',
                        unsafe_allow_html=True)
            st.caption("Warmer = stronger influence on the prediction. Coarse focus "
                       "(~7–10px upsampled) — an approximate view, not a clinical map.")
            wrong = _batch_misclassified(batch, ((model_name, thr),))
            options = [_img_option(i, p, l, i in wrong) for i, (p, l) in enumerate(batch)]
            sel = st.radio("Choose an X-ray", options, key="gc_single")
            gp, gl = batch[options.index(sel)]
            render_gradcam_single(entry, gp, gl, thr, device)

        # analysis on top ...
        if show_analysis:
            labels, probs = full_test_probs(model_name)
            render_metrics_bar(labels, probs, thr)
            render_charts_row(labels, probs, thr)

        # ... predictions below
        st.markdown('<div class="fulltitle">Sample predictions</div>',
                    unsafe_allow_html=True)
        render_single(entry, batch, device, thr)
    else:
        thresholds = {name: thr_for(name) for name in MODEL_ORDER}

        # Grad-CAM inspector — all three models on one chosen X-ray
        if gradcam:
            st.markdown('<div class="fulltitle">Grad-CAM — where each model looks</div>',
                        unsafe_allow_html=True)
            st.caption("The same X-ray through each model. Compare whether the compact "
                       "distilled student focuses where the ResNet-18 teacher does.")
            wrong = _batch_misclassified(batch, tuple((n, thresholds[n]) for n in MODEL_ORDER))
            options = [_img_option(i, p, l, i in wrong) for i, (p, l) in enumerate(batch)]
            sel = st.radio("Choose an X-ray", options, key="gc_compare")
            gp, gl = batch[options.index(sel)]
            render_gradcam_compare(entries, gp, gl, thresholds, device)

        if show_analysis:
            render_full_compare(thresholds)
        st.markdown('<div class="fulltitle">Sample predictions</div>',
                    unsafe_allow_html=True)
        render_compare(entries, batch, device, thresholds)


if __name__ == "__main__":
    main()
