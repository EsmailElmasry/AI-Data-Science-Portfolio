import gc
import base64
import io
import json
import os
import uuid
import html as _html
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import joblib
from PIL import Image, ImageDraw

import torch
import torch.nn as nn
from torchvision import models, transforms

try:
    import cv2
except Exception:
    cv2 = None

try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    HAS_GRADCAM = True
except Exception:
    HAS_GRADCAM = False


st.set_page_config(
    page_title="Hospital AI System",
    page_icon="✚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# Paths / constants
# ----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
VITAL_MODEL_PATH = BASE_DIR / "best_vital_signs_model.joblib"
BREAST_MODEL_PATH = BASE_DIR / "best_breast_cancer_768_multitask.pth"
CONFIG_PATH = BASE_DIR / "config.json"

# Test history is stored on disk so it survives refreshes and restarts.
HISTORY_DIR = BASE_DIR / "history"
HISTORY_IMG_DIR = HISTORY_DIR / "images"
HISTORY_FILE = HISTORY_DIR / "records.json"

USERNAME = "Esmail"
PASSWORD = "123456"

PAGES = [
    ("dashboard", "🏠  Dashboard"),
    ("vital", "❤️  Vital-Signs Screening"),
    ("breast", "🩻  Breast Cancer Detection"),
    ("history", "🗂️  Test History"),
]
PAGE_TITLES = {k: v for k, v in PAGES}

for k, v in {
    "authenticated": False,
    "display_name": "",
    "dark_mode": False,
    "page": "dashboard",
    "vital_result": None,
    "breast_result": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


def esc(s):
    return _html.escape(str(s if s is not None else ""))


# ----------------------------------------------------------------------------
# History storage (JSON file + saved images)
# ----------------------------------------------------------------------------
def load_history():
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except Exception:
        return []


def save_history(records):
    HISTORY_IMG_DIR.mkdir(parents=True, exist_ok=True)
    tmp = HISTORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    os.replace(tmp, HISTORY_FILE)  # atomic: never leaves a half-written file


def _save_history_image(img, name, max_side=900):
    HISTORY_IMG_DIR.mkdir(parents=True, exist_ok=True)
    im = img.convert("RGB")
    im.thumbnail((max_side, max_side))
    path = HISTORY_IMG_DIR / f"{name}.jpg"
    im.save(path, "JPEG", quality=88)
    return path.name


def add_record(rec, images=None):
    rid = uuid.uuid4().hex[:10]
    full = {
        "id": rid,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "user": st.session_state.display_name,
        **rec,
    }
    if images:
        full["images"] = {
            key: _save_history_image(img, f"{rid}_{key}")
            for key, img in images.items() if img is not None
        }
    records = load_history()
    records.insert(0, full)
    save_history(records)
    return full


def delete_record(rid):
    records = load_history()
    keep = []
    for r in records:
        if r.get("id") == rid:
            for fname in (r.get("images") or {}).values():
                try:
                    (HISTORY_IMG_DIR / fname).unlink(missing_ok=True)
                except Exception:
                    pass
        else:
            keep.append(r)
    save_history(keep)


def clear_history():
    for r in load_history():
        for fname in (r.get("images") or {}).values():
            try:
                (HISTORY_IMG_DIR / fname).unlink(missing_ok=True)
            except Exception:
                pass
    save_history([])


def fmt_ts(ts):
    try:
        return datetime.fromisoformat(ts).strftime("%d %b %Y, %I:%M %p")
    except Exception:
        return str(ts)


# ----------------------------------------------------------------------------
# Theme
# ----------------------------------------------------------------------------
def inject_css(dark: bool):
    if dark:
        p = dict(
            bg="#0d1218", surface="#151c25", surface2="#1b2430", text="#eef3f9", muted="#9aa8b8",
            border="#27323f", input_bg="#101720", blue="#3b82f6", teal="#14b8a6",
            success="#22c55e", danger="#ef4444", warn="#f59e0b",
            success_bg="#10281a", danger_bg="#2a1517", warn_bg="#2b2313",
            density_bg="#102134", view_bg="#0f2a1f", later_bg="#211a33",
            bar_track="#232d39", shadow="0 4px 18px rgba(0,0,0,.35)",
            side1="#0a0f15", side2="#101a26",
        )
    else:
        p = dict(
            bg="#f1f5f9", surface="#ffffff", surface2="#f6f9fc", text="#16263a", muted="#68798c",
            border="#dbe4ee", input_bg="#ffffff", blue="#0b66c3", teal="#0e9f8e",
            success="#16a34a", danger="#dc2626", warn="#f59e0b",
            success_bg="#ecfaf1", danger_bg="#fff3f2", warn_bg="#fff8ec",
            density_bg="#eaf3ff", view_bg="#e9f9f1", later_bg="#f3edfe",
            bar_track="#e8eef5", shadow="0 4px 18px rgba(16,42,70,.08)",
            side1="#0a2540", side2="#0d3a63",
        )

    root = f"""
    <style>
      :root {{
        --bg:{p['bg']}; --surface:{p['surface']}; --surface2:{p['surface2']}; --text:{p['text']};
        --muted:{p['muted']}; --border:{p['border']}; --input-bg:{p['input_bg']}; --blue:{p['blue']};
        --teal:{p['teal']}; --success:{p['success']}; --danger:{p['danger']}; --warn:{p['warn']};
        --success-bg:{p['success_bg']}; --danger-bg:{p['danger_bg']}; --warn-bg:{p['warn_bg']};
        --density-bg:{p['density_bg']}; --view-bg:{p['view_bg']}; --later-bg:{p['later_bg']};
        --bar-track:{p['bar_track']}; --shadow:{p['shadow']};
        --side1:{p['side1']}; --side2:{p['side2']};
      }}
    </style>
    """
    st.markdown(root, unsafe_allow_html=True)

    st.markdown("""
    <style>
      html, body, [data-testid="stAppViewContainer"] { background:var(--bg)!important; color:var(--text)!important; }
      [data-testid="stHeader"] { background:transparent!important; }
      #MainMenu, footer { visibility:hidden; }
      .block-container { max-width:1320px!important; padding-top:1.2rem!important; padding-bottom:2rem!important; }
      [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span,
      [data-testid="stMarkdownContainer"] label, .stTextInput label, .stNumberInput label,
      .stSelectbox label, .stSlider label, .stFileUploader label, .stCheckbox label { color:var(--text)!important; }

      /* ---------- Sidebar ---------- */
      [data-testid="stSidebar"] { background:linear-gradient(180deg,var(--side1),var(--side2))!important; border-right:0!important; }
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span,
      [data-testid="stSidebar"] label { color:#e8f0f8!important; }
      [data-testid="stSidebar"] .stButton > button {
        justify-content:flex-start!important; text-align:left!important; height:44px!important;
        background:transparent!important; color:#d6e3f1!important; border:1px solid transparent!important;
        border-radius:10px!important; font-size:13.5px!important; font-weight:600!important; padding:0 14px!important;
      }
      [data-testid="stSidebar"] .stButton > button:hover { background:rgba(255,255,255,.08)!important; color:#fff!important; border-color:transparent!important; }
      [data-testid="stSidebar"] .stButton > button[kind="primary"],
      [data-testid="stSidebar"] .stButton > button[data-testid="stBaseButton-primary"] {
        background:rgba(255,255,255,.16)!important; color:#fff!important;
        box-shadow:inset 3px 0 0 #5eead4!important; border-color:transparent!important;
      }
      .sb-brand { display:flex; align-items:center; gap:12px; padding:6px 4px 16px; border-bottom:1px solid rgba(255,255,255,.12); margin-bottom:14px; }
      .sb-logo { width:42px;height:42px;border-radius:12px;background:linear-gradient(135deg,#14b8a6,#3b82f6);display:flex;align-items:center;justify-content:center;font-size:22px;font-weight:800;color:#fff!important;box-shadow:0 6px 16px rgba(0,0,0,.25); }
      .sb-name { font-size:15px; font-weight:800; color:#fff!important; line-height:1.15; }
      .sb-tag { font-size:10.5px; color:#9fb6cd!important; }
      .sb-label { font-size:10px; font-weight:800; letter-spacing:1.2px; color:#7f9bb7!important; margin:14px 4px 6px; }
      .sb-user { display:flex; align-items:center; gap:10px; background:rgba(255,255,255,.08); border-radius:12px; padding:10px 12px; margin:8px 0 12px; }
      .sb-avatar { width:34px;height:34px;border-radius:50%;background:linear-gradient(135deg,#14b8a6,#3b82f6);display:flex;align-items:center;justify-content:center;font-weight:800;color:#fff!important; }
      .sb-uname { font-size:13px; font-weight:700; color:#fff!important; }
      .sb-urole { font-size:10.5px; color:#9fb6cd!important; }
      .sb-foot { font-size:9.5px; color:#7f9bb7!important; text-align:center; margin-top:14px; line-height:1.5; }

      /* ---------- Page header ---------- */
      .page-head { background:linear-gradient(110deg,var(--blue),var(--teal)); color:#fff; border-radius:16px; padding:20px 26px; margin-bottom:18px;
        display:flex; align-items:center; justify-content:space-between; box-shadow:var(--shadow); }
      .ph-title { font-size:21px; font-weight:800; letter-spacing:-.2px; color:#fff!important; }
      .ph-sub { font-size:12.5px; color:rgba(255,255,255,.88)!important; margin-top:3px; }
      .ph-date { font-size:11.5px; font-weight:600; background:rgba(255,255,255,.18); padding:7px 12px; border-radius:20px; color:#fff!important; white-space:nowrap; }

      /* ---------- Cards ---------- */
      [data-testid="stVerticalBlockBorderWrapper"] { background:var(--surface)!important; border:1px solid var(--border)!important; border-radius:14px!important; box-shadow:var(--shadow)!important; }
      .section-title { font-size:15px;font-weight:800;margin-bottom:2px;color:var(--text); display:flex; align-items:center; gap:8px; }
      .section-sub { font-size:11.5px;color:var(--muted);margin-bottom:12px; }

      .stat-card { background:var(--surface); border:1px solid var(--border); border-radius:14px; padding:16px 18px; box-shadow:var(--shadow); position:relative; overflow:hidden; }
      .stat-card::before { content:""; position:absolute; left:0; top:0; bottom:0; width:4px; background:var(--accent,var(--blue)); }
      .stat-icon { font-size:20px; margin-bottom:6px; }
      .stat-value { font-size:28px; font-weight:800; color:var(--text); line-height:1.1; }
      .stat-label { font-size:11.5px; color:var(--muted); font-weight:600; margin-top:3px; }

      .recent-row { display:flex; align-items:center; gap:12px; padding:11px 4px; border-bottom:1px solid var(--border); }
      .recent-row:last-child { border-bottom:0; }
      .recent-ico { width:36px;height:36px;border-radius:10px;background:var(--surface2);display:flex;align-items:center;justify-content:center;font-size:17px; }
      .recent-main { flex:1; min-width:0; }
      .recent-name { font-size:13px; font-weight:700; color:var(--text); }
      .recent-sub { font-size:11px; color:var(--muted); }
      .pill { display:inline-block; font-size:11px; font-weight:700; padding:4px 10px; border-radius:20px; }
      .pill.good { background:var(--success-bg); color:var(--success); border:1px solid var(--success); }
      .pill.bad { background:var(--danger-bg); color:var(--danger); border:1px solid var(--danger); }
      .empty-box { min-height:150px;border:1px dashed var(--border);border-radius:10px;background:var(--surface2);display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12.5px;text-align:center;padding:16px; }

      /* ---------- Buttons / inputs ---------- */
      .stButton > button { border-radius:9px!important;font-size:13px!important;font-weight:700!important;background:var(--surface)!important;color:var(--text)!important;border:1px solid var(--border)!important; }
      .stButton > button:hover { border-color:var(--blue)!important;color:var(--blue)!important; }
      .stButton > button[kind="primary"], .stButton > button[data-testid="stBaseButton-primary"],
      [data-testid="stFormSubmitButton"] button { background:linear-gradient(110deg,var(--blue),var(--teal))!important;border:0!important;color:#fff!important; }
      .stButton > button[kind="primary"]:hover { color:#fff!important; filter:brightness(1.07); }
      [data-testid="stDownloadButton"] button { border-radius:9px!important; font-weight:700!important; background:var(--surface)!important; color:var(--text)!important; border:1px solid var(--border)!important; }
      [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { background:var(--surface)!important;color:var(--text)!important;border-color:var(--border)!important; }
      .stTextInput input, .stNumberInput input { background:var(--input-bg)!important;color:var(--text)!important;border-color:var(--border)!important; }
      div[data-baseweb="select"] > div { background:var(--input-bg)!important; color:var(--text)!important; border-color:var(--border)!important; }
      .stToggle label, .stCheckbox label { color:var(--text)!important; }

      /* ---------- Vital result ---------- */
      .predict-result-title { font-size:14px;font-weight:800;margin-bottom:10px;color:var(--text); }
      .risk-badge { border-radius:12px; padding:14px 16px; margin-bottom:16px; }
      .risk-badge.low { background:var(--success-bg); border:1px solid var(--success); }
      .risk-badge.high { background:var(--danger-bg); border:1px solid var(--danger); }
      .risk-badge-label { font-size:17px; font-weight:800; display:flex; align-items:center; gap:8px; margin-bottom:3px; }
      .risk-badge-label.low { color:var(--success); }
      .risk-badge-label.high { color:var(--danger); }
      .risk-badge-sub { font-size:11.5px; color:var(--muted); }
      .prob-heading { font-size:12.5px; font-weight:800; color:var(--text); margin:4px 0 10px; }
      .prob-row { display:flex; align-items:center; gap:10px; margin-bottom:9px; font-size:11.5px; }
      .prob-dot { width:8px;height:8px;border-radius:50%;flex-shrink:0; }
      .prob-label { width:78px; flex-shrink:0; color:var(--text); }
      .prob-track { flex:1; height:8px; background:var(--bar-track); border-radius:5px; overflow:hidden; }
      .prob-fill { display:block; height:100%; border-radius:5px; }
      .prob-pct { width:44px; flex-shrink:0; text-align:right; font-weight:700; color:var(--text); }
      .info-note { margin-top:14px; background:var(--density-bg); border:1px solid var(--border); border-radius:10px; padding:10px 12px; font-size:10.5px; color:var(--muted); line-height:1.5; }
      .saved-note { margin-top:10px; font-size:11px; color:var(--success); font-weight:700; }

      /* ---------- Breast page ---------- */
      div[data-testid="stFileUploader"] { border:1px solid var(--border);border-radius:10px;padding:5px;background:var(--surface)!important; }
      [data-testid="stFileUploaderDropzone"] { background:var(--surface2)!important; border-color:var(--border)!important; }
      [data-testid="stFileUploaderDropzone"] button { background:var(--surface)!important;color:var(--text)!important;border:1px solid var(--border)!important; }
      [data-testid="stFileUploaderDropzoneInstructions"] { color:var(--text)!important; }
      [data-testid="stFileUploaderDropzoneInstructions"] span, [data-testid="stFileUploaderDropzoneInstructions"] small { color:var(--muted)!important; }
      .img-card-title { font-size:12px; font-weight:800; color:var(--text); margin-bottom:6px; }
      .img-frame { border:1px solid var(--border); border-radius:10px; overflow:hidden; background:#0c0e12; min-height:210px; display:flex; align-items:center; justify-content:center; }
      [data-testid="stImage"] img { width:100%!important; border:1px solid var(--border); border-radius:10px; display:block; }
      .img-empty { color:#8a94a3; font-size:11px; text-align:center; padding:10px; }
      .img-caption { font-size:10px; color:var(--muted); margin-top:6px; display:flex; align-items:center; gap:5px; }
      .legend-box { width:9px;height:9px;border-radius:2px; }
      .threshold-note { font-size:11px; color:var(--muted); margin:-6px 0 10px; }
      .colorbar { width:14px; height:210px; border-radius:4px; background:linear-gradient(to top, #2b2fb3, #1f9dfb, #21e07a, #f4e12b, #ff8a1e, #d61f1f); }
      .colorbar-wrap { display:flex; flex-direction:column; align-items:center; justify-content:space-between; height:210px; font-size:9.5px; color:var(--muted); font-weight:700; }

      .breast-result-card { border-radius:12px; padding:14px 16px; border:1px solid var(--danger); background:var(--danger-bg); height:100%; }
      .breast-result-card.negative { border-color:var(--success); background:var(--success-bg); }
      .breast-result-kicker { font-size:10.5px; font-weight:800; color:var(--danger); margin-bottom:4px; }
      .breast-result-card.negative .breast-result-kicker { color:var(--success); }
      .breast-result-value { font-size:18px; font-weight:800; color:var(--danger); margin-bottom:4px; }
      .breast-result-card.negative .breast-result-value { color:var(--success); }
      .breast-result-meta { font-size:11px; color:var(--text); margin-bottom:8px; }
      .breast-result-bar { height:6px; border-radius:4px; background:var(--bar-track); overflow:hidden; }
      .breast-result-bar-fill { height:100%; background:var(--danger); border-radius:4px; }
      .breast-result-card.negative .breast-result-bar-fill { background:var(--success); }

      .meta-card { border-radius:12px; padding:14px 16px; height:100%; border:1px solid var(--border); }
      .meta-card .meta-icon { font-size:15px; margin-bottom:6px; }
      .meta-card .meta-label { font-size:10.5px; font-weight:800; color:var(--muted); margin-bottom:4px; }
      .meta-card .meta-value { font-size:14px; font-weight:800; color:var(--text); }
      .meta-card .meta-sub { font-size:10px; color:var(--muted); margin-top:2px; }
      .meta-card.density { background:var(--density-bg); }
      .meta-card.view { background:var(--view-bg); }
      .meta-card.later { background:var(--later-bg); }
      .meta-card.plain { background:var(--surface2); }

      .stat-card, .meta-card { transition:transform .15s ease, box-shadow .15s ease; }
      .stat-card:hover { transform:translateY(-2px); }
      .rv-row { display:flex; align-items:center; gap:8px; padding:8px 2px; border-bottom:1px solid var(--border); font-size:12px; }
      .rv-row:last-of-type { border-bottom:0; }
      .rv-lab { flex:1.3; font-weight:700; color:var(--text); }
      .rv-val { flex:1; font-weight:800; color:var(--text); }
      .rv-val small { font-weight:600; color:var(--muted); }
      .rv-rng { flex:.9; font-size:10.5px; color:var(--muted); }
      .chip { font-size:10.5px; font-weight:800; padding:3px 9px; border-radius:20px; min-width:56px; text-align:center; }
      .chip.ok { background:var(--success-bg); color:var(--success); }
      .chip.low { background:var(--warn-bg); color:var(--warn); }
      .chip.high { background:var(--danger-bg); color:var(--danger); }
      .bars { display:flex; align-items:flex-end; justify-content:space-between; gap:10px; height:170px; padding-top:6px; }
      .bar-col { flex:1; display:flex; flex-direction:column; align-items:center; justify-content:flex-end; height:100%; }
      .bar-num { font-size:11px; font-weight:800; color:var(--text); height:16px; }
      .bar-stack { width:100%; max-width:46px; height:120px; display:flex; flex-direction:column; justify-content:flex-end; border-radius:8px 8px 3px 3px; overflow:hidden; background:var(--bar-track); }
      .bar-n { background:linear-gradient(180deg,var(--blue),var(--teal)); }
      .bar-f { background:var(--danger); }
      .bar-lab { font-size:10.5px; color:var(--muted); font-weight:700; margin-top:6px; }
      .legend { display:flex; gap:14px; font-size:10.5px; color:var(--muted); margin-top:8px; }
      .legend i { display:inline-block; width:9px; height:9px; border-radius:2px; margin-right:5px; }
      .small-note { color:var(--muted);font-size:10px;line-height:1.5; }
      .footer-note { text-align:center;color:var(--muted);font-size:10px;padding:18px 0 4px; }
    </style>
    """, unsafe_allow_html=True)


inject_css(st.session_state.dark_mode)


class MultiTaskEfficientNetB0(nn.Module):
    def __init__(self, density_classes=4, view_classes=6, laterality_classes=2):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        feature_dim = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.cancer_head = nn.Sequential(nn.Dropout(0.40), nn.Linear(feature_dim, 1))
        self.density_head = nn.Sequential(nn.Dropout(0.30), nn.Linear(feature_dim, density_classes))
        self.view_head = nn.Sequential(nn.Dropout(0.30), nn.Linear(feature_dim, view_classes))
        self.laterality_head = nn.Sequential(nn.Dropout(0.30), nn.Linear(feature_dim, laterality_classes))

    def forward(self, image):
        features = self.backbone(image)
        return {
            "cancer": self.cancer_head(features),
            "density": self.density_head(features),
            "view": self.view_head(features),
            "laterality": self.laterality_head(features),
        }


class CancerOutputWrapper(nn.Module):
    def __init__(self, model):
        super().__init__(); self.model = model
    def forward(self, x): return self.model(x)["cancer"]


@st.cache_resource(show_spinner=False)
def load_vital_model():
    if not VITAL_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {VITAL_MODEL_PATH.name}")
    return joblib.load(VITAL_MODEL_PATH)


@st.cache_resource(show_spinner=False)
def load_breast_assets():
    if not BREAST_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing {BREAST_MODEL_PATH.name}")
    config = {}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f: config = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Keep CPU inference predictable on memory-constrained machines.
    if device.type == "cpu":
        torch.set_num_threads(min(2, os.cpu_count() or 1))
        torch.set_num_interop_threads(1)

    model = MultiTaskEfficientNetB0(4, 6, 2).to(device)
    state = torch.load(BREAST_MODEL_PATH, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, config, device


class SquarePad:
    def __call__(self, image):
        w,h=image.size; size=max(w,h)
        left=(size-w)//2; top=(size-h)//2
        right=size-w-left; bottom=size-h-top
        return transforms.functional.pad(image,(left,top,right,bottom),fill=0)


def breast_transform(image, image_size=768):
    return transforms.Compose([
        transforms.Resize(image_size), SquarePad(), transforms.Resize((image_size,image_size)),
        transforms.ToTensor(), transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    ])(image)


def get_breast_mask(image_rgb):
    arr=np.asarray(image_rgb.convert("L"))
    if cv2 is None: return np.ones_like(arr,dtype=np.uint8)*255
    mask=(arr>8).astype(np.uint8)*255
    kernel=np.ones((7,7),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel,iterations=2)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel,iterations=1)
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
    if n>1:
        idx=1+np.argmax(stats[1:,cv2.CC_STAT_AREA]); mask=np.where(labels==idx,255,0).astype(np.uint8)
    return mask


def get_poi_box_from_cam(grayscale_cam, original_image, percentile=88):
    if grayscale_cam is None or grayscale_cam.max() <= 0:
        return None, None

    cam = np.nan_to_num(np.asarray(grayscale_cam, dtype=np.float32))
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Keep attention inside breast tissue and suppress the breast/background
    # boundary, which can otherwise be selected as a false POI.
    mask = get_breast_mask(original_image)
    mask_small = np.array(
        Image.fromarray(mask).resize(
            (cam.shape[1], cam.shape[0]),
            Image.Resampling.NEAREST
        )
    )
    breast = mask_small > 0

    if cv2 is not None:
        safe_mask = cv2.erode(
            mask_small, np.ones((11, 11), np.uint8), iterations=1
        ) > 0
    else:
        safe_mask = breast.copy()

    # Ignore a thin image border.
    mx = max(2, int(cam.shape[1] * 0.035))
    my = max(2, int(cam.shape[0] * 0.035))
    safe_mask[:my, :] = False
    safe_mask[-my:, :] = False
    safe_mask[:, :mx] = False
    safe_mask[:, -mx:] = False

    masked_cam = np.where(safe_mask, cam, 0)
    values = masked_cam[safe_mask]

    if values.size == 0 or float(values.max()) <= 0:
        return None, masked_cam

    threshold = max(0.45, float(np.percentile(values, percentile)))
    binary = ((masked_cam >= threshold) & safe_mask).astype(np.uint8)

    if cv2 is None:
        ys, xs = np.where(binary > 0)
        if len(xs) == 0:
            return None, masked_cam
        x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()
    else:
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        candidates = []
        total = binary.shape[0] * binary.shape[1]

        for i in range(1, n):
            x, y, w, h, area = stats[i]
            frac = area / max(total, 1)

            if area >= 8 and 0.0001 <= frac <= 0.12:
                comp = labels == i
                mean_score = float(masked_cam[comp].mean())
                max_score = float(masked_cam[comp].max())
                score = 0.75 * mean_score + 0.25 * max_score

                # Slightly prefer regions away from the image boundary.
                cx = x + w / 2.0
                cy = y + h / 2.0
                edge_dist = min(
                    cx / max(cam.shape[1], 1),
                    cy / max(cam.shape[0], 1),
                    (cam.shape[1] - cx) / max(cam.shape[1], 1),
                    (cam.shape[0] - cy) / max(cam.shape[0], 1),
                )
                score *= 0.75 + 0.25 * min(edge_dist / 0.15, 1.0)
                candidates.append((score, x, y, w, h))

        if candidates:
            _, x1, y1, w, h = max(candidates, key=lambda t: t[0])
            x2, y2 = x1 + w - 1, y1 + h - 1
        else:
            ys, xs = np.where(binary > 0)
            if len(xs) == 0:
                return None, masked_cam
            x1, x2, y1, y2 = xs.min(), xs.max(), ys.min(), ys.max()

    pad_x = int((x2 - x1 + 1) * 0.12)
    pad_y = int((y2 - y1 + 1) * 0.12)
    x1 = max(0, x1 - pad_x)
    x2 = min(cam.shape[1] - 1, x2 + pad_x)
    y1 = max(0, y1 - pad_y)
    y2 = min(cam.shape[0] - 1, y2 + pad_y)

    ow, oh = original_image.size
    sx, sy = ow / cam.shape[1], oh / cam.shape[0]

    return (
        int(x1 * sx),
        int(y1 * sy),
        max(1, int((x2 + 1) * sx - x1 * sx)),
        max(1, int((y2 + 1) * sy - y1 * sy)),
    ), masked_cam


def compute_gradcam(model, input_tensor, original_image, cam_size=384):
    """Returns (box_in_original_coords, grayscale_cam) or (None, None).

    The cancer prediction is still made at the configured 768x768 resolution.
    Grad-CAM is computed at a smaller resolution on CPU to avoid large peak
    RAM usage from storing gradients/activations.
    """
    if not HAS_GRADCAM:
        return None, None

    cam_model = CancerOutputWrapper(model)
    target_layers = [cam_model.model.backbone.features[-1]]

    # Grad-CAM needs gradients, so do not wrap this block in no_grad().
    cam_tensor = input_tensor
    if cam_size and input_tensor.shape[-1] != cam_size:
        cam_tensor = torch.nn.functional.interpolate(
            input_tensor, size=(cam_size, cam_size),
            mode="bilinear", align_corners=False
        )

    try:
        with GradCAM(model=cam_model, target_layers=target_layers) as cam:
            grayscale_cam = cam(
                input_tensor=cam_tensor,
                targets=[ClassifierOutputTarget(0)]
            )[0]

        box, _ = get_poi_box_from_cam(grayscale_cam, original_image, 90)
        return box, grayscale_cam

    except RuntimeError as e:
        # A Grad-CAM memory failure should not invalidate the actual cancer
        # prediction. Return no visualization instead of crashing the page.
        if "out of memory" in str(e).lower() or "not enough memory" in str(e).lower():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return None, None
        raise
    finally:
        del cam_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_bbox_image(image, box):
    """Image 2: original mammogram with just the model-derived bounding box."""
    img = image.convert("RGB").copy()
    if box is None:
        return img
    draw = ImageDraw.Draw(img)
    x, y, w, h = box
    sw = max(3, int(min(img.size) * 0.012))
    draw.rectangle([x, y, x + w, y + h], outline=(240, 40, 40), width=sw)
    return img


def build_gradcam_image(image, grayscale_cam, box):
    """Image 3: Grad-CAM heatmap overlay (jet colormap) + bounding box."""
    img = image.convert("RGB")
    w, h = img.size
    base = np.array(img).astype(np.float32)

    if grayscale_cam is None:
        return Image.fromarray(base.astype(np.uint8))

    cam = np.nan_to_num(np.asarray(grayscale_cam, dtype=np.float32))
    cam = np.maximum(cam, 0)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    # Prevent background/air around the breast from appearing as high attention.
    breast_mask = get_breast_mask(image) > 0

    if cv2 is not None:
        cam_resized = cv2.resize(cam, (w, h))
        cam_resized = cam_resized * breast_mask.astype(np.float32)
        heatmap_bgr = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    else:
        cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h))
        cam_resized = np.array(cam_img).astype(np.float32) / 255.0
        heatmap = np.stack(
            [cam_resized * 255, np.zeros_like(cam_resized), (1 - cam_resized) * 255], axis=-1
        )

    blended = (0.55 * base + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    out = Image.fromarray(blended)

    if box is not None:
        draw = ImageDraw.Draw(out)
        x, y, bw, bh = box
        sw = max(3, int(min(w, h) * 0.012))
        draw.rectangle([x, y, x + bw, y + bh], outline=(255, 255, 255), width=sw)
    return out


def read_config():
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
def page_header(icon, title, sub):
    today = datetime.now().strftime("%A, %d %B %Y")
    st.markdown(
        f'<div class="page-head"><div><div class="ph-title">{icon} {esc(title)}</div>'
        f'<div class="ph-sub">{esc(sub)}</div></div><div class="ph-date">{today}</div></div>',
        unsafe_allow_html=True,
    )


def meta_card(cls, icon, label, value, sub="&nbsp;"):
    return (
        f'<div class="meta-card {cls}"><div class="meta-icon">{icon}</div>'
        f'<div class="meta-label">{esc(label)}</div>'
        f'<div class="meta-value">{esc(value)}</div>'
        f'<div class="meta-sub">{sub}</div></div>'
    )


def pill(label, positive):
    cls = "bad" if positive else "good"
    return f'<span class="pill {cls}">{esc(label)}</span>'


VITAL_RANGES = {
    "hr": ("Heart Rate", "bpm", 60, 100, "{:.0f}"),
    "resp": ("Respiratory Rate", "/min", 12, 20, "{:.0f}"),
    "spo2": ("SpO\u2082", "%", 95, 100, "{:.0f}"),
    "temp": ("Temperature", "\u00b0C", 36.1, 37.5, "{:.1f}"),
    "sbp": ("Systolic BP", "mmHg", 90, 129, "{:.0f}"),
    "dbp": ("Diastolic BP", "mmHg", 60, 84, "{:.0f}"),
}


def vital_flags(inp):
    """Compare each vital against general adult resting reference ranges."""
    out = []
    for key, (lab, unit, lo, hi, fmt) in VITAL_RANGES.items():
        if inp.get(key) is None:
            continue
        v = float(inp[key])
        status = "Low" if v < lo else "High" if v > hi else "Normal"
        out.append((lab, fmt.format(v), unit, f"{fmt.format(lo)}\u2013{fmt.format(hi)}", status))
    return out


def render_vital_review(inp):
    rows = ""
    for lab, val, unit, rng, status in vital_flags(inp):
        cls = {"Normal": "ok", "Low": "low", "High": "high"}[status]
        rows += (f'<div class="rv-row"><span class="rv-lab">{esc(lab)}</span>'
                 f'<span class="rv-val">{val} <small>{esc(unit)}</small></span>'
                 f'<span class="rv-rng">{rng}</span><span class="chip {cls}">{status}</span></div>')
    st.markdown(rows + '<div class="small-note" style="margin-top:6px;">Reference: general adult resting ranges.</div>',
                unsafe_allow_html=True)


def render_prob_bar(label, pct, color):
    st.markdown(
        f'<div class="prob-row">'
        f'<span class="prob-dot" style="background:{color};"></span>'
        f'<span class="prob-label">{label}</span>'
        f'<span class="prob-track"><span class="prob-fill" style="width:{pct*100:.1f}%;background:{color};"></span></span>'
        f'<span class="prob-pct">{pct*100:.1f}%</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def _risk_breakdown(proba, classes=None):
    """Turn predict_proba row into (low, high). Medium is folded into High."""
    proba = list(proba)
    if len(proba) >= 3:
        low = float(proba[0]); high = float(proba[1]) + float(proba[2])
    elif len(proba) == 2:
        low, high = float(proba[0]), float(proba[1])
    else:
        high = min(max(float(proba[0]), 0.0), 1.0); low = 1.0 - high
    total = low + high or 1.0
    return low / total, high / total


def render_vital_result(r, saved=False):
    icon = "\u2705" if r["level"] == "low" else "\U0001F6A8"
    st.markdown(
        f'<div class="risk-badge {r["level"]}">'
        f'<div class="risk-badge-label {r["level"]}">{icon} {esc(r["label"])}</div>'
        f'<div class="risk-badge-sub">Risk Probability: {r["high"]*100:.1f}%</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="prob-heading">Model Probabilities</div>', unsafe_allow_html=True)
    if r["level"] == "low":
        render_prob_bar("Low Risk", r["low"], "var(--success)")
        render_prob_bar("High Risk", 0.0, "var(--danger)")
    else:
        render_prob_bar("Low Risk", 0.0, "var(--success)")
        render_prob_bar("High Risk", r["high"], "var(--danger)")
    if saved:
        st.markdown('<div class="saved-note">\u2713 Saved to Test History</div>', unsafe_allow_html=True)


def render_breast_cards(r):
    is_cancer = r["is_cancer"]
    c_result, c_density, c_view, c_later = st.columns([1.3, 1, 1, 1], gap="medium")
    with c_result:
        icon = "\u26a0\ufe0f" if is_cancer else "\u2705"
        neg = "" if is_cancer else " negative"
        st.markdown(
            f'<div class="breast-result-card{neg}">'
            f'<div class="breast-result-kicker">{icon} Prediction Results</div>'
            f'<div class="breast-result-value">{esc(r["label"])}</div>'
            f'<div class="breast-result-meta">Cancer Probability: {r["prob"]:.2f} ({r["prob"]*100:.1f}%)</div>'
            f'<div class="breast-result-bar"><div class="breast-result-bar-fill" style="width:{min(r["prob"]*100,100):.1f}%;"></div></div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with c_density:
        st.markdown(meta_card("density", "\U0001f5c3\ufe0f", "Breast Density", r["density_name"], f'(Category {esc(r["density_code"])})'), unsafe_allow_html=True)
    with c_view:
        st.markdown(meta_card("view", "\U0001f5bc\ufe0f", "View", r["view_code"], f'({esc(r["view_name"])})'), unsafe_allow_html=True)
    with c_later:
        st.markdown(meta_card("later", "\U0001f9ed", "Laterality", r["laterality"]), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Navigation / sidebar
# ----------------------------------------------------------------------------
def _go(page):
    st.session_state.page = page


def _sync_dark_from_sidebar():
    st.session_state.dark_mode = bool(st.session_state.sb_dark)


def _logout():
    st.session_state.authenticated = False
    st.session_state.vital_result = None
    st.session_state.breast_result = None
    st.session_state.page = "dashboard"


def render_sidebar():
    name = st.session_state.display_name or USERNAME
    initial = name[:1].upper() if name else "U"
    with st.sidebar:
        st.markdown(
            '<div class="sb-brand"><div class="sb-logo">\u271a</div>'
            '<div><div class="sb-name">Hospital AI System</div>'
            '<div class="sb-tag">Clinical Decision Support</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="sb-user"><div class="sb-avatar">{esc(initial)}</div>'
            f'<div><div class="sb-uname">{esc(name)}</div><div class="sb-urole">Medical Staff</div></div></div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sb-label">MAIN MENU</div>', unsafe_allow_html=True)
        for key, label in PAGES:
            st.button(
                label, key=f"nav_{key}", use_container_width=True,
                type="primary" if st.session_state.page == key else "secondary",
                on_click=_go, args=(key,),
            )

        n_total = len(load_history())
        st.markdown(
            f'<div class="sb-label">SESSION</div>'
            f'<div class="sb-foot" style="text-align:left;font-size:11.5px;margin:0 4px 10px;">'
            f'Saved tests: <b style="color:#fff">{n_total}</b></div>',
            unsafe_allow_html=True,
        )
        st.toggle("Dark Mode", value=st.session_state.dark_mode, key="sb_dark", on_change=_sync_dark_from_sidebar)
        st.button("\U0001F6AA  Logout", key="sb_logout", use_container_width=True, on_click=_logout)
        st.markdown(
            '<div class="sb-foot">AI-assisted screening.<br>Not a substitute for professional clinical judgment.</div>',
            unsafe_allow_html=True,
        )


# ----------------------------------------------------------------------------
# Login
# ----------------------------------------------------------------------------
def _sync_login_theme():
    st.session_state.dark_mode = bool(st.session_state.login_dark)


def login_screen():
    dark = st.session_state.dark_mode
    blob1, blob2 = ("#16324a", "#143a36") if dark else ("#d6e8ff", "#d8f5ef")
    st.markdown(
        f"""
        <style>
          [data-testid="stSidebar"], [data-testid="collapsedControl"], [data-testid="stSidebarCollapsedControl"] {{ display:none!important; }}
          [data-testid="stAppViewContainer"] {{
            background:
              radial-gradient(560px 420px at 8% 0%, {blob1}, transparent 60%),
              radial-gradient(520px 380px at 96% 92%, {blob2}, transparent 55%),
              var(--bg) !important;
          }}
          .login-shell {{ max-width:460px; margin:8px auto 0; }}
          .login-plus {{ width:56px;height:56px;margin:4px auto 14px;border-radius:16px;
            background:linear-gradient(135deg,var(--blue),var(--teal));color:#fff;display:flex;
            align-items:center;justify-content:center;font-size:28px;font-weight:700;
            box-shadow:0 8px 20px rgba(11,102,195,.35); }}
          .login-title {{ text-align:center;font-size:22px;font-weight:800;color:var(--text);margin-bottom:4px; letter-spacing:-.2px; }}
          .login-tagline {{ text-align:center;font-size:12.5px;color:var(--muted);margin-bottom:22px; }}
          .login-sub {{ text-align:center;font-size:10px;color:var(--muted);margin-top:10px; }}
          .login-card-inner [data-testid="stForm"] {{ border:0!important; padding:0!important; background:transparent!important; box-shadow:none!important; }}
          .login-card-inner [data-testid="stTextInput"] label {{ color:var(--text)!important; font-size:12px!important; font-weight:600!important; }}
          .login-card-inner [data-testid="stTextInput"] input {{ height:40px!important; font-size:13px!important; border-radius:9px!important; }}
          .login-card-inner [data-testid="stFormSubmitButton"] button {{ height:42px!important; font-size:13px!important; font-weight:700!important; letter-spacing:.3px; border-radius:9px!important; margin-top:4px; }}
          .login-theme-row {{ display:flex; justify-content:flex-end; max-width:460px; margin:0 auto; }}
          .feature-strip {{ display:flex; justify-content:space-between; gap:10px; max-width:460px; margin:26px auto 0; }}
          .feature-chip {{ flex:1; text-align:center; padding:14px 8px; background:var(--surface);
            border:1px solid var(--border); border-radius:12px; box-shadow:var(--shadow); }}
          .feature-chip .fc-icon {{ font-size:19px; margin-bottom:6px; }}
          .feature-chip .fc-label {{ font-size:10.5px; font-weight:700; color:var(--muted); line-height:1.3; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="login-theme-row">', unsafe_allow_html=True)
    st.toggle("Dark Mode", value=dark, key="login_dark", on_change=_sync_login_theme)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="login-shell login-card-inner">', unsafe_allow_html=True)
    with st.container(border=True):
        st.markdown(
            '<div style="padding:22px 22px 6px;">'
            '<div class="login-plus">\u271a</div>'
            '<div class="login-title">Hospital AI System</div>'
            '<div class="login-tagline">AI-assisted screening for better healthcare</div>'
            '</div>',
            unsafe_allow_html=True,
        )
        col_pad, col_form, col_pad2 = st.columns([0.08, 1, 0.08])
        with col_form:
            with st.form("login_form", clear_on_submit=False):
                username = st.text_input("\U0001F464 Username", placeholder="Enter your username")
                password = st.text_input("\U0001F512 Password", type="password")
                submitted = st.form_submit_button("LOG IN", type="primary", use_container_width=True)
            st.markdown('<div class="login-sub">Authorized hospital staff only</div>', unsafe_allow_html=True)
            st.markdown('<div style="height:18px"></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(
        '<div class="feature-strip">'
        '<div class="feature-chip"><div class="fc-icon">\U0001F6E1\ufe0f</div><div class="fc-label">Secure Access</div></div>'
        '<div class="feature-chip"><div class="fc-icon">\U0001F9E0</div><div class="fc-label">AI-Powered Screening</div></div>'
        '<div class="feature-chip"><div class="fc-icon">\U0001F4CA</div><div class="fc-label">Clinical Support</div></div>'
        '</div>',
        unsafe_allow_html=True,
    )

    if submitted:
        if username == USERNAME and password == PASSWORD:
            st.session_state.authenticated = True
            st.session_state.display_name = username
            st.session_state.page = "dashboard"
            st.rerun()
        else:
            st.error("Invalid username or password.")


# ----------------------------------------------------------------------------
# Pages
# ----------------------------------------------------------------------------
def activity_chart_html(records, n=7):
    days = [date.today() - timedelta(days=i) for i in range(n - 1, -1, -1)]
    counts = {d.isoformat(): [0, 0] for d in days}
    for r in records:
        k = str(r.get("timestamp", ""))[:10]
        if k in counts:
            counts[k][1 if r.get("positive") else 0] += 1
    mx = max(1, max(sum(v) for v in counts.values()))
    cols = ""
    for d in days:
        nrm, flg = counts[d.isoformat()]
        tot = nrm + flg
        cols += (f'<div class="bar-col"><div class="bar-num">{tot if tot else ""}</div>'
                 f'<div class="bar-stack"><div class="bar-f" style="height:{flg / mx * 100:.1f}%"></div>'
                 f'<div class="bar-n" style="height:{nrm / mx * 100:.1f}%"></div></div>'
                 f'<div class="bar-lab">{d.strftime("%a")}</div></div>')
    legend = '<div class="legend"><span><i style="background:var(--teal)"></i>Normal</span><span><i style="background:var(--danger)"></i>Flagged</span></div>'
    return f'<div class="bars">{cols}</div>{legend}'


def dashboard_page():
    page_header("\U0001F3E5", f"Welcome back, {st.session_state.display_name or USERNAME}",
                "Overview of screening activity")
    records = load_history()
    today = date.today().isoformat()
    total = len(records)
    today_n = sum(1 for r in records if str(r.get("timestamp", ""))[:10] == today)
    vit = [r for r in records if r.get("type") == "vital"]
    brs = [r for r in records if r.get("type") == "breast"]
    flagged = sum(1 for r in records if r.get("positive"))

    stats = [
        ("\U0001F4CB", total, "Total Tests", "var(--blue)"),
        ("\U0001F4C5", today_n, "Tests Today", "var(--teal)"),
        ("\u2764\ufe0f", len(vit), "Vital-Signs Tests", "var(--warn)"),
        ("\U0001FA7B", len(brs), "Breast Scans", "#8b5cf6"),
        ("\U0001F6A8", flagged, "Flagged Cases", "var(--danger)"),
    ]
    cols = st.columns(len(stats), gap="medium")
    for col, (ico, val, lab, accent) in zip(cols, stats):
        with col:
            st.markdown(
                f'<div class="stat-card" style="--accent:{accent};"><div class="stat-icon">{ico}</div>'
                f'<div class="stat-value">{val}</div><div class="stat-label">{lab}</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div style="height:16px"></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.25], gap="medium")
    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">\u26a1 Quick Actions</div>'
                        '<div class="section-sub">Start a new screening</div>', unsafe_allow_html=True)
            st.button("\u2764\ufe0f  New Vital-Signs Screening", key="qa_vital", use_container_width=True,
                      type="primary", on_click=_go, args=("vital",))
            st.button("\U0001FA7B  New Breast Cancer Detection", key="qa_breast", use_container_width=True,
                      type="primary", on_click=_go, args=("breast",))
            st.button("\U0001F5C2\ufe0f  Open Test History", key="qa_hist", use_container_width=True,
                      on_click=_go, args=("history",))
    with right:
        with st.container(border=True):
            st.markdown('<div class="section-title">\U0001F4C8 Activity \u2014 Last 7 Days</div>'
                        '<div class="section-sub">Number of tests per day</div>', unsafe_allow_html=True)
            st.markdown(activity_chart_html(records), unsafe_allow_html=True)

    st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
    left2, right2 = st.columns([1, 1.25], gap="medium")
    with left2:
        with st.container(border=True):
            st.markdown('<div class="section-title">\U0001F3AF Flagged Rate</div>'
                        '<div class="section-sub">Share of tests flagged by each model</div>', unsafe_allow_html=True)
            for name, group, color in (("Vital Signs", vit, "var(--warn)"), ("Breast", brs, "#8b5cf6")):
                n = len(group)
                f = sum(1 for r in group if r.get("positive"))
                render_prob_bar(name, (f / n) if n else 0.0, color)
                st.markdown(f'<div class="small-note" style="margin:-4px 0 8px 18px;">{f} flagged of {n} tests</div>',
                            unsafe_allow_html=True)
    with right2:
        with st.container(border=True):
            st.markdown('<div class="section-title">\U0001F552 Recent Activity</div>'
                        '<div class="section-sub">Last 5 saved tests</div>', unsafe_allow_html=True)
            if not records:
                st.markdown('<div class="empty-box">No tests yet. Run a screening and it will appear here.</div>',
                            unsafe_allow_html=True)
            else:
                rows = ""
                for r in records[:5]:
                    ico = "\u2764\ufe0f" if r.get("type") == "vital" else "\U0001FA7B"
                    who = r.get("patient_name") or "Unnamed patient"
                    pid = f' \u00b7 ID {esc(r["patient_id"])}' if r.get("patient_id") else ""
                    rows += (
                        f'<div class="recent-row"><div class="recent-ico">{ico}</div>'
                        f'<div class="recent-main"><div class="recent-name">{esc(who)}</div>'
                        f'<div class="recent-sub">{fmt_ts(r.get("timestamp"))}{pid}</div></div>'
                        f'{pill(r.get("label", ""), r.get("positive"))}</div>'
                    )
                st.markdown(rows, unsafe_allow_html=True)


def vital_page():
    page_header("\u2764\ufe0f", "Vital-Signs Risk Screening",
                "Enter the patient's vital signs to predict risk level.")
    left, right = st.columns([1.55, 1], gap="medium")

    with left:
        with st.container(border=True):
            st.markdown('<div class="section-title">\U0001F9D1\u200D\u2695\ufe0f Patient Information</div>'
                        '<div class="section-sub">Optional \u2014 used to identify the record in Test History.</div>',
                        unsafe_allow_html=True)
            p1, p2 = st.columns(2)
            pname = p1.text_input("Patient name", key="v_pname", placeholder="e.g. Ahmed Ali")
            pid = p2.text_input("Patient ID / MRN", key="v_pid", placeholder="e.g. 100234")
            notes = st.text_area("Clinical notes (optional)", key="v_notes", height=80,
                                 placeholder="Symptoms, context, medications...")

        with st.container(border=True):
            st.markdown('<div class="section-title">\U0001FA7A Patient Vitals</div>'
                        '<div class="section-sub">Enter the current measurements.</div>', unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                hr = st.number_input("\u2764\ufe0f Heart Rate (bpm)", min_value=0.0, value=88.0, step=1.0)
                temp = st.number_input("\U0001F321\ufe0f Body Temperature (\u00b0C)", min_value=0.0, value=37.6, step=.1, format="%.1f")
                sbp = st.number_input("\U0001FAC0 Systolic BP (mmHg)", min_value=0.0, value=128.0, step=1.0)
            with c2:
                resp = st.number_input("\U0001FAC1 Respiratory Rate", min_value=0.0, value=18.0, step=1.0)
                spo2 = st.number_input("\U0001F4A7 Oxygen Saturation (%)", min_value=0.0, max_value=100.0, value=96.0, step=1.0)
                dbp = st.number_input("\U0001FAC0 Diastolic BP (mmHg)", min_value=0.0, value=82.0, step=1.0)
            predict = st.button("\u26a1 PREDICT RISK", type="primary", use_container_width=True, key="vital_predict")

        if predict:
            try:
                model = load_vital_model()
                X = pd.DataFrame([{" HR (BPM)": hr, " RESP (BPM)": resp, " SpO2 (%)": spo2, "TEMP (*C)": temp}])
                proba = model.predict_proba(X)[0]
                low, high = _risk_breakdown(proba)
                label, level = ("Low Risk", "low") if low >= high else ("High Risk", "high")
                details = {
                    "level": level, "label": label, "low": low, "high": high,
                    "inputs": {"hr": hr, "temp": temp, "sbp": sbp, "dbp": dbp, "resp": resp, "spo2": spo2},
                }
                rec = add_record({
                    "type": "vital",
                    "patient_name": pname.strip(), "patient_id": pid.strip(),
                    "label": label, "positive": level == "high", "score": high, "notes": notes.strip(),
                    "details": details,
                })
                st.session_state.vital_result = {**details, "saved_id": rec["id"]}
            except Exception as e:
                st.error(f"Vital-sign model error: {e}")

    with right:
        with st.container(border=True):
            st.markdown('<div class="predict-result-title">Prediction Result</div>', unsafe_allow_html=True)
            r = st.session_state.vital_result
            if r:
                render_vital_result(r, saved=True)
                st.markdown('<div class="prob-heading" style="margin-top:14px;">Vitals Review</div>', unsafe_allow_html=True)
                render_vital_review(r["inputs"])
            else:
                st.markdown('<div class="empty-box" style="min-height:170px;">Enter vitals and click PREDICT RISK to see results</div>',
                            unsafe_allow_html=True)
            st.markdown(
                '<div class="info-note">\u2139\ufe0f&nbsp; This prediction is for screening support only '
                'and is not a substitute for professional clinical judgment.</div>',
                unsafe_allow_html=True,
            )


def _render_image_card(title, pil_image, caption_html=None, empty_text="Upload and analyze an image"):
    st.markdown(f'<div class="img-card-title">{title}</div>', unsafe_allow_html=True)
    if pil_image is not None:
        st.image(pil_image, use_container_width=True, output_format="PNG")
    else:
        st.markdown(f'<div class="img-frame"><div class="img-empty">{empty_text}</div></div>', unsafe_allow_html=True)
    if caption_html:
        st.markdown(caption_html, unsafe_allow_html=True)


DENSITY_NAMES = {"A": "Almost Entirely Fatty", "B": "Scattered Fibroglandular",
                 "C": "Heterogeneously Dense", "D": "Extremely Dense"}
VIEW_NAMES = {"AT": "Axillary Tail", "CC": "Craniocaudal", "LM": "Lateromedial",
              "LMO": "Lateromedial Oblique", "ML": "Mediolateral", "MLO": "Mediolateral Oblique"}


def breast_page():
    page_header("\U0001FA7B", "Breast Cancer Detection",
                "Upload a mammogram (ROI image) to get AI analysis with visualization.")
    config = read_config()

    with st.container(border=True):
        st.markdown('<div class="section-title">\U0001F9D1\u200D\u2695\ufe0f Patient Information</div>', unsafe_allow_html=True)
        p1, p2, p3 = st.columns([1, 1, 1.4])
        pname = p1.text_input("Patient name", key="b_pname", placeholder="e.g. Sara Mohamed")
        pid = p2.text_input("Patient ID / MRN", key="b_pid", placeholder="e.g. 100235")
        notes = p3.text_area("Clinical notes (optional)", key="b_notes", height=68)

    with st.container(border=True):
        col_upload, col_orig, col_box, col_cam = st.columns([1.1, 1, 1, 1.15], gap="medium")

        with col_upload:
            st.markdown('<div class="img-card-title">Upload</div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                "Drag and drop an image here", type=["png", "jpg", "jpeg"],
                key="breast_upload", help="PNG, JPG, JPEG (Max 200MB)",
            )
            try:
                config_threshold = float(config.get("cancer_threshold", 0.93))
            except Exception:
                config_threshold = 0.93

            threshold = st.slider(
                "Cancer Detection Threshold", min_value=0.05, max_value=0.95,
                value=float(min(max(config_threshold, 0.05), 0.95)), step=0.01, format="%.2f",
                key="breast_threshold",
                help="Lower values make the app classify more images as Cancer Detected. "
                     "This does not change the model's predicted probability.",
            )
            st.markdown(
                f'<div class="threshold-note">Current threshold: <strong>{threshold:.2f}</strong> ({threshold*100:.0f}%)</div>',
                unsafe_allow_html=True,
            )
            analyze = st.button("\U0001f50d ANALYZE IMAGE", type="primary", use_container_width=True, key="breast_analyze")

        r = st.session_state.breast_result

        if analyze:
            if uploaded is None:
                st.warning("Please upload a mammogram image first.")
            else:
                try:
                    with st.spinner("Analyzing image..."):
                        model, cfg, device = load_breast_assets()
                        image_size = int(cfg.get("image_size", 768))
                        image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
                        tensor = breast_transform(image, image_size).unsqueeze(0).to(device)
                        with torch.inference_mode():
                            outputs = model(tensor)
                            cancer_prob = float(torch.sigmoid(outputs["cancer"].view(-1))[0].item())
                            density_idx = int(outputs["density"].argmax(dim=1)[0].item())
                            view_idx = int(outputs["view"].argmax(dim=1)[0].item())
                            laterality_idx = int(outputs["laterality"].argmax(dim=1)[0].item())

                        del outputs
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                        density_classes = ["A", "B", "C", "D"]
                        view_classes = ["AT", "CC", "LM", "LMO", "ML", "MLO"]
                        laterality_classes = ["L", "R"]

                        box, grayscale_cam = compute_gradcam(
                            model, tensor, image, cam_size=512 if device.type == "cpu" else 768
                        )
                        bbox_image = build_bbox_image(image, box)
                        cam_image = build_gradcam_image(image, grayscale_cam, box)

                        del tensor
                        gc.collect()
                        if device.type == "cuda":
                            torch.cuda.empty_cache()

                    density_code = density_classes[density_idx]
                    view_code = view_classes[view_idx]
                    laterality_code = laterality_classes[laterality_idx]
                    is_cancer = cancer_prob >= threshold
                    label = "Cancer Detected" if is_cancer else "No Cancer Detected"

                    summary = {
                        "label": label, "is_cancer": is_cancer, "prob": cancer_prob, "threshold": threshold,
                        "density_code": density_code, "density_name": DENSITY_NAMES.get(density_code, density_code),
                        "view_code": view_code, "view_name": VIEW_NAMES.get(view_code, view_code),
                        "laterality": "Left Breast" if laterality_code == "L" else "Right Breast",
                        "poi": list(box) if box is not None else None,
                        "filename": getattr(uploaded, "name", ""),
                    }
                    rec = add_record({
                        "type": "breast",
                        "patient_name": pname.strip(), "patient_id": pid.strip(),
                        "label": label, "positive": is_cancer, "score": cancer_prob, "notes": notes.strip(),
                        "details": summary,
                    }, images={"original": image, "bbox": bbox_image, "cam": cam_image})

                    st.session_state.breast_result = {
                        **summary,
                        "original": image, "bbox_image": bbox_image, "cam_image": cam_image,
                        "poi": box, "has_gradcam": HAS_GRADCAM,
                        "grayscale_cam_failed": HAS_GRADCAM and grayscale_cam is None,
                        "saved_id": rec["id"],
                    }
                    r = st.session_state.breast_result
                except Exception as e:
                    st.error(f"Breast-cancer model error: {e}")

        preview = None
        if r:
            preview = r["original"]
        elif uploaded is not None:
            try:
                preview = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
            except Exception:
                preview = None

        with col_orig:
            _render_image_card("Original Mammogram", preview)

        with col_box:
            cap = ('<div class="img-caption"><span class="legend-box" style="background:#f02828;"></span> '
                   'Detected Region (Model-derived)</div>') if r else None
            _render_image_card("Detection (Bounding Box)", r["bbox_image"] if r else None, cap)

        with col_cam:
            sub_img, sub_bar = st.columns([5, 1])
            with sub_img:
                _render_image_card("Grad-CAM Visualization", r["cam_image"] if r else None)
            with sub_bar:
                st.markdown('<div class="img-card-title">&nbsp;</div>', unsafe_allow_html=True)
                st.markdown('<div class="colorbar-wrap"><span>High</span><div class="colorbar"></div><span>Low</span></div>',
                            unsafe_allow_html=True)
            if r and not r.get("has_gradcam", True):
                st.markdown('<div class="small-note">Grad-CAM is unavailable in this environment; '
                            'the cancer prediction remains available without the heatmap.</div>', unsafe_allow_html=True)
            elif r and r.get("grayscale_cam_failed", False):
                st.markdown('<div class="small-note">The prediction completed successfully, but '
                            'Grad-CAM was skipped because of available memory.</div>', unsafe_allow_html=True)

    if r:
        render_breast_cards(r)
        st.markdown('<div class="saved-note">\u2713 Saved to Test History</div>', unsafe_allow_html=True)
        if r["poi"] is not None:
            x, y, w, h = r["poi"]
            st.markdown(
                f'<div class="small-note" style="margin-top:8px;">Model-derived POI: x={x}, y={y}, '
                f'width={w}, height={h}. This is an explainability region, not a ground-truth lesion annotation.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="small-note" style="margin-top:8px;">No suspicious region was localized by the model-derived attention map.</div>',
                        unsafe_allow_html=True)
        st.markdown('<div class="info-note">\u2139\ufe0f&nbsp; This AI system provides screening '
                    'support and should not replace professional radiological evaluation.</div>', unsafe_allow_html=True)


def _history_df(records):
    rows = []
    for r in records:
        rows.append({
            "Date": fmt_ts(r.get("timestamp")),
            "Type": "Vital Signs" if r.get("type") == "vital" else "Breast Cancer",
            "Patient": r.get("patient_name") or "\u2014",
            "Patient ID": r.get("patient_id") or "\u2014",
            "Result": r.get("label", ""),
            "Score (%)": round(float(r.get("score", 0)) * 100, 1),
            "By": r.get("user", ""),
        })
    return pd.DataFrame(rows)


def _render_history_detail(r):
    d = r.get("details", {})
    who = r.get("patient_name") or "Unnamed patient"
    pid = f' \u00b7 ID {esc(r["patient_id"])}' if r.get("patient_id") else ""
    st.markdown(
        f'<div class="section-title">{esc(who)}{pid}</div>'
        f'<div class="section-sub">{fmt_ts(r.get("timestamp"))} \u00b7 by {esc(r.get("user", ""))} \u00b7 '
        f'{"Vital-Signs Screening" if r.get("type") == "vital" else "Breast Cancer Detection"}</div>',
        unsafe_allow_html=True,
    )
    if r.get("notes"):
        st.markdown(f'<div class="info-note" style="margin:0 0 12px;">\U0001F4DD&nbsp; {esc(r["notes"])}</div>',
                    unsafe_allow_html=True)
    if r.get("type") == "vital":
        a, b = st.columns([1, 1.3], gap="medium")
        with a:
            render_vital_result({"level": d.get("level", "low"), "label": d.get("label", r.get("label", "")),
                                 "low": d.get("low", 0.0), "high": d.get("high", 0.0)})
        with b:
            st.markdown('<div class="prob-heading">Vitals Review</div>', unsafe_allow_html=True)
            render_vital_review(d.get("inputs", {}))
    else:
        imgs = r.get("images", {})
        cols = st.columns(3, gap="medium")
        for col, (key, title) in zip(cols, [("original", "Original Mammogram"),
                                            ("bbox", "Detection (Bounding Box)"),
                                            ("cam", "Grad-CAM Visualization")]):
            with col:
                path = HISTORY_IMG_DIR / imgs.get(key, "")
                img = None
                if imgs.get(key) and path.exists():
                    try:
                        img = Image.open(path)
                    except Exception:
                        img = None
                _render_image_card(title, img, empty_text="Image not available")
        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        if d:
            render_breast_cards({
                "is_cancer": d.get("is_cancer", False), "label": d.get("label", r.get("label", "")),
                "prob": d.get("prob", 0.0), "density_name": d.get("density_name", ""),
                "density_code": d.get("density_code", ""), "view_code": d.get("view_code", ""),
                "view_name": d.get("view_name", ""), "laterality": d.get("laterality", ""),
            })
            st.markdown(f'<div class="small-note" style="margin-top:8px;">Threshold used: {d.get("threshold", 0):.2f}'
                        f' \u00b7 File: {esc(d.get("filename", "\u2014"))}</div>', unsafe_allow_html=True)


def build_report_html(r):
    d = r.get("details", {})
    is_v = r.get("type") == "vital"
    title = "Vital-Signs Risk Screening Report" if is_v else "Breast Cancer Detection Report"
    color = "#dc2626" if r.get("positive") else "#16a34a"
    rows = ""
    if is_v:
        for lab, val, unit, rng, status in vital_flags(d.get("inputs", {})):
            c = {"Normal": "#16a34a", "Low": "#d97706", "High": "#dc2626"}[status]
            rows += (f"<tr><td>{esc(lab)}</td><td>{val} {esc(unit)}</td><td>{rng}</td>"
                     f"<td style='color:{c};font-weight:700'>{status}</td></tr>")
        table = f"<table><tr><th>Vital</th><th>Value</th><th>Reference</th><th>Status</th></tr>{rows}</table>"
        extra = f"<p>Risk probability: <b>{float(d.get('high', 0)) * 100:.1f}%</b></p>"
    else:
        for k, v in (("Cancer probability", f"{float(d.get('prob', 0)) * 100:.1f}%"),
                     ("Decision threshold", f"{float(d.get('threshold', 0)):.2f}"),
                     ("Breast density", f"{d.get('density_name', '')} (Category {d.get('density_code', '')})"),
                     ("View", f"{d.get('view_code', '')} - {d.get('view_name', '')}"),
                     ("Laterality", d.get("laterality", "")), ("File", d.get("filename", ""))):
            rows += f"<tr><th>{esc(k)}</th><td>{esc(v)}</td></tr>"
        table = f"<table>{rows}</table>"
        extra = ""
        for key, cap in (("original", "Original"), ("bbox", "Bounding box"), ("cam", "Grad-CAM")):
            fn = (r.get("images") or {}).get(key)
            fp = HISTORY_IMG_DIR / fn if fn else None
            if fp and fp.exists():
                b64 = base64.b64encode(fp.read_bytes()).decode()
                extra += (f"<figure><img src='data:image/jpeg;base64,{b64}'/>"
                          f"<figcaption>{cap}</figcaption></figure>")
        extra = f"<div class='imgs'>{extra}</div>"
    notes = f"<h3>Clinical notes</h3><p>{esc(r.get('notes'))}</p>" if r.get("notes") else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title><style>
body{{font-family:Segoe UI,Arial,sans-serif;max-width:820px;margin:30px auto;color:#16263a;padding:0 18px}}
.head{{background:linear-gradient(110deg,#0b66c3,#0e9f8e);color:#fff;padding:18px 24px;border-radius:12px}}
.head h1{{margin:0;font-size:22px}} .res{{margin:18px 0;padding:14px 18px;border-radius:10px;border:2px solid {color};color:{color};font-size:20px;font-weight:800}}
table{{border-collapse:collapse;width:100%;margin:10px 0}} td,th{{border-bottom:1px solid #dbe4ee;padding:8px 10px;text-align:left;font-size:14px}}
.imgs{{display:flex;gap:10px}} figure{{flex:1;margin:0;text-align:center;font-size:12px}} img{{width:100%;border-radius:8px}}
.foot{{margin-top:26px;font-size:11px;color:#68798c}}</style></head><body>
<div class="head"><h1>\u271a Hospital AI System</h1><div>{title}</div></div>
<p><b>Patient:</b> {esc(r.get('patient_name') or '-')} &nbsp; <b>ID:</b> {esc(r.get('patient_id') or '-')}<br>
<b>Date:</b> {fmt_ts(r.get('timestamp'))} &nbsp; <b>Performed by:</b> {esc(r.get('user'))}</p>
<div class="res">{esc(r.get('label'))}</div>{table}{extra}{notes}
<p class="foot">This report is AI-assisted screening support and is not a substitute for professional clinical or radiological judgment.</p>
</body></html>"""


def history_page():
    page_header("\U0001F5C2\ufe0f", "Test History", "Every screening you run is saved here automatically.")
    records = load_history()

    if not records:
        with st.container(border=True):
            st.markdown('<div class="empty-box" style="min-height:220px;">No saved tests yet. '
                        'Run a Vital-Signs or Breast Cancer test and the result will be stored here.</div>',
                        unsafe_allow_html=True)
        return

    with st.container(border=True):
        f1, f2, f4, f3 = st.columns([1, 1, 1, 1.5], gap="medium")
        type_f = f1.selectbox("Test type", ["All", "Vital Signs", "Breast Cancer"])
        out_f = f2.selectbox("Outcome", ["All", "Flagged (High Risk / Cancer)", "Normal (Low Risk / No Cancer)"])
        sort_f = f4.selectbox("Sort by", ["Newest first", "Oldest first", "Highest score"])
        q = f3.text_input("Search by patient name or ID", placeholder="Type to filter...").strip().lower()

    filtered = []
    for r in records:
        if type_f == "Vital Signs" and r.get("type") != "vital":
            continue
        if type_f == "Breast Cancer" and r.get("type") != "breast":
            continue
        if out_f.startswith("Flagged") and not r.get("positive"):
            continue
        if out_f.startswith("Normal") and r.get("positive"):
            continue
        if q and q not in str(r.get("patient_name", "")).lower() and q not in str(r.get("patient_id", "")).lower():
            continue
        filtered.append(r)

    if sort_f == "Oldest first":
        filtered = sorted(filtered, key=lambda r: r.get("timestamp", ""))
    elif sort_f == "Highest score":
        filtered = sorted(filtered, key=lambda r: float(r.get("score", 0)), reverse=True)

    with st.container(border=True):
        h1, h2 = st.columns([3, 1])
        h1.markdown(f'<div class="section-title">\U0001F4CB Records</div>'
                    f'<div class="section-sub">Showing {len(filtered)} of {len(records)} saved tests</div>',
                    unsafe_allow_html=True)
        if filtered:
            df = _history_df(filtered)
            h2.download_button("\u2B07\ufe0f Export CSV", df.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"test_history_{date.today().isoformat()}.csv",
                               mime="text/csv", use_container_width=True)
            st.dataframe(df, use_container_width=True, hide_index=True, height=min(80 + 35 * len(df), 380))
        else:
            st.markdown('<div class="empty-box">No records match the current filters.</div>', unsafe_allow_html=True)

    if filtered:
        with st.container(border=True):
            st.markdown('<div class="section-title">\U0001F50E Record Details</div>', unsafe_allow_html=True)
            by_id = {r["id"]: r for r in filtered}

            def _fmt(rid):
                r = by_id[rid]
                ico = "\u2764\ufe0f" if r.get("type") == "vital" else "\U0001FA7B"
                return f'{ico} {fmt_ts(r.get("timestamp"))} \u2014 {r.get("patient_name") or "Unnamed"} \u2014 {r.get("label", "")}'

            chosen = st.selectbox("Select a record", list(by_id.keys()), format_func=_fmt)
            _render_history_detail(by_id[chosen])
            st.markdown('<div style="height:6px"></div>', unsafe_allow_html=True)
            b1, b2, _ = st.columns([1, 1, 2])
            b1.download_button("\U0001F4C4 Download report", build_report_html(by_id[chosen]).encode("utf-8"),
                               file_name=f"report_{chosen}.html", mime="text/html",
                               key=f"rep_{chosen}", use_container_width=True)
            if b2.button("\U0001F5D1\ufe0f Delete record", key=f"del_{chosen}", use_container_width=True):
                delete_record(chosen)
                st.rerun()

    groups = {}
    for r in records:
        gk = (r.get("patient_id") or r.get("patient_name") or "").strip()
        if gk:
            groups.setdefault(gk, []).append(r)
    multi = {k: v for k, v in groups.items() if len(v) >= 2}
    if multi:
        with st.container(border=True):
            st.markdown('<div class="section-title">\U0001F4C8 Patient Trend</div>'
                        '<div class="section-sub">Compare repeated tests for the same patient over time.</div>',
                        unsafe_allow_html=True)
            t1, t2 = st.columns([1.4, 1])
            pk = t1.selectbox("Patient (ID or name)", list(multi.keys()))
            tests = multi[pk]
            avail = [t for t in ("vital", "breast") if sum(1 for r in tests if r.get("type") == t) >= 2]
            if avail:
                tt = t2.radio("Test type", avail, horizontal=True,
                              format_func=lambda t: "Vital Signs" if t == "vital" else "Breast Cancer")
                pts = sorted([r for r in tests if r.get("type") == tt], key=lambda r: r["timestamp"])
                dfp = pd.DataFrame({"Risk score (%)": [round(float(r.get("score", 0)) * 100, 1) for r in pts]},
                                   index=pd.to_datetime([r["timestamp"] for r in pts]))
                st.line_chart(dfp)
            else:
                st.caption("This patient needs at least two tests of the same type to show a trend.")

    with st.expander("\u26a0\ufe0f Danger zone"):
        st.caption("This permanently removes all saved tests and images.")
        confirm = st.checkbox("I understand, delete the entire history", key="confirm_clear")
        if st.button("Clear all history", disabled=not confirm, key="clear_all"):
            clear_history()
            st.session_state.confirm_clear = False
            st.rerun()


# ----------------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------------
if not st.session_state.authenticated:
    login_screen()
else:
    render_sidebar()
    page = st.session_state.page
    if page == "vital":
        vital_page()
    elif page == "breast":
        breast_page()
    elif page == "history":
        history_page()
    else:
        dashboard_page()
    st.markdown(
        '<div class="footer-note">Hospital AI System \u00b7 AI-assisted screening \u00b7 '
        'Not a substitute for professional clinical judgment</div>',
        unsafe_allow_html=True,
    )
