"""Look and feel: a case-file design on carbon-blue, shared by all the demo apps."""

from __future__ import annotations

import datetime as _dt

import gradio as gr

NAVY = "#1C2B4A"
PAPER = "#F1EFE8"
INK = "#1D1D1B"
MUTED = "#6B6A64"
RED = "#B3261E"
STAMP = "#FF5A4F"

HEAD = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@600;700&family=Courier+Prime:wght@400;700&display=swap" rel="stylesheet">
"""

_THEME_VARS = dict(
    body_background_fill=NAVY,
    body_text_color=INK,
    background_fill_primary=PAPER,
    background_fill_secondary=PAPER,
    block_background_fill=PAPER,
    block_border_color=INK,
    block_border_width="1px",
    block_label_background_fill=PAPER,
    block_label_text_color=INK,
    block_label_border_color=INK,
    block_title_text_color=MUTED,
    block_shadow="none",
    border_color_primary=INK,
    input_background_fill=PAPER,
    input_border_color=INK,
    input_placeholder_color=MUTED,
    color_accent=RED,
    color_accent_soft="#E8D9D2",
    slider_color=RED,
    checkbox_background_color=PAPER,
    checkbox_background_color_selected=INK,
    checkbox_border_color=INK,
    checkbox_border_color_selected=INK,
    button_primary_background_fill=INK,
    button_primary_background_fill_hover="#000000",
    button_primary_text_color=PAPER,
    button_secondary_background_fill=PAPER,
    button_secondary_text_color=INK,
    button_secondary_border_color=INK,
    panel_background_fill=PAPER,
    panel_border_color=INK,
)

_base = gr.themes.Base(
    font=[gr.themes.GoogleFont("Courier Prime"), gr.themes.Font("Courier New"), gr.themes.Font("monospace")],
    font_mono=[gr.themes.GoogleFont("Courier Prime"), gr.themes.Font("Courier New"), gr.themes.Font("monospace")],
    radius_size=gr.themes.sizes.radius_none,
)
# The design has one look. Use the same values in dark mode, so visitors whose
# browser or Hugging Face account is set to dark see the same page.
_dark = {f"{k}_dark": v for k, v in _THEME_VARS.items() if hasattr(_base, f"{k}_dark")}
THEME = _base.set(**_THEME_VARS, **_dark)

CSS = f"""
:root, .dark {{ color-scheme: light; }}
.gradio-container {{ max-width: 1240px !important; padding: 28px 40px 40px !important; }}
footer:not(.foot) {{ display: none !important; }}

/* ---- header ---- */
.case-head *, .case-lede, .stats *, .foot, .foot * {{ color:{PAPER} !important; }}
.case-head .stamp {{ color:{STAMP} !important; }}
.case-head .case-title .bar {{ color:transparent !important; }}
.more .label-wrap span, .more button span {{ color:{INK} !important; }}
.case-head {{ display:flex; justify-content:space-between; align-items:flex-end; gap:24px;
  border-bottom:2px solid {PAPER}; padding-bottom:16px; margin-bottom:6px; color:{PAPER}; }}
.case-eyebrow {{ font-size:12px; letter-spacing:3px; text-transform:uppercase; opacity:.85; }}
.case-title {{ font-family:'Lora', Georgia, serif; font-size:46px; font-weight:700; line-height:1.05;
  letter-spacing:-.5px; margin:6px 0 0; color:{PAPER}; }}
.case-title .bar {{ background:{PAPER}; color:transparent; padding:0 4px; user-select:none; }}
.case-meta {{ text-align:right; font-size:12px; line-height:1.8; letter-spacing:.5px; }}
.stamp {{ display:inline-block; border:2px solid {STAMP}; color:{STAMP}; padding:2px 10px; font-weight:700;
  letter-spacing:3px; transform:rotate(-4deg); margin-top:6px; transition:opacity .3s; }}
.stamp.idle {{ opacity:.35; }}
.case-lede {{ color:{PAPER}; opacity:.85; font-size:14px; max-width:780px; margin:10px 0 18px; line-height:1.6; }}

/* ---- tabs ---- */
.tabs > .tab-wrapper button, .tabs button[role="tab"] {{ color:{PAPER} !important; opacity:.6;
  letter-spacing:2px; text-transform:uppercase; font-size:13px; border:none !important; background:transparent !important; }}
.tabs button[role="tab"][aria-selected="true"] {{ opacity:1; border-bottom:2px solid {STAMP} !important; }}
.tabs .tabitem {{ background:transparent !important; border:none !important; padding:16px 0 0 !important; }}

/* ---- control panel (paper) ---- */
.paper {{ background:{PAPER}; border:1px solid {INK}; padding:18px !important; gap:14px !important; }}
.paper .block, .paper .form {{ border:none !important; background:transparent !important; box-shadow:none !important; }}
.paper label > span, .paper .block-title, .paper [data-testid="block-info"] {{ text-transform:uppercase;
  letter-spacing:2px; font-size:11px !important; color:{MUTED} !important; }}
.drop {{ border:1px dashed {INK} !important; }}

/* segmented radios */
.seg .wrap {{ display:flex !important; gap:0 !important; border:1px solid {INK}; }}
.seg .wrap label {{ flex:1; justify-content:center; margin:0 !important; border:none !important;
  border-right:1px solid {INK} !important; background:{PAPER} !important; padding:9px 4px !important;
  box-shadow:none !important; border-radius:0 !important; }}
.seg .wrap label:last-child {{ border-right:none !important; }}
.seg .wrap label input {{ display:none; }}
.seg .wrap label.selected, .seg .wrap label:has(input:checked) {{ background:{INK} !important; color:{PAPER} !important; }}
.seg .wrap label span {{ text-transform:none !important; letter-spacing:0 !important; font-size:13px !important;
  color:inherit !important; margin:0 !important; }}

.go button, button.go {{ letter-spacing:3px; text-transform:uppercase; font-weight:700; padding:14px !important; }}
.note {{ font-size:11px; color:{MUTED}; margin:0; }}
.more {{ border:1px solid {INK} !important; }}

/* ---- exhibit (results) ---- */
.exhibit {{ gap:12px !important; }}
.figure {{ border:1px solid {INK} !important; padding:10px !important; background:{PAPER} !important; }}
.figure img {{ object-fit:contain; }}
.stats {{ display:flex; gap:40px; color:{PAPER}; padding:4px 2px; }}
.stats b {{ font-family:'Lora', Georgia, serif; font-size:30px; display:block; line-height:1.1; }}
.stats span {{ font-size:12px; letter-spacing:.5px; opacity:.8; }}
.paperfile {{ border:1px solid {INK} !important; }}
.paperfile table, .paperfile tr, .paperfile td, .paperfile tbody, .paperfile .file-preview-holder {{ background:{PAPER} !important; color:{INK} !important; }}
.more.dark {{ background:{PAPER}; }}

/* ---- CCTV monitor for video ---- */
.cam {{ display:flex; justify-content:space-between; background:#05070A; color:#5EE38A;
  font-size:12px; letter-spacing:1.5px; padding:8px 12px; border:1px solid #000; border-bottom:none; }}
.cam, .cam span {{ color:#5EE38A !important; }}
.cam .rec {{ color:{STAMP} !important; }}
.monitor {{ background:#05070A !important; border:1px solid #000 !important; padding:8px !important; }}
.monitor, .monitor > div, .monitor [data-testid] {{ background:#05070A !important; }}
.monitor label, .monitor .block-label {{ background:{PAPER} !important; }}
.monitor video {{ filter:saturate(.85) contrast(1.05); }}

.foot {{ color:{PAPER}; opacity:.75; font-size:12px; margin-top:22px; text-align:center; }}
.foot a {{ color:{PAPER}; text-decoration:underline; }}

@media (max-width: 760px) {{
  .gradio-container {{ padding:16px !important; }}
  .case-head {{ flex-direction:column; align-items:flex-start; }}
  .case-meta {{ text-align:left; }}
  .case-title {{ font-size:34px; }}
  .stats {{ gap:20px; flex-wrap:wrap; }}
}}
"""


def header_html(file_info: str = "awaiting upload", stamped: bool = False) -> str:
    case_no = "FR-" + _dt.date.today().strftime("%Y-%m%d")
    stamp_cls = "stamp" if stamped else "stamp idle"
    return f"""
<div class="case-head">
  <div>
    <div class="case-eyebrow">Exhibit 01 · Privacy tooling</div>
    <h1 class="case-title" aria-label="Face Redaction">Face <span class="bar">Redac</span>tion</h1>
  </div>
  <div class="case-meta">CASE NO. {case_no}<br>FILE: {file_info}<br><span class="{stamp_cls}">REDACTED</span></div>
</div>
<p class="case-lede">Find every face in a photo or video and hide it, keeping the rest of the image intact.
Faces are found with YuNet and scanned again on zoomed tiles to catch small faces in crowds and CCTV.
In video, a tracker keeps covering a face for a few frames when the detector misses it.</p>
"""


def stats_html(found="—", redacted="—", spared="—", time_taken="—",
               labels=("faces located", "redacted", "exempt", "scan time")) -> str:
    values = (found, redacted, spared, time_taken)
    cells = "".join(f"<div><b>{v}</b><span>{l}</span></div>" for v, l in zip(values, labels))
    return f'<div class="stats">{cells}</div>'


def video_header_html(frames: int | None = None, tracks: int | None = None) -> str:
    now = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    left = "CAM 01 · REDACTED OUTPUT" if frames else "CAM 01 · NO SIGNAL"
    mid = f"FRAMES {frames} · TRACKS {tracks}" if frames else now
    return f'<div class="cam"><span>{left}</span><span>{mid}</span><span class="rec">● REC</span></div>'
