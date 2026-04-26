"""
Mood Machine — Streamlit frontend.

Run fine-tuning BEFORE starting the app (one-time setup):
    python fine_tune.py

Run tests separately:
    python reliability_tests.py

Then launch the app:
    streamlit run streamlit_app.py

Terminal logs:
  [frontend]          → request received / ← response sent
  [agentic_pipeline]  each pipeline step with rule-based, cross-validation, synthesis
  [gemini_analyzer]   Gemini API call, timing, label, confidence, reasoning
"""

import logging
import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
# force=True so our format wins even if Streamlit already set up the root logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  [%(name)-20s]  %(message)s",
    datefmt="%H:%M:%S",
    force=True,
)
logger = logging.getLogger("frontend")

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mood Machine",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.html("""
<link rel="stylesheet"
      href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"
      crossorigin="anonymous">
<style>
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 640px; padding-top: 2.5rem; padding-bottom: 3rem; }

body, p, span, div, label, li, td, th { color: #111 !important; }
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] span { color: #111 !important; }

.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #fafdff !important;
    font-size: 0.94rem !important;
    line-height: 1.6 !important;
    color: #111 !important;
}
.stTextArea textarea:focus {
    border-color: #7c8cf8 !important;
    background: #fff !important;
}
div.stButton > button {
    border-radius: 12px !important;
    font-weight: 600 !important;
    border: none !important;
    font-size: 0.94rem !important;
}
div.stButton > button[kind="primary"] {
    background: #1a202c !important;
    color: #fff !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #2d3748 !important;
    border: none !important;
}
.tag {
    display: inline-block;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 3px 9px;
    border-radius: 999px;
    background: rgba(0,0,0,0.08);
    color: #111 !important;
    margin-right: 4px;
}
.tag-ft { background: #ebf4ff !important; color: #1a365d !important; }
.row-label {
    font-size: .72rem;
    font-weight: 700;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #111 !important;
    display: flex;
    align-items: center;
    gap: 5px;
    margin-bottom: 5px;
}
.row-label i { color: #555 !important; }
</style>
""")

# ── Shared app state (module-level — survives Streamlit reruns) ───────────────

@st.cache_resource(show_spinner=False)
def _init() -> dict:
    """Run once per server start. Returns a mutable state dict shared across reruns."""
    state: dict = {
        "pipeline": None,
        "model": "initializing",
        "init_error": None,
    }
    try:
        logger.info("=== Mood Machine starting up ===")
        from agentic_pipeline import AgenticMoodPipeline
        pipeline = AgenticMoodPipeline()
        state["pipeline"] = pipeline
        state["model"] = pipeline._gemini.model_description
        logger.info("Backend ready — %s", pipeline._gemini.model_description)
    except Exception as exc:
        logger.error("Initialization failed: %s", exc)
        state["init_error"] = str(exc)
    return state


# ── UI helpers ────────────────────────────────────────────────────────────────

MOOD_STYLE = {
    "positive": ("#111", "#f0fff6", "#38a169"),
    "negative": ("#111", "#fff5f5", "#e53e3e"),
    "neutral":  ("#111", "#f7fafc", "#718096"),
    "mixed":    ("#111", "#faf5ff", "#805ad5"),
}

MOOD_ICON = {
    "positive": "fa-face-smile",
    "negative": "fa-cloud-rain",
    "neutral":  "fa-circle-minus",
    "mixed":    "fa-shuffle",
}


def _render_result(r) -> None:
    tc, bg, bc = MOOD_STYLE.get(r.final_label, MOOD_STYLE["neutral"])
    icon_cls = MOOD_ICON.get(r.final_label, "fa-circle-minus")
    pct = round(r.final_confidence * 100)

    tags_html = "".join(
        f'<span class="tag">{name}</span>'
        for name, on in [("sarcasm", r.sarcasm_detected),
                         ("negation", r.negation_detected),
                         ("slang", r.slang_detected)]
        if on
    )
    if r.used_finetuned:
        tags_html += '<span class="tag tag-ft">fine-tuned</span>'
    tags_block = f'<div style="margin-top:13px;">{tags_html}</div>' if tags_html else ""

    st.markdown(f"""
<div style="background:{bg};border-left:4px solid {bc};border-radius:12px;
            padding:20px 24px;margin:4px 0 4px;">
  <div style="display:flex;justify-content:space-between;align-items:center;
              margin-bottom:12px;">
    <span style="font-size:1.35rem;font-weight:800;letter-spacing:.08em;
                 text-transform:uppercase;color:{tc};display:flex;align-items:center;gap:10px;">
      <i class="fa-solid {icon_cls}" style="font-size:1.1rem;color:{bc};"></i>
      {r.final_label}
    </span>
    <span style="font-size:.92rem;font-weight:700;color:#111;">{pct}%</span>
  </div>
  <div style="height:5px;background:rgba(0,0,0,.07);border-radius:999px;
              margin-bottom:14px;overflow:hidden;">
    <div style="width:{pct}%;height:100%;background:{bc};border-radius:999px;"></div>
  </div>
  <p style="font-size:.86rem;color:#111;line-height:1.65;margin:0;">
    {r.gemini_reasoning}
  </p>
  {tags_block}
</div>
""", unsafe_allow_html=True)


def _render_comparison(r) -> None:
    """Model comparison row: rule-based vs Gemini vs ML tiebreaker."""
    ml_cell = (
        f'<span><b>ML tiebreaker</b>: {r.ml_label}</span>'
        if r.tiebreaker_used else ""
    )
    verdict_icon = "fa-circle-check" if r.models_agreed else "fa-code-branch"
    verdict_text = "agreed" if r.models_agreed else "tiebreaker used"
    st.markdown(
        '<p class="row-label">'
        '<i class="fa-solid fa-robot"></i> Model comparison</p>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
<div style="font-size:.78rem;color:#111;padding:8px 14px;background:#f7fafc;
            border-radius:8px;display:flex;gap:16px;flex-wrap:wrap;margin-bottom:6px;">
  <span><b>Rule-based</b>: {r.rule_based_label}</span>
  <span><b>Gemini</b>: {r.gemini_label} ({round(r.gemini_confidence * 100)}%)</span>
  {ml_cell}
  <span style="margin-left:auto;color:#111;">
    <i class="fa-solid {verdict_icon}" style="margin-right:4px;"></i>
    {verdict_text} &middot; {r.latency_ms:.0f} ms
  </span>
</div>
""", unsafe_allow_html=True)



def _render_status(state: dict) -> None:
    model = state.get("model", "")
    pipeline = state.get("pipeline")
    has_finetuned = pipeline and getattr(pipeline._gemini, "_tuned_name", None)
    icon_cls = "fa-microchip" if has_finetuned else "fa-server"
    st.markdown(
        f'<p style="color:#111;font-size:.75rem;margin:0;">'
        f'<i class="fa-solid {icon_cls}" style="margin-right:5px;color:#555;"></i>'
        f'{model}</p>',
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

state = _init()

st.markdown(
    '<h2 style="color:#111;display:flex;align-items:center;gap:12px;margin-bottom:2px;">'
    '<i class="fa-solid fa-brain" style="color:#555;font-size:1.4rem;"></i>'
    'Mood Machine</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p style="color:#111;font-size:.88rem;margin-top:-6px;margin-bottom:20px;">'
    '<i class="fa-solid fa-mobile-screen" style="margin-right:5px;color:#555;"></i>'
    "AI-powered emotional analysis for social media posts</p>",
    unsafe_allow_html=True,
)

# API key gate
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key or api_key == "your_api_key_here":
    st.error("**GEMINI_API_KEY not set.**  Add your key to `.env` and restart.")
    st.code("GEMINI_API_KEY=your_actual_key_here", language=None)
    st.stop()

# Init error
if state.get("init_error"):
    st.error(f"Initialization error: {state['init_error']}")
    st.stop()

# Wait for pipeline (should be ready almost instantly)
if state["pipeline"] is None:
    with st.spinner("Initializing…"):
        time.sleep(0.6)
    st.rerun()
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────

st.markdown(
    '<p class="row-label">'
    '<i class="fa-solid fa-comment-dots"></i> Your message</p>',
    unsafe_allow_html=True,
)
text = st.text_area(
    "input",
    placeholder="Type a phrase, sentence, or social media post…",
    max_chars=500,
    height=130,
    label_visibility="collapsed",
)

col_btn, col_count = st.columns([6, 1])
with col_btn:
    clicked = st.button("Analyze Mood", type="primary", use_container_width=True)
with col_count:
    st.markdown(
        f'<p style="text-align:right;color:#111;font-size:.75rem;'
        f'padding-top:.5rem;">{len(text)}/500</p>',
        unsafe_allow_html=True,
    )

# ── Analysis ──────────────────────────────────────────────────────────────────

if clicked:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        logger.info("─── Frontend request ─────────────────────────────────────")
        logger.info("→ Received: %r (%d chars)", text.strip()[:60], len(text.strip()))
        with st.spinner("Analyzing…"):
            result = state["pipeline"].analyze(text.strip())
        logger.info(
            "← Returning: %s (%.0f%%) in %.0fms",
            result.final_label.upper(), result.final_confidence * 100, result.latency_ms,
        )
        logger.info("─────────────────────────────────────────────────────────")
        st.session_state["result"] = result

# ── Result display ────────────────────────────────────────────────────────────

if "result" in st.session_state:
    r = st.session_state["result"]
    _render_result(r)
    _render_comparison(r)

# ── Status bar ────────────────────────────────────────────────────────────────

st.markdown("---")
_render_status(state)
