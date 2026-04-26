"""
Mood Machine — Streamlit frontend.

On startup: initializes GeminiMoodAnalyzer and triggers fine-tuning in the
background if no tuned model exists. The base model handles requests immediately
while fine-tuning runs; the analyzer hot-swaps when it completes.

Run:
    streamlit run streamlit_app.py
"""

import os
import threading
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Mood Machine",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
[data-testid="stAppViewContainer"] { background: #f0f4f8; }
[data-testid="stHeader"] { background: transparent; }
.block-container { max-width: 640px; padding-top: 2.5rem; padding-bottom: 3rem; }

.stTextArea textarea {
    border-radius: 12px !important;
    border: 1.5px solid #e2e8f0 !important;
    background: #fafdff !important;
    font-size: 0.94rem !important;
    line-height: 1.6 !important;
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
    background: rgba(0,0,0,0.06);
    color: #4a5568;
    margin-right: 4px;
}
.tag-ft { background: #ebf4ff !important; color: #2b6cb0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Shared app state (module-level — survives Streamlit reruns) ───────────────

# This dict is returned by the cached _init() so it's the same object
# across all reruns and all sessions on this server.
# The background fine-tuning thread mutates it directly.


@st.cache_resource(show_spinner=False)
def _init() -> dict:
    """Run once per server start. Returns a mutable state dict."""
    state: dict = {
        "analyzer": None,
        "finetune": "checking",   # checking | not_needed | running | done | error
        "finetune_msg": "",
        "model": "initializing",
        "init_error": None,
    }
    try:
        from gemini_analyzer import GeminiMoodAnalyzer
        a = GeminiMoodAnalyzer()
        state["analyzer"] = a
        state["model"] = a.model_description

        if a._tuned_name:
            state["finetune"] = "not_needed"
        else:
            key = os.environ.get("GEMINI_API_KEY", "")
            if key and key != "your_api_key_here":
                state["finetune"] = "running"
                state["finetune_msg"] = "Fine-tuning job submitted (5–30 min)…"
                threading.Thread(target=_finetune_bg, args=(state,), daemon=True).start()
            else:
                state["finetune"] = "error"
                state["finetune_msg"] = "API key not configured"
    except Exception as exc:
        state["init_error"] = str(exc)

    return state


def _finetune_bg(state: dict) -> None:
    """Background thread: run fine-tuning, then hot-swap the analyzer."""
    try:
        from fine_tune import run_fine_tuning
        run_fine_tuning()
        state["finetune"] = "done"
        state["finetune_msg"] = ""

        from gemini_analyzer import GeminiMoodAnalyzer
        new_a = GeminiMoodAnalyzer()
        state["analyzer"] = new_a
        state["model"] = new_a.model_description
    except (SystemExit, Exception) as exc:
        state["finetune"] = "error"
        state["finetune_msg"] = str(exc)


# ── UI helpers ────────────────────────────────────────────────────────────────

MOOD_STYLE = {
    "positive": ("#276749", "#f0fff6", "#38a169"),
    "negative": ("#c53030", "#fff5f5", "#e53e3e"),
    "neutral":  ("#4a5568", "#f7fafc", "#718096"),
    "mixed":    ("#553c9a", "#faf5ff", "#805ad5"),
}


def _render_result(r) -> None:
    tc, bg, bc = MOOD_STYLE.get(r.label, MOOD_STYLE["neutral"])
    pct = round(r.confidence * 100)

    tags_html = ""
    for name, on in [
        ("sarcasm",  r.sarcasm_detected),
        ("negation", r.negation_detected),
        ("slang",    r.slang_detected),
        ("emojis",   r.emojis_detected),
    ]:
        if on:
            tags_html += f'<span class="tag">{name}</span>'
    if r.used_finetuned:
        tags_html += '<span class="tag tag-ft">fine-tuned</span>'
    tags_block = f'<div style="margin-top:13px;">{tags_html}</div>' if tags_html else ""

    st.markdown(f"""
<div style="background:{bg};border-left:4px solid {bc};border-radius:12px;
            padding:20px 24px;margin:4px 0 12px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;
              margin-bottom:12px;">
    <span style="font-size:1.35rem;font-weight:800;letter-spacing:.08em;
                 text-transform:uppercase;color:{tc};">{r.label}</span>
    <span style="font-size:.92rem;font-weight:700;color:#718096;">{pct}%</span>
  </div>
  <div style="height:5px;background:rgba(0,0,0,.07);border-radius:999px;
              margin-bottom:14px;overflow:hidden;">
    <div style="width:{pct}%;height:100%;background:{bc};border-radius:999px;"></div>
  </div>
  <p style="font-size:.86rem;color:#4a5568;line-height:1.65;margin:0;">
    {r.reasoning}
  </p>
  {tags_block}
</div>
""", unsafe_allow_html=True)


def _render_status(state: dict) -> None:
    ft    = state.get("finetune", "checking")
    model = state.get("model", "")
    msg   = state.get("finetune_msg", "")

    if ft == "running":
        icon, label = "🔵", f"{model} · fine-tuning in background…"
    elif ft == "done":
        icon, label = "🟢", f"{model} · fine-tuned ✓"
    elif ft == "error":
        icon, label = "🔴", f"{model} · {msg}".strip(" ·") if model and model != "initializing" else msg
    elif ft == "not_needed":
        icon, label = "🟢", model
    else:
        icon, label = "⚪", model

    st.markdown(
        f'<p style="color:#a0aec0;font-size:.75rem;margin:0;">{icon} {label}</p>',
        unsafe_allow_html=True,
    )


# ── Main ──────────────────────────────────────────────────────────────────────

state = _init()

st.markdown("## Mood Machine")
st.markdown(
    '<p style="color:#718096;font-size:.88rem;margin-top:-10px;margin-bottom:20px;">'
    "AI-powered emotional analysis</p>",
    unsafe_allow_html=True,
)

# API key gate
api_key = os.environ.get("GEMINI_API_KEY", "")
if not api_key or api_key == "your_api_key_here":
    st.error("**GEMINI_API_KEY not set.**  Add your key to the `.env` file and restart.")
    st.code("GEMINI_API_KEY=your_actual_key_here", language=None)
    st.stop()

# Init error
if state.get("init_error"):
    st.error(f"Initialization error: {state['init_error']}")
    st.stop()

# Wait for analyzer to be ready (should only take a moment)
if state["analyzer"] is None:
    with st.spinner("Initializing…"):
        time.sleep(0.6)
    st.rerun()
    st.stop()

# ── Input form ────────────────────────────────────────────────────────────────

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
        f'<p style="text-align:right;color:#a0aec0;font-size:.75rem;'
        f'padding-top:.5rem;">{len(text)}/500</p>',
        unsafe_allow_html=True,
    )

# ── Analysis ──────────────────────────────────────────────────────────────────

if clicked:
    if not text.strip():
        st.warning("Please enter some text first.")
    else:
        with st.spinner("Analyzing…"):
            result = state["analyzer"].analyze(text.strip())
        st.session_state["result"] = result

if "result" in st.session_state:
    _render_result(st.session_state["result"])

# ── Status bar ────────────────────────────────────────────────────────────────

st.markdown("---")
_render_status(state)
