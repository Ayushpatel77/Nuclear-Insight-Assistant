"""
Nuclear Insight Assistant — Streamlit Frontend
User just asks a question. That's it.
"""

import sys
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from rag.document_processor import index_exists
from pipeline import run_pipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Nuclear Insight Assistant",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Exo+2:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Exo 2', sans-serif; }
.stApp { background: #0a0e1a; color: #c8d8f0; }

/* Hide sidebar toggle and default streamlit elements */
[data-testid="collapsedControl"] { display: none; }
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

.main-header {
    text-align: center;
    padding: 3rem 0 2rem;
    border-bottom: 1px solid #1a3060;
    margin-bottom: 2.5rem;
}
.main-header h1 {
    font-family: 'Share Tech Mono', monospace;
    font-size: 2.6rem;
    color: #00d4ff;
    text-shadow: 0 0 30px #00d4ff88;
    letter-spacing: 4px;
    margin: 0;
}
.main-header p {
    color: #5a7aaa;
    font-size: 0.95rem;
    margin-top: 0.5rem;
    letter-spacing: 1px;
}

.atom-deco {
    font-size: 3.5rem;
    animation: pulse 3s ease-in-out infinite;
    display: inline-block;
}
@keyframes pulse {
    0%, 100% { opacity: 0.7; transform: scale(1); }
    50%       { opacity: 1;   transform: scale(1.08); }
}

/* Question input */
.stTextArea textarea {
    background: #0d1525 !important;
    color: #c8d8f0 !important;
    border: 1px solid #1a3060 !important;
    border-radius: 10px !important;
    font-family: 'Exo 2', sans-serif !important;
    font-size: 1rem !important;
    padding: 1rem !important;
}
.stTextArea textarea:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 20px #00d4ff22 !important;
}

/* Analyze button */
@keyframes ripple {
    0%   { box-shadow: 0 0 0 0 #00aaff88; }
    70%  { box-shadow: 0 0 0 18px #00aaff00; }
    100% { box-shadow: 0 0 0 0 #00aaff00; }
}
@keyframes clickFlash {
    0%   { background: linear-gradient(135deg, #003366, #005599); }
    30%  { background: linear-gradient(135deg, #00aaff, #0055ff); filter: brightness(1.4); }
    100% { background: linear-gradient(135deg, #003366, #005599); }
}
.stButton > button {
    background: linear-gradient(135deg, #003366, #005599) !important;
    color: #00d4ff !important;
    border: 1px solid #00aaff !important;
    border-radius: 8px !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 3px !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
    position: relative !important;
    overflow: hidden !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #004488, #0077cc) !important;
    box-shadow: 0 0 25px #00aaff55 !important;
    transform: translateY(-2px) !important;
}
.stButton > button:active {
    animation: ripple 0.5s ease-out, clickFlash 0.4s ease-out !important;
    transform: translateY(1px) scale(0.98) !important;
    filter: brightness(1.3) !important;
    box-shadow: 0 0 40px #00d4ffaa !important;
}

/* Cards */
.card {
    background: #0d1525;
    border: 1px solid #1a3060;
    border-radius: 12px;
    padding: 1.5rem 1.8rem;
    margin: 0.8rem 0;
    line-height: 1.7;
}
.card-cyan   { border-left: 4px solid #00d4ff; }
.card-blue   { border-left: 4px solid #00aaff; }
.card-yellow { border-left: 4px solid #ffcc00; }

/* Agent badge */
.agent-badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    padding: 3px 12px;
    border-radius: 20px;
    margin-bottom: 0.7rem;
    letter-spacing: 1px;
    text-transform: uppercase;
}
.badge-final      { background: #0a2a2a; color: #00d4ff; border: 1px solid #00d4ff; }
.badge-explainer  { background: #0a1a3a; color: #00aaff; border: 1px solid #00aaff; }
.badge-simplifier { background: #2a1a0a; color: #ffaa00; border: 1px solid #ffaa00; }
.badge-retriever  { background: #0d2a1a; color: #00ff88; border: 1px solid #00ff88; }

/* Source tag */
.source-tag {
    display: inline-block;
    background: #0a1e3a;
    color: #5599ff;
    border: 1px solid #1a4080;
    border-radius: 4px;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem;
    padding: 2px 10px;
    margin: 3px 4px 3px 0;
}

/* Metrics */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-box {
    background: #0d1525;
    border: 1px solid #1a3060;
    border-radius: 8px;
    padding: 0.6rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.8rem;
    color: #5a8aaa;
}
.metric-box span { color: #00d4ff; font-size: 1rem; }

/* Tabs */
.stTabs [data-baseweb="tab"] {
    font-family: 'Share Tech Mono', monospace !important;
    color: #5a7aaa !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
}
.stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
}

/* Status box */
.status-box {
    background: #060e1a;
    border: 1px solid #1a3060;
    border-radius: 8px;
    padding: 1rem 1.2rem;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #3a8a6a;
}
.log-line { padding: 2px 0; }

/* Example buttons */
.example-btn {
    display: inline-block;
    background: #0d1830;
    border: 1px solid #1a3060;
    border-radius: 20px;
    color: #5a8aaa;
    font-size: 0.8rem;
    padding: 4px 14px;
    margin: 3px 4px;
    cursor: pointer;
    transition: all 0.2s;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #060a14; }
::-webkit-scrollbar-thumb { background: #1a3060; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <span class="atom-deco">⚛️</span>
    <h1>NUCLEAR INSIGHT ASSISTANT</h1>
    <p>Ask any nuclear science question · Powered by verified documents</p>
</div>
""", unsafe_allow_html=True)

# ── Guard: index must exist ────────────────────────────────────────────────────
if not index_exists():
    st.error(
        "⚠️ Document index not found. "
        "Run `python index_documents.py` first to index the documents, then restart the app."
    )
    st.stop()


# ── Main layout ────────────────────────────────────────────────────────────────

query = st.text_area(
    label="",
    placeholder="e.g. How does a pressurized water reactor control its chain reaction?",
    height=110,
    label_visibility="collapsed",
)
run_btn = st.button("⚡  ANALYZE", use_container_width=True, type="primary")

# ── Pipeline ───────────────────────────────────────────────────────────────────
if run_btn:
    if not query.strip():
        st.warning("Please type a question first.")
        st.stop()

    status_area = st.empty()
    log_lines   = []

    def update_status(msg: str):
        log_lines.append(msg)
        status_area.markdown(
            '<div class="status-box">' +
            "".join(f'<div class="log-line">{l}</div>' for l in log_lines) +
            '</div>',
            unsafe_allow_html=True,
        )

    t0 = time.time()
    with st.spinner(""):
        result = run_pipeline(query, status_cb=update_status)
    elapsed = time.time() - t0

    if result is None:
        st.error("❌ Pipeline returned no result. Check your API key in agents/llm_caller.py and try again.")
        st.stop()

    status_area.empty()

    if result["error"]:
        st.error(f"❌ {result['error']}")
        st.stop()



    # ── Answer tabs ────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs([
        "📚 SIMPLIFIED", "🧠 TECHNICAL", "✅ FINAL ANSWER"
    ])

    with tab1:
        st.markdown('<span class="agent-badge badge-simplifier">Simplifier Agent</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="card card-yellow">{result["simple_answer"]}</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<span class="agent-badge badge-explainer">Explainer Agent</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="card card-blue">{result["technical_answer"]}</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown('<span class="agent-badge badge-final">Critic Agent · Verified</span>', unsafe_allow_html=True)
        st.markdown(f'<div class="card card-cyan">{result["final_answer"]}</div>', unsafe_allow_html=True)

    with st.expander("🖥️ Pipeline Log"):
        for line in result["pipeline_log"]:
            st.markdown(f'<div class="log-line" style="font-family:monospace;font-size:0.78rem;color:#3a7a5a;">{line}</div>', unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#1a3050; font-family:'Share Tech Mono',monospace; font-size:0.72rem; padding:0.8rem 0;">
NUCLEAR INSIGHT · Multi-Agent RAG · Document-Grounded · No Hallucination by Design
</div>
""", unsafe_allow_html=True)
