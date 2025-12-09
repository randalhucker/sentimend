import streamlit as st

from services.utils import *
from views.modal import review_modal

# ---------- Page Config ----------
st.set_page_config(
    page_title="Sentimend | Play Store Sentiment Explorer",
    page_icon="📱",
    layout="wide",
)

LANGUAGE_MAP = {
    "en": "🇺🇸 English",
    "es": "🇪🇸 Spanish",
    "de": "🇩🇪 German",
    "fr": "🇫🇷 French",
    "it": "🇮🇹 Italian",
}

COUNTRY_MAP = {
    "us": "🇺🇸 United States",
    "gb": "🇬🇧 United Kingdom",
    "de": "🇩🇪 Germany",
    "fr": "🇫🇷 France",
    "it": "🇮🇹 Italy",
    "in": "🇮🇳 India",
}

# ---------- Sidebar (State Management) ----------
st.sidebar.header("🔧 Settings")
# Store sidebar inputs in session_state keys prefixed with 's_' so modal can find them
lang = st.sidebar.selectbox(
    "Language",
    options=list(LANGUAGE_MAP.keys()),
    format_func=lambda x: LANGUAGE_MAP[x],
    key="s_lang",
)

country = st.sidebar.selectbox(
    "Country",
    options=list(COUNTRY_MAP.keys()),
    format_func=lambda x: COUNTRY_MAP[x],
    key="s_country",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Review Fetch")
count = st.sidebar.slider("Number of reviews", 100, 2000, 400, 100, key="s_count")
sort_newest = st.sidebar.toggle("Sort by newest", False, key="s_sort")

# Handle Star Logic
star_val = st.sidebar.select_slider(
    "Star rating filter",
    options=[0, 1, 2, 3, 4, 5],
    value=0,
    format_func=lambda x: "All stars" if x == 0 else f"{x} stars only",
)
st.session_state["s_star_filter"] = None if star_val == 0 else int(star_val)

# ---------- Main Search UI ----------
st.title("📱 Sentimend | Google Play Sentiment Explorer")

with st.form("search_form", clear_on_submit=False):
    col1, col2 = st.columns([3, 1], vertical_alignment="bottom")
    query = col1.text_input(
        "Search for any app:",
        placeholder="i.e. WhatsApp, Spotify, ...",
        key="query_input",
    )
    submitted = col2.form_submit_button("🔍 Search", width="stretch")

if submitted and query.strip():
    st.session_state["candidates"] = cached_candidates(query.strip(), lang, country)
    st.session_state["best_pick"] = cached_best_match(query.strip(), lang, country)

# Best Match
if st.session_state.get("best_pick"):
    best = st.session_state["best_pick"]
    with st.expander("✨ Best match", expanded=True):
        cols = st.columns([1, 3, 2, 2, 2], vertical_alignment="center")
        cols[0].image(best.icon, width=64)
        cols[1].markdown(f"**{best.title}** - {best.developer}")
        if cols[4].button("Review in modal", key="use_best_modal"):
            st.session_state["modal_app"] = best
            st.session_state["open_modal"] = True
            st.rerun()

# Results Grid
cands = st.session_state.get("candidates", [])
if cands:
    st.subheader("Results")
    grid = st.columns(3, vertical_alignment="top")
    for idx, cand in enumerate(cands):
        with grid[idx % 3]:
            st.image(cand.icon, width=64)
            st.markdown(f"**{cand.title}** - {cand.developer}")
            st.caption(cand.app_id)
            _summary = (
                f"⭐ {cand.score:.2f}  •  {cand.installs}"
                if cand.installs
                else f"⭐ {cand.score:.2f}"
            )
            st.write(_summary)
            if st.button("Review in modal", key=f"select_{idx}"):
                st.session_state["modal_app"] = cand
                st.session_state["open_modal"] = True
                st.rerun()
            st.space()

# Logic to Keep Modal Open
if st.session_state.get("open_modal") and st.session_state.get("modal_app"):
    review_modal(st.session_state["modal_app"])
