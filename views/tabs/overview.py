import streamlit as st

from services.sentiment_analyzer import SentimentAnalyzer
from services.utils import *


@st.fragment
def render_overview(
    app_id: str,
    details: dict,
    lang: str,
    country: str,
    count: int,
    sort_newest: bool,
    star_filter_param,
):
    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("**⭐ Rating**:", f"{float(details.get('score', 0) or 0):.2f}")
    c2.metric("**Number of Ratings**:", f"{int(details.get('ratings', 0) or 0):,}")
    c3.metric("**Installs**:", details.get("installs", "—"))
    c4.metric("**Genre**:", (details.get("genre", "—")))

    st.write("**Summary:** " + (details.get("summary", "—")))

    with st.expander("More details"):
        st.json(
            {
                k: details.get(k)
                for k in ["appId", "title", "developer", "description", "free", "score"]
            }
        )

    st.markdown("### 🧪 Analysis")

    # Auto-run Logic
    auto_key_val = auto_key(
        app_id, lang, country, count, sort_newest, star_filter_param
    )

    def _fetch_and_analyze():
        clear_downstream_state(app_id)
        df = cached_reviews(
            app_id, lang, country, count, sort_newest, star_filter_param
        )
        _an = SentimentAnalyzer(lang=lang, country=country)
        df["compound"] = df["content"].map(_an._compound)
        summary = cached_summary(df, lang, country)
        set_df_and_summary(app_id, df, summary)

    if not st.session_state.get(auto_key_val):
        with st.spinner("Fetching & analyzing reviews...", show_time=True):
            _fetch_and_analyze()
        st.session_state[auto_key_val] = True
        st.toast("Analysis ready.", icon="✅")

    if st.button("Refetch & analyze", key=f"analyze_{app_id}"):
        with st.spinner("Refetching...", show_time=True):
            _fetch_and_analyze()
        st.session_state[auto_key_val] = True
        st.toast("Updated.", icon="✅")
