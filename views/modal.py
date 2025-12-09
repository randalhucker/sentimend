import streamlit as st

from .tabs import overview, reviews, triage, charts, github
from services.models import PackageCandidate
from services.utils import *


def _on_dismiss():
    st.session_state["modal_app"] = None
    st.session_state["open_modal"] = False


@st.dialog("📦 App Review", width="large", on_dismiss=_on_dismiss)
def review_modal(app_candidate: PackageCandidate):
    app_id = app_candidate.app_id
    # Settings passed from sidebar via session state
    lang = st.session_state.get("s_lang", "en")
    country = st.session_state.get("s_country", "us")
    count = st.session_state.get("s_count", 400)
    sort_newest = st.session_state.get("s_sort", False)
    star_filter = st.session_state.get("s_star_filter", None)

    details = cached_app_details(app_id, lang, country)
    tabs = st.tabs(["Overview", "Reviews", "Charts", "Triage", "GitHub Sync"])

    with tabs[0]:
        overview.render_overview(
            app_id, details, lang, country, count, sort_newest, star_filter
        )

    with tabs[1]:
        reviews.render_reviews(app_id)

    with tabs[2]:
        charts.render_charts(app_id)

    with tabs[3]:
        triage.render_triage(app_id, details.get("title", app_id))

    with tabs[4]:
        github.render_github_sync(app_id)
