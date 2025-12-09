from typing import List, Optional

import streamlit as st
import pandas as pd

from services.identifier_finder import PackageIdentifierFinder
from services.sentiment_analyzer import SentimentAnalyzer
from services.models import (
    PackageCandidate,
    SentimentSummary,
    ClusterModel,
    ClusterSolution,
    ReviewRecord,
)


# ---------- Cache ----------
@st.cache_resource
def get_identifier_finder(lang: str, country: str) -> PackageIdentifierFinder:
    return PackageIdentifierFinder(lang=lang, country=country)


@st.cache_resource
def get_analyzer(lang: str, country: str) -> SentimentAnalyzer:
    return SentimentAnalyzer(lang=lang, country=country)


@st.cache_resource(show_spinner=False)
def cached_candidates(query: str, lang: str, country: str) -> List[PackageCandidate]:
    finder = get_identifier_finder(lang, country)
    return finder.search_candidates(query, max_results=15)


@st.cache_resource(show_spinner=False)
def cached_best_match(
    query: str, lang: str, country: str
) -> Optional[PackageCandidate]:
    finder = get_identifier_finder(lang, country)
    return finder.best_match(query, max_results=20)


@st.cache_data(show_spinner=False)
def cached_app_details(app_id: str, lang: str, country: str) -> dict:
    analyzer = get_analyzer(lang, country)
    return analyzer.fetch_app_details(app_id)


@st.cache_data(show_spinner=True)
def cached_reviews(
    app_id: str,
    lang: str,
    country: str,
    count: int,
    sort_newest: bool,
    star_filter: Optional[int],
):
    analyzer = get_analyzer(lang, country)
    reviews_list = analyzer.fetch_reviews(
        app_id, count=count, sort_newest=sort_newest, star_filter=star_filter
    )
    return pd.DataFrame([r.model_dump() for r in reviews_list])


@st.cache_resource(show_spinner=True)
def cached_summary(
    reviews_df: pd.DataFrame, lang: str, country: str
) -> SentimentSummary:
    analyzer = get_analyzer(lang, country)
    records = reviews_df.to_dict(orient="records")
    reviews_list = [
        ReviewRecord(**{str(k): v for k, v in record.items()}) for record in records
    ]
    return analyzer.analyze(reviews_list)


# ---------- State Helpers ----------
def _key(prefix: str, app_id: str) -> str:
    return f"{prefix}:{app_id}"


def get_df(app_id: str) -> Optional[pd.DataFrame]:
    return st.session_state.get(_key("reviews_df", app_id))


def get_summary(app_id: str) -> Optional[SentimentSummary]:
    return st.session_state.get(_key("summary", app_id))


def set_df_and_summary(
    app_id: str, df: pd.DataFrame, summary: SentimentSummary
) -> None:
    st.session_state[_key("reviews_df", app_id)] = df
    st.session_state[_key("summary", app_id)] = summary


def get_clusters(app_id: str) -> Optional[List[ClusterModel]]:
    return st.session_state.get(_key("clusters_models", app_id))


def get_triage_df(app_id: str) -> Optional[pd.DataFrame]:
    return st.session_state.get(_key("clusters_triage_df", app_id))


def get_viz_df(app_id: str) -> Optional[pd.DataFrame]:
    return st.session_state.get(_key("viz_df", app_id))


def set_clusters_triage_and_viz(
    app_id: str,
    clusters: List[ClusterModel],
    triage_df: pd.DataFrame,
    viz_df: pd.DataFrame,
) -> None:
    st.session_state[_key("clusters_models", app_id)] = clusters
    st.session_state[_key("clusters_triage_df", app_id)] = triage_df
    st.session_state[_key("viz_df", app_id)] = viz_df


def get_solutions(app_id: str) -> dict[int, ClusterSolution]:
    return st.session_state.get(_key("cluster_solutions", app_id), {})


def set_solutions(app_id: str, sols_map: dict[int, ClusterSolution]) -> None:
    st.session_state[_key("cluster_solutions", app_id)] = sols_map


def auto_key(
    app_id: str,
    lang: str,
    country: str,
    count: int,
    sort_newest: bool,
    star_filter: Optional[int],
) -> str:
    sf = 0 if star_filter is None else int(star_filter)
    return f"auto:{app_id}:{lang}:{country}:{count}:{1 if sort_newest else 0}:{sf}"


def clear_downstream_state(app_id: str) -> None:
    st.session_state.pop(_key("clusters_models", app_id), None)
    st.session_state.pop(_key("clusters_triage_df", app_id), None)
    st.session_state.pop(_key("cluster_solutions", app_id), None)
