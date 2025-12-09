import streamlit as st

from services.utils import *


@st.fragment
def render_reviews(app_id: str):
    reviews_df = get_df(app_id)
    summary = get_summary(app_id)

    if summary is None or reviews_df is None:
        st.info(
            "Analysis runs automatically on open. If empty, click **Refetch & analyze** in Overview."
        )
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("**Reviews Analyzed**:", f"{summary.total:,}", help="Total number of reviews analyzed.")
    m2.metric("**Avg. Star Rating**:", f"{summary.avg_star_rating:.2f}", help="Average star rating of analyzed reviews.")
    m3.metric("**Avg. Sentiment**:", f"{summary.avg_compound:.3f}", help="Average sentiment score ranging from -1 (negative) to +1 (positive).")
    m4.metric(
        "**Pos / Neu / Neg Counts**:",
        f"{summary.positive} / {summary.neutral} / {summary.negative}",
        help="Counts of positive, neutral, and negative reviews.",
    )

    with st.expander("Raw Reviews (first 200)", expanded=False):
        st.dataframe(reviews_df.head(200), width="stretch")

    colA, colB = st.columns(2, border=True)
    with colA:
        st.header("Top Positive", divider="green")
        for t in summary.top_positive_examples:
            st.markdown(f"- {t}")
    with colB:
        st.header("Top Negative", divider="red")
        for t in summary.top_negative_examples:
            st.markdown(f"- {t}")
