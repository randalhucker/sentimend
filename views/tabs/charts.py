import streamlit as st

import altair as alt
import pandas as pd

from services.utils import *


@st.fragment
def render_charts(app_id: str):
    reviews_df = get_df(app_id)
    summary = get_summary(app_id)

    if summary is None or reviews_df is None:
        st.info("Analysis runs automatically on open.")
        return

    st.subheader(
        "Sentiment Distribution",
        help="Distribution of positive, neutral, and negative sentiments.",
    )
    buckets = pd.DataFrame(
        {
            "sentiment": ["Positive", "Neutral", "Negative"],
            "Count of Reviews": [summary.positive, summary.neutral, summary.negative],
        }
    )
    st.altair_chart(
        alt.Chart(buckets)
        .mark_bar()
        .encode(
            x="sentiment",
            y="Count of Reviews",
            tooltip=["sentiment", "Count of Reviews"],
        ),
        width="stretch",
    )

    st.subheader(
        "Compound Score Histogram",
        help="A compound sentiment score is a normalized, weighted composite score ranging from -1 (most extreme negative) to +1 (most extreme positive).",
    )
    st.altair_chart(
        alt.Chart(reviews_df)
        .mark_bar()
        .encode(alt.X("compound:Q", bin=alt.Bin(maxbins=30)), y="count()"),
        width="stretch",
    )

    st.subheader(
        "Top Keywords",
        help="The most frequently occurring keywords in all reviews. (Excludes common stop words)",
    )
    kw_df = pd.DataFrame(summary.top_keywords, columns=["Unique Tokens", "count"]).head(
        25
    )
    st.altair_chart(
        alt.Chart(kw_df)
        .mark_bar()
        .encode(
            x="count:Q",
            y=alt.Y("Unique Tokens:N", sort="-x"),
            tooltip=["Unique Tokens", "count"],
        ),
        width="stretch",
    )

    st.divider()
    st.subheader(
        "Cluster Map (t-SNE)",
        help="2D projection of review similarity. Points closer together are semantically similar.",
    )
    viz_df = get_viz_df(app_id)
    if viz_df is not None and not viz_df.empty:
        # Create an interactive scatter plot
        scatter = (
            alt.Chart(viz_df)
            .mark_circle(size=60)
            .encode(
                x=alt.X("x", axis=None),  # Hide axes for cleaner look
                y=alt.Y("y", axis=None),
                color=alt.Color("cluster_label", legend=alt.Legend(title="Cluster")),
                tooltip=[
                    alt.Tooltip("cluster_label", title="Group"),
                    alt.Tooltip("keywords", title="Keywords"),
                    alt.Tooltip("text_snippet", title="Review"),
                ],
            )
            .interactive()
            .properties(height=500)
        )

        st.altair_chart(scatter, width="stretch")
    else:
        st.info(
            "Cluster visualization is not available. Please run triage analysis first."
        )
