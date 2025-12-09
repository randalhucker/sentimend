from typing import Optional
import streamlit as st

import pandas as pd

from services.clustering import ReviewClusterer
from services.exporters import (
    generate_jira_csv,
    generate_json_export,
    generate_markdown_report,
)
from services.models import ClusterSolution
from services.solution_generator import OpenAIModels, GeminiModels, SolutionGenerator
from services.triage import compute_cluster_severity
from services.utils import *


@st.fragment
def render_triage(app_id: str, app_title: str):
    reviews_df = get_df(app_id)
    summary = get_summary(app_id)

    if summary is None or reviews_df is None:
        st.info("Please run analysis in the Overview tab first.")
        return

    st.subheader("Cluster Settings")

    # 1. Filter
    neg_mode = st.radio(
        "Negative Filter:",
        ["compound < -0.05", "stars ≤ 2", "Both"],
        horizontal=True,
        key=f"negmode_{app_id}",
    )

    _df = reviews_df.copy()
    if neg_mode == "compound < -0.05":
        _neg_df = _df[_df["compound"] < -0.05]
    elif neg_mode == "stars ≤ 2":
        _neg_df = _df[_df["score"] <= 2]
    else:
        _neg_df = _df[(_df["compound"] < -0.05) & (_df["score"] <= 2)]

    texts = [str(t) for t in _neg_df["content"].tolist() if str(t).strip()]
    df_indices = list(_neg_df.index)
    st.caption(f"Negative subset size: {len(texts)}")

    # 2. Clustering
    col_c1, col_c2 = st.columns([1, 2], vertical_alignment="center", gap="large")
    with col_c1:
        dynamic_k = st.checkbox(
            "Dynamically-select clusters",
            value=True,
            help="Enable the dynamic selection of the number of clusters based on distance threshold.",
            key=f"dynamic_k_{app_id}",
        )

        min_cluster_size = st.slider(
            "Minimum Cluster Size",
            min_value=2,
            max_value=20,
            value=2,
            help="Minimum number of reviews required per cluster to be shown.",
            key=f"min_cluster_size_{app_id}",
            width="stretch",
        )
    with col_c2:
        k_val = st.slider("Cluster Count (k)", 2, 20, 8) if not dynamic_k else None

    if st.button("Cluster Negative Reviews", key=f"group_{app_id}"):
        with st.spinner("Analyzing clusters...", show_time=True):
            if len(texts) < (k_val or 2) * 2:
                st.warning("Not enough negative reviews.")
            else:
                clusterer = ReviewClusterer()
                clusters, sil_score, viz_df = clusterer.cluster_texts(
                    texts=texts,
                    df_indices=df_indices,
                    k=k_val,
                    min_cluster_size=min_cluster_size,
                )

                st.session_state[f"sil_score_{app_id}"] = sil_score

                triage_rows = []
                for c in clusters:
                    sev = compute_cluster_severity(c, reviews_df=_df, recent_days=60)
                    triage_rows.append(
                        {
                            "cluster_id": c.id,
                            "size": c.size,
                            "severity": sev,
                            "keywords": ", ".join(c.keywords),
                        }
                    )

                try:
                    triage_df = pd.DataFrame(triage_rows).sort_values(
                        ["severity", "size"], ascending=False
                    )
                except KeyError:
                    st.error(
                        "No clusters fit the **`Minimum Cluster Size`** requested. Try raising the minimum threshold distance."
                    )
                    return

                set_clusters_triage_and_viz(app_id, clusters, triage_df, viz_df)

    # Retrieve Computed Data
    triage_df = get_triage_df(app_id)
    clusters_models = get_clusters(app_id)
    solutions_map = get_solutions(app_id)

    if triage_df is not None and clusters_models is not None:
        clusters_map = {c.id: c for c in clusters_models}

        # 3. AI Solutions
        st.divider()
        st.subheader("AI Analysis & Issue Generation")
        with st.expander("LLM Configuration:"):
            extra = st.text_area(
                "Context for LLM:",
                placeholder="Additional context for AI analysis. i.e., recent app updates, user demographics, etc.",
                key=f"extra_{app_id}",
            )
            selected_model = st.selectbox(
                "LLM Model:",
                list(GeminiModels) + list(OpenAIModels),
                format_func=lambda x: x.value,
                index=2,
                help="Select the LLM model to use for generating solutions.",
                key=f"model_{app_id}",
            )

        if st.button("Perform Analysis", type="primary", key=f"gensol_{app_id}"):
            with st.spinner("Contacting LLM service...", show_time=True):
                gen = SolutionGenerator(model=selected_model)
                if not gen.available():
                    st.error("LLM client not available.")
                else:
                    sols = gen.generate_for_clusters(
                        clusters=clusters_models,
                        app_name=app_title,
                        extra_instructions=extra,
                    )
                    set_solutions(app_id, {s.cluster_id: s for s in sols})
                    st.toast("Solutions generated.", icon="✅")
                    st.rerun()

        # 4. Display & Expanders
        st.markdown("### Review")

        score = st.session_state.get(f"sil_score_{app_id}", 0.0)
        # Color code the score
        color = "normal"
        if score > 0.5:
            color = "normal"  # Streamlit metric handles green via delta
        elif score < 0.2:
            color = "off"
        st.metric(
            "Clustering Quality",
            f"{score:.2f}",
            help="Silhouette Score (-1 to 1). Higher is better separated.",
            delta="Good" if score > 0.4 else None,
            delta_color=color,
        )

        display_df = triage_df.copy()
        display_df["cluster_id"] = display_df["cluster_id"].astype(str)
        display_df["size"] = display_df["size"].astype(str)
        display_df["severity"] = display_df["severity"].apply(lambda x: f"{x:.2f}")

        st.dataframe(
            display_df[["cluster_id", "severity", "size", "keywords"]],
            width="stretch",
            hide_index=True,
            column_config={
                "cluster_id": "Cluster ID",
                "severity": "Severity",
                "size": "Size",
                "keywords": "Keywords",
            },
        )

        # Show Detailed Expanders
        st.write("**Detailed Breakdown:**")
        for _, row in triage_df.iterrows():
            cid = int(row["cluster_id"])
            c = clusters_map.get(cid)
            s: Optional[ClusterSolution] = solutions_map.get(cid)

            # Dynamic Header: Show Problem Summary if available, else keywords
            header_text = f"Cluster {cid} (Sev: {row['severity']:.2f})"
            if s:
                header_text += f" | {s.problem_summary}"
            else:
                if c is not None:
                    header_text += f" | {', '.join(c.keywords[:4])}..."

            with st.expander(header_text):
                col_det1, col_det2 = st.columns([1, 1])

                # Left: The AI Analysis
                with col_det1:
                    if s:
                        st.markdown("## AI Analysis")
                        st.markdown(f"**Summary:** {s.problem_summary}")
                        st.markdown("**Proposed Solutions:**")
                        for sol in s.solutions:
                            st.markdown(f"- {sol}")
                        st.markdown("**Success Metrics:**")
                        for met in s.metrics:
                            st.code(met, language="text")
                    else:
                        st.info("Click 'Perform Analysis' to populate this report")

                # Right: The Raw Data
                with col_det2:
                    st.markdown("## User Reviews (Samples)")
                    if c is not None:
                        for ex in c.examples:
                            st.markdown(f"> *{ex}*")
                        st.caption(f"Total reviews in group: {c.size}")

        # 5. Export
        st.divider()
        st.subheader("Export")
        # Ensure maps are ready
        clusters_map = {c.id: c for c in clusters_models}

        # Generate Data
        csv_data = generate_jira_csv(triage_df, clusters_map, solutions_map)
        json_data = generate_json_export(triage_df, solutions_map)
        md_data = generate_markdown_report(triage_df, solutions_map)

        # Render Buttons
        b_col1, b_col2, b_col3 = st.columns(3)

        with b_col1:
            st.download_button(
                "Download Jira CSV",
                data=csv_data,
                file_name="jira_backlog.csv",
                mime="text/csv",
                key=f"j_{app_id}",
                width="stretch",
            )

        with b_col2:
            st.download_button(
                "Download JSON",
                data=json_data,
                file_name="data.json",
                mime="application/json",
                key=f"js_{app_id}",
                width="stretch",
            )

        with b_col3:
            st.download_button(
                "Download Markdown",
                data=md_data,
                file_name="report.md",
                mime="text/markdown",
                key=f"m_{app_id}",
                width="stretch",
            )
