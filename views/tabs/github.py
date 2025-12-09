import streamlit as st

import pandas as pd
import requests

from services.utils import *


def create_github_issue(repo_full_name, token, title, body, labels):
    url = f"https://api.github.com/repos/{repo_full_name}/issues"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json",
    }
    data = {"title": title, "body": body, "labels": labels}
    return requests.post(url, json=data, headers=headers)


@st.fragment
def render_github_sync(app_id: str):
    solutions_map = get_solutions(app_id)
    clusters_models = get_clusters(app_id)

    if not solutions_map or not clusters_models:
        st.info(
            "🔒 **Locked:** Please generate AI Solutions in the **Triage** tab first."
        )
        return

    st.header("Sync to GitHub")
    with st.expander("Configuration", expanded=True):
        col1, col2 = st.columns(2)
        repo_url = col1.text_input(
            "GitHub Repo URL", placeholder="https://github.com/owner/repo"
        )
        try:
            gh_token = st.secrets["github"]["token"]
        except KeyError:
            gh_token = ""
        gh_token = col2.text_input(
            "Personal Access Token",
            value=gh_token,
            type="password",
        )

    if repo_url and gh_token:
        # Parse owner/repo from URL
        try:
            # simplistic parser: remove trailing slash, split by /, take last 2
            clean_url = repo_url.rstrip("/")
            parts = clean_url.split("/")
            repo_path = f"{parts[-2]}/{parts[-1]}"
            st.success(f"Target: **{repo_path}**")
        except:
            st.error("Invalid URL format. Expected: https://github.com/owner/repo")
            repo_path = None

        if repo_path:
            st.subheader("Select Issues to Sync")

            # --- 1. PREPARE DATA ---
            # Retrieve triage data early to show Severity in the table
            triage_df = get_triage_df(app_id)
            severity_lookup = {}
            if triage_df is not None and not triage_df.empty:
                severity_lookup = dict(
                    zip(
                        triage_df["cluster_id"].astype(str),
                        triage_df["severity"],
                    )
                )

            preview_data = []
            clusters_map = {c.id: c for c in clusters_models}

            for cid, sol in solutions_map.items():
                c = clusters_map.get(cid)
                sev_score = severity_lookup.get(str(cid), 0.0)

                preview_data.append(
                    {
                        "Sync": True,  # Default to Checked
                        "Cluster": cid,
                        "Severity": float(f"{sev_score:.2f}"),  # Float for sorting
                        "Size": c.size if c else 0,
                        "Issue Title": sol.problem_summary,
                    }
                )

            # --- 2. INTERACTIVE TABLE ---
            df = pd.DataFrame(preview_data)

            # Use st.data_editor to allow checkboxes
            edited_df = st.data_editor(
                df,
                column_config={
                    "Sync": st.column_config.CheckboxColumn(
                        "Sync?",
                        help="Uncheck to skip this issue",
                        default=False,
                    ),
                    "Cluster": st.column_config.NumberColumn("ID", width="small"),
                    "Severity": st.column_config.ProgressColumn(
                        "Sev", format="%.2f", min_value=0, max_value=1, width="small"
                    ),
                    "Size": st.column_config.NumberColumn("Count", width="small"),
                    "Issue Title": st.column_config.TextColumn(
                        "Problem Summary", width="large"
                    ),
                },
                disabled=[
                    "Cluster",
                    "Severity",
                    "Size",
                    "Issue Title",
                ],  # Only allow editing "Sync"
                hide_index=True,
                width="stretch",
                key=f"editor_{app_id}",
            )

            # --- 3. FILTER & UPLOAD ---
            # Filter the dataframe for rows where 'Sync' is True
            selected_rows = edited_df[edited_df["Sync"] == True]
            selected_cids = selected_rows["Cluster"].tolist()

            count = len(selected_cids)

            if st.button(
                f"Create {count} Issues on GitHub",
                type="primary",
                disabled=(count == 0),
            ):
                progress_bar = st.progress(0)
                status_text = st.empty()
                success_count = 0

                # Iterate only over SELECTED CIDs
                for i, cid in enumerate(selected_cids):
                    sol = solutions_map.get(cid)
                    if not sol or not sol.problem_summary:
                        st.error(f"Missing solution data for Cluster {cid}, skipping.")
                        continue
                    c = clusters_map.get(cid)
                    sev_score = severity_lookup.get(str(cid), 0.0)

                    # --- Body Construction ---
                    body_lines = [
                        f"**Severity Score:** {sev_score:.2f}",
                        f"**Review Count:** {c.size if c else 'N/A'}",
                        "",
                        "## 🚨 Problem Context",
                        f"Keywords: *{', '.join(c.keywords) if c else 'Unknown'}*",
                        "",
                        "## 💡 Proposed Solutions",
                    ]
                    body_lines.extend([f"- {s}" for s in sol.solutions])

                    body_lines.append("")
                    body_lines.append("## ✅ Acceptance Criteria")
                    body_lines.extend([f"- [ ] {m}" for m in sol.metrics])

                    if sol.key_signals:
                        body_lines.append("")
                        body_lines.append("## 🔍 Signals")
                        body_lines.extend([f"- {sig}" for sig in sol.key_signals])

                    full_body = "\n".join(body_lines)
                    labels = ["user-feedback", "ai-triage"]

                    # --- API Call ---
                    status_text.write(f"Creating issue for Cluster {cid}...")
                    resp = create_github_issue(
                        repo_path,
                        gh_token,
                        sol.problem_summary,
                        full_body,
                        labels,
                    )

                    if resp.status_code == 201:
                        success_count += 1
                    else:
                        st.error(
                            f"Failed to create issue for Cluster {cid}: {resp.status_code} - {resp.text}"
                        )

                    progress_bar.progress((i + 1) / count)

                progress_bar.empty()
                status_text.empty()

                if success_count == count:
                    st.balloons()
                    st.success(
                        f"Successfully created {success_count} issues in {repo_path}!"
                    )
                    st.link_button("View Repository", repo_url)
                else:
                    st.warning(
                        f"Completed with errors. Created {success_count}/{count} issues."
                    )
