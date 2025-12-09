from typing import Dict

import json
import pandas as pd

from services.models import ClusterModel, ClusterSolution


def generate_jira_csv(
    triage_df: pd.DataFrame,
    clusters_map: Dict[int, ClusterModel],
    solutions_map: Dict[int, ClusterSolution],
) -> bytes:
    """Generates a CSV byte string formatted for Jira import."""
    jira_rows = []

    for _, row in triage_df.iterrows():
        cid = int(row["cluster_id"])
        c = clusters_map.get(cid)
        s = solutions_map.get(cid)

        # 1. Summary
        summary_txt = (s.problem_summary if s else f"Problem cluster {cid}")[:250]

        # 2. Labels
        labels = ["user-feedback", "android"] + (c.keywords[:3] if c else [])

        # 3. Acceptance Criteria
        acceptance = (
            "; ".join(s.metrics) if (s and s.metrics) else "Metric-based targets TBD"
        )

        # 4. Priority Logic
        severity = row["severity"]
        if severity >= 0.66:
            priority = "High"
        elif severity >= 0.33:
            priority = "Medium"
        else:
            priority = "Low"

        # 5. Description Body
        description_parts = []
        if s:
            if s.key_signals:
                description_parts.append(
                    "**Signals:**\n- " + "\n- ".join(s.key_signals)
                )
            if s.solutions:
                description_parts.append(
                    "**Proposed Solutions:**\n- " + "\n- ".join(s.solutions)
                )
        else:
            description_parts.append(
                "Cluster keywords: " + ", ".join(c.keywords if c else [])
            )

        jira_rows.append(
            {
                "Summary": summary_txt,
                "Description": "\n\n".join(description_parts),
                "Labels": ",".join(labels),
                "Priority": priority,
                "Acceptance Criteria": acceptance,
            }
        )

    return pd.DataFrame(jira_rows).to_csv(index=False).encode("utf-8")


def generate_json_export(
    triage_df: pd.DataFrame, solutions_map: Dict[int, ClusterSolution]
) -> bytes:
    """Generates a JSON byte string of the raw data."""
    export_json = {
        "triage": triage_df.to_dict(orient="records"),
        "solutions": {str(cid): s.model_dump() for cid, s in solutions_map.items()},
    }
    return json.dumps(export_json, indent=2).encode("utf-8")


def generate_markdown_report(
    triage_df: pd.DataFrame,
    solutions_map: Dict[int, ClusterSolution],
) -> bytes:
    """Generates a Markdown report string."""
    md_lines = ["# Problem Clusters & Solutions\n"]

    for _, row in triage_df.iterrows():
        cid = int(row["cluster_id"])
        s = solutions_map.get(cid)

        md_lines.append(
            f"## Cluster {cid} | Severity: {row['severity']:.2f} | Size: {row['size']}"
        )

        if s:
            md_lines.append(f"**Summary:** {s.problem_summary}")
            if s.solutions:
                md_lines.append("\n**Solutions:**")
                md_lines.extend([f"- {x}" for x in s.solutions])

            if s.metrics:
                md_lines.append("\n**Acceptance Criteria:**")
                md_lines.extend([f"- {x}" for x in s.metrics])

        md_lines.append("")  # Empty line between sections

    return "\n".join(md_lines).encode("utf-8")
