import pandas as pd

from .models import ClusterModel


def compute_cluster_severity(
    cluster: ClusterModel,
    reviews_df: pd.DataFrame,
    recent_days: int = 60,
) -> float:
    """
    Heuristic severity score in [0, 1]:
      0.40 * frequency      (cluster size / total negatives)
      0.30 * negativity     (mean max(0, -compound))
      0.20 * low_star_rate  (% of 1-2 star)
      0.10 * recency        (% within recent_days)
    Assumes reviews_df has 'compound', 'stars', and 'at' parseable by pandas.to_datetime.
    """
    if len(reviews_df) == 0 or cluster.size == 0:
        return 0.0

    rows = reviews_df.loc[cluster.df_indices].copy()

    # frequency
    frequency = cluster.size / max(1, len(reviews_df))

    # negativity
    neg_component = (
        rows["compound"].apply(lambda x: max(0.0, -float(x))).mean()
        if "compound" in rows
        else 0.0
    )

    # low star rate
    low_star_rate = 0.0
    if "stars" in rows:
        low_star_rate = (rows["stars"] <= 2).mean()

    # recency
    recency = 0.0
    if "at" in rows:
        try:
            rows["at"] = pd.to_datetime(
                rows["at"], errors="coerce", utc=True, format="mixed"
            )
            cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=recent_days)
            recency = rows["at"].ge(cutoff).mean()
        except Exception:
            recency = 0.0

    score = (
        0.40 * frequency
        + 0.30 * float(neg_component)
        + 0.20 * float(low_star_rate)
        + 0.10 * float(recency)
    )
    # clamp
    return float(max(0.0, min(1.0, score)))
