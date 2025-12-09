from typing import List, Tuple
from pydantic import BaseModel, Field


# ---------- Identifier ----------
class PackageCandidate(BaseModel):
    app_id: str
    title: str
    developer: str
    score: float = 0.0
    installs: str = ""
    icon: str = ""
    url: str = ""


# ---------- Sentiment ----------
class ReviewRecord(BaseModel):
    reviewId: str = ""
    userName: str = ""
    content: str = ""
    score: int = 0
    at: str = ""
    thumbsUpCount: int = 0


class SentimentSummary(BaseModel):
    total: int
    avg_star_rating: float
    avg_compound: float
    positive: int
    neutral: int
    negative: int
    top_positive_examples: List[str] = Field(default_factory=list)
    top_negative_examples: List[str] = Field(default_factory=list)
    top_keywords: List[Tuple[str, int]] = Field(default_factory=list)


# ---------- Clustering ----------
class ClusterModel(BaseModel):
    id: int
    size: int
    keywords: List[str]
    indices: List[int] = Field(default_factory=list)  # indices into the source subset
    df_indices: List[int] = Field(default_factory=list)  # original DataFrame indices
    examples: List[str] = Field(default_factory=list)


# ---------- Solutions ----------
class ClusterSolution(BaseModel):
    cluster_id: int
    problem_summary: str
    key_signals: List[str] = Field(default_factory=list)
    solutions: List[str] = Field(default_factory=list)
    metrics: List[str] = Field(default_factory=list)
