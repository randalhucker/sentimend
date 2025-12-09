from typing import Callable, Dict, List, Optional

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

from google_play_scraper import Sort
from google_play_scraper import app as gp_app
from google_play_scraper import reviews
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from .models import ReviewRecord, SentimentSummary


class SentimentAnalyzer:
    """
    Fetch Google Play details/reviews and run sentiment analysis.
    Optionally accept a custom analyzer_fn(text)->compound[-1..1].
    """

    def __init__(
        self,
        lang: str = "en",
        country: str = "us",
        analyzer_fn: Optional[Callable[[str], float]] = None,
    ) -> None:
        self.lang = lang
        self.country = country
        self._vader = SentimentIntensityAnalyzer()
        self._custom = analyzer_fn

    def _compound(self, text: str) -> float:
        if self._custom:
            return float(self._custom(text))
        return float(self._vader.polarity_scores(text)["compound"])

    def fetch_app_details(self, app_id: str) -> Dict:
        return gp_app(app_id, lang=self.lang, country=self.country)

    def fetch_reviews(
        self,
        app_id: str,
        count: int = 200,
        sort_newest: bool = True,
        star_filter: Optional[int] = 0,  # single star value (1..5) or None
    ) -> List[ReviewRecord]:
        sort_order = Sort.NEWEST if sort_newest else Sort.MOST_RELEVANT
        result, _ = reviews(
            app_id,
            lang=self.lang,
            country=self.country,
            sort=sort_order,
            count=count,
            filter_score_with=star_filter or 0,
        )
        out: List[ReviewRecord] = []
        for r in result:
            out.append(
                ReviewRecord(
                    reviewId=r.get("reviewId", ""),
                    userName=r.get("userName", ""),
                    content=r.get("content", "") or "",
                    score=int(r.get("score", 0) or 0),
                    at=str(r.get("at", "")),
                    thumbsUpCount=int(r.get("thumbsUpCount", 0) or 0),
                )
            )
        return out

    def analyze(
        self, reviews_list: List[ReviewRecord], top_k: int = 25
    ) -> SentimentSummary:
        if not reviews_list:
            return SentimentSummary(
                total=0,
                avg_star_rating=0.0,
                avg_compound=0.0,
                positive=0,
                neutral=0,
                negative=0,
                top_positive_examples=[],
                top_negative_examples=[],
                top_keywords=[],
            )

        # 1. Build DataFrame & Calc Sentiment
        rows = []
        texts_for_keywords = []  # distinct list for vectorizer

        for r in reviews_list:
            # We collect texts here to avoid re-looping the dataframe later
            if r.content:
                texts_for_keywords.append(r.content)

            c = self._compound(r.content)
            rows.append(
                {
                    "text": r.content,
                    "stars": r.score,
                    "compound": c,
                    "thumbs": r.thumbsUpCount,
                }
            )

        df = pd.DataFrame(rows)
        pos = int((df["compound"] > 0.05).sum())
        neu = int(((df["compound"] >= -0.05) & (df["compound"] <= 0.05)).sum())
        neg = int((df["compound"] < -0.05).sum())

        top_positive = (
            df.sort_values(["compound", "thumbs"], ascending=[False, False])
            .head(5)["text"]
            .tolist()
        )
        top_negative = (
            df.sort_values(["compound", "thumbs"], ascending=[True, False])
            .head(5)["text"]
            .tolist()
        )

        # 2. Robust Keyword Extraction (CountVectorizer)
        top_keywords = []
        if texts_for_keywords:
            try:
                # stop_words="english" -> removes "the", "and", "is", etc.
                # min_df=2 -> ignores words that only appear in 1 review (typos/outliers)
                base_stops = list(
                    CountVectorizer(
                        stop_words="english"
                    ).get_stop_words()  # pyright: ignore[reportArgumentType]
                )
                custom_stops = base_stops + [
                    "app",
                    "application",
                    "phone",
                    "mobile",
                    "android",
                ]
                vec = CountVectorizer(
                    stop_words=custom_stops, min_df=2, max_features=top_k * 2
                )
                # This returns a sparse matrix of counts
                X = vec.fit_transform(texts_for_keywords)

                # Sum columns (words) across all rows (reviews)
                # Convert the sparse matrix to a dense array before summing to avoid static type issues
                if sparse.isspmatrix(X):
                    counts = np.asarray(sparse.csr_matrix(X).sum(axis=0)).ravel()
                else:
                    counts = np.asarray(
                        X.sum(axis=0)  # pyright: ignore[reportAttributeAccessIssue]
                    ).ravel()
                vocab = vec.get_feature_names_out()

                # Zip, Sort, and Slice
                freqs = list(zip(vocab, counts))
                top_keywords = sorted(freqs, key=lambda x: x[1], reverse=True)[:top_k]

            except ValueError:
                # This occurs if the vocabulary is empty (e.g., all words were stop words)
                top_keywords = []

        return SentimentSummary(
            total=len(df),
            avg_star_rating=float(df["stars"].mean()),
            avg_compound=float(df["compound"].mean()),
            positive=pos,
            neutral=neu,
            negative=neg,
            top_positive_examples=top_positive,
            top_negative_examples=top_negative,
            top_keywords=top_keywords,
        )
