from typing import List, Optional, Tuple

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from .models import ClusterModel


class ReviewClusterer:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        tfidf_max_features: int = 2000,
        tfidf_ngram_range: Tuple[int, int] = (1, 2),
    ) -> None:
        self.encoder = SentenceTransformer(embedding_model)
        self.vectorizer = TfidfVectorizer(
            max_features=tfidf_max_features,
            ngram_range=tfidf_ngram_range,
            stop_words="english",
            strip_accents="unicode",
            lowercase=True,
            min_df=1,
        )

    def _keywords_for_cluster(self, texts: List[str], top_terms: int = 6) -> List[str]:
        if not texts:
            return []
        if len(texts) == 1:
            return ["Unique Review"]
        try:
            X = self.vectorizer.fit_transform(texts)
            terms = self.vectorizer.get_feature_names_out()
        except ValueError:
            return []

        if sparse.isspmatrix(X):
            summed = np.asarray(sparse.csr_matrix(X).sum(axis=0)).ravel()
        else:
            summed = np.asarray(
                X.sum(axis=0)  # pyright: ignore[reportAttributeAccessIssue]
            ).ravel()

        top_k = min(top_terms, summed.size)
        if top_k == 0:
            return []
        top_idx = np.argpartition(summed, -top_k)[-top_k:]
        top_idx = top_idx[np.argsort(summed[top_idx])[::-1]]
        return terms[top_idx].tolist()

    def _representatives(self, embeddings: np.ndarray, count: int = 3) -> List[int]:
        if embeddings.shape[0] == 0:
            return []
        centroid = embeddings.mean(axis=0, keepdims=True)

        def _norm(a):
            n = np.linalg.norm(a, axis=1, keepdims=True) + 1e-12
            return a / n

        sims = cosine_similarity(_norm(embeddings), _norm(centroid)).ravel()
        order = np.argsort(sims)[::-1]
        return order[: max(1, min(count, embeddings.shape[0]))].tolist()

    def cluster_texts(
        self,
        texts: List[str],
        df_indices: Optional[List[int]] = None,
        *,
        k: Optional[int] = None,
        # Increased threshold slightly because Euclidean distance is on a different scale than Cosine
        # Euclidean 1.0 ~= Cosine 0.5 for unit vectors
        distance_threshold: float = 2.0,
        top_terms: int = 6,
        examples_per_cluster: int = 3,
        min_cluster_size: int = 2,
    ) -> Tuple[List[ClusterModel], float, pd.DataFrame]:

        if not texts:
            return [], 0.0, pd.DataFrame()

        # 1. Encode (Must Normalize for Ward/Euclidean to act like Cosine)
        emb = self.encoder.encode(
            texts, batch_size=64, show_progress_bar=False, normalize_embeddings=True
        )
        n_samples = emb.shape[0]

        # 2. Configure Clustering (Switching to Ward)
        if k is not None and k > 1:
            model = AgglomerativeClustering(
                n_clusters=k,
                linkage="ward",
                metric="euclidean",
            )
        else:
            model = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=distance_threshold,
                linkage="ward",
                metric="euclidean",
            )

        try:
            labels = model.fit_predict(emb)
        except Exception:
            labels = np.zeros(n_samples, dtype=int)

        # 3. Silhouette Score
        try:
            unique_labels = set(labels)
            if len(unique_labels) > 1 and len(texts) > 1:
                score = silhouette_score(emb, labels, metric="euclidean")
            else:
                score = 0.0
        except Exception:
            score = 0.0

        # 4. Generate t-SNE Coordinates for Visualization (NEW)
        # We reduce 384 dims -> 2 dims
        # perplexity must be < n_samples
        perp = min(30, n_samples - 1) if n_samples > 1 else 1
        tsne = TSNE(
            n_components=2,
            perplexity=perp,
            random_state=42,
            init="pca",
            learning_rate="auto",
        )
        coords = tsne.fit_transform(emb)

        viz_data = []
        for i in range(n_samples):
            viz_data.append(
                {
                    "x": coords[i, 0],
                    "y": coords[i, 1],
                    "cluster_id": int(labels[i]),
                    "text_snippet": texts[i][:100] + "...",  # Truncate for tooltip
                }
            )
        viz_df = pd.DataFrame(viz_data)

        # 5. Build Models & Filter Singletons
        active_clusters = set()
        clusters: List[ClusterModel] = []
        for cid in sorted(unique_labels):
            idxs = [i for i, lbl in enumerate(labels) if lbl == cid]

            # --- FILTER: Skip clusters smaller than min_size ---
            if len(idxs) < min_cluster_size:
                continue

            active_clusters.add(cid)

            sub_texts = [texts[i] for i in idxs]
            sub_emb = emb[idxs]

            keywords = self._keywords_for_cluster(sub_texts, top_terms=top_terms)

            rep_local_indices = self._representatives(
                sub_emb, count=min(examples_per_cluster, len(sub_texts))
            )
            examples = [sub_texts[i] for i in rep_local_indices]
            original_df_idxs = [df_indices[i] for i in idxs] if df_indices else idxs

            clusters.append(
                ClusterModel(
                    id=int(cid),
                    size=len(idxs),
                    keywords=keywords,
                    indices=idxs,
                    df_indices=original_df_idxs,
                    examples=examples,
                )
            )

        viz_df["cluster_label"] = viz_df["cluster_id"].apply(
            lambda x: f"Cluster {x}" if x in active_clusters else "Other/Noise"
        )
        kw_map = {c.id: ", ".join(c.keywords[:3]) for c in clusters}
        viz_df["keywords"] = viz_df["cluster_id"].map(kw_map).fillna("Heterogeneous")

        clusters.sort(key=lambda c: c.size, reverse=True)
        return clusters, float(score), viz_df
