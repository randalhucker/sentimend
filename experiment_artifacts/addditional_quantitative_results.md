
# 4.1 Quantitative Results (Summary Tables)

## 4.1.1 Cluster Coherence & Quality

| Dataset   | Silhouette Score |
|-----------|------------------:|
| Spotify   | 0.04              |
| Google    | 0.04              |
| WhatsApp  | 0.04              |

> **Interpretation:** Scores are modest due to highly unstructured text and cross-topic similarities, but acceptable for noisy real-world review data.

---

## 4.1.2 Effort Reduction — Negative Review Filtering

| App       | Total Reviews | Negative Subset | Reduction    |
|-----------|---------------:|----------------:|--------------|
| Spotify   |            500 |             203 | **59.4%**    |
| Google    |           1000 |             513 | **48.7%**    |
| WhatsApp  |           2000 |             797 | **60.15%**   |

> **Outcome:** ~2,500 total reviews reduced to ~1,513 negative-sentiment reviews for targeted problem discovery.

### Runtime Comparison

| Task                     | Duration      |
|--------------------------|---------------|
| Manual review (est.)     | Hours         |
| Sentimend processing     | ~3 seconds    |

> **Result:** A lightweight sentiment filter isolates **high-signal feedback** with minimal overhead.

---

## 4.1.3 Effort Reduction — Clustering Time

| Method        | Reviews | Time Required         |
|---------------|--------:|----------------------:|
| Manual        |     200 | 25–45 min            |
| Automated     |    2500 | <5 sec               |

### Speedup

| Comparison Metric           | Value        |
|-----------------------------|-------------:|
| Speedup (worst-case)        | **3750×**     |

200 reviews / (25 minutes * 60 seconds) = 0.13333 reviews / second

2500 reviews / 5 seconds = 500 reviews / second

500 / 0.13333 ~ 3750

> **Conclusion:** Automated hierarchical clustering enables meaningful triaging at scale, converting a multi-hour task into seconds.
