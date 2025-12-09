# Sentimend

## Google Play Store Sentiment Explorer (Streamlit)

A modular Streamlit app that:

1. Lets you fuzzy-search for an Android app by name.
2. Finds the Google Play package ID (e.g., `com.spotify.music`).
3. Fetches reviews and runs sentiment analysis.
4. Presents clean, attractive charts and allows CSV/JSON export.

## Tech Stack

- **UI**: [Streamlit](https://streamlit.io/) & [Altair](https://altair-viz.github.io/)
- **ML/NLP**: [VADER Sentiment Analysis](https://github.com/cjhutto/vaderSentiment), [Scikit-learn](https://scikit-learn.org/stable/) & [sentence-transformers](https://www.sbert.net/)
- **LLM Orchestration**: [Google Generative AI SDK](https://developers.generativeai.google/) & [OpenAI API](https://openai.com/api/)
- **Data Fetching**: [google-play-scraper](https://github.com/facundoolano/google-play-scraper)

## Quickstart

### 1. Environment Setup:

- Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/products/distribution).
- Clone this repository.
- Navigate to the project directory.

```bash
# create & activate a conda environment with the necessary dependencies
conda env create -f environment.yml
conda activate sentimend
```

### 2. API Configuration:

To use the AI generation and GitHub sync features, you need to configure your secrets. Create a file at `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml

# Required for AI Solutions
[api]
openai = "your-openai-key"
gemini = "your-gemini-key"

# Optional: Default GitHub token (can also be entered in UI)
[github]
token = "your-personal-access-token"
```

**Get your API keys from:**

- [OpenAI API Keys](https://platform.openai.com/settings/organization/api-keys)
- [Google Gemini API Keys](https://aistudio.google.com/api-keys)
- [GitHub Personal Access Tokens](https://github.com/settings/tokens)

### 3. Run the App:

```bash
streamlit run app.py
```

### Access the App:

You can then access Sentimend by navigating to either provided URL in your browser of choice. Those URLs should be:

- Local URL: http://localhost:8501
- Network URL: http://172.28.175.103:8501

## Usage Guide

Please see the Appendix of `report/sentimend_report.pdf` for an in-depth usage guide complete with screenshots.

## Project Structure

```
streamlit_play_sentiment/

├─ .streamlit/
│ └─ secrets.toml                                   # Streamlit secrets (API keys)
├─ experiment_artifacts/                            # artifacts from Streamlit's evaluation
|  └─ ...
├─ presentation/
|  └─ Sentimend_Presentation_Graler_Hucker.pdf      # slides from in-class presentation
├─ report/
|  └─ Sentimend_Project_Report_Graler_Hucker.pdf    # report complete with description, evaluation, and usage guide
├─ services/
│ ├─ clustering.py                                  # ReviewClusterer (optional k-means clustering)
│ ├─ exporters.py                                   # Exporters for CSV, JSON, Markdown
│ ├─ identifier_finder.py                           # PackageIdentifierFinder (fuzzy search → appId)
│ ├─ models.py                                      # Data models (PackageCandidate, Review, AnalysisResult)
│ ├─ sentiment_analyzer.py                          # SentimentAnalyzer (fetch reviews, analyze)
│ ├─ solution_generator.py                          # SolutionGenerator (simple rule-based suggestions)
│ ├─ triage.py                                      # Cluster severity triage logic
│ └─ utils.py                                       # Utility functions
├─ views/                                           # Streamlit view components for Review Modal
├─ app.py                                           # Core Streamlit UI
├─ environment.yml                                  # conda environment configuration
└─ README.md                                        # this file
```

## Customization

### Swapping in your own sentiment package:

The `SentimentAnalyzer` uses VADER by default but accepts a custom analyzer function (`analyzer_fn: Callable[[str], float]`). To use your own model (e.g., a `fine-tuned BERT`):

Return a single compound-like score between -1 and 1 where:

- negative < -0.05
- neutral between -0.05 and 0.05
- positive > 0.05

Example:

```python
from services.sentiment_analyzer import SentimentAnalyzer

def my_bert_model(text: str) -> float:
    # return a float in [-1, 1]
    return 0.0

# Inject into the service
analyzer = SentimentAnalyzer(analyzer_fn=my_bert_model)
```
