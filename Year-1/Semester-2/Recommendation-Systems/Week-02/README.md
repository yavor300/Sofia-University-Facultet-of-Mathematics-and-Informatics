# Content-Based Recommender (Improved) - Technical Documentation

This document describes the implementation added under the notebook section:
`# My Content-Based Recommender (Improved)` in
`Content_based_Recommender_Example.ipynb`.

## Scope

The improved implementation keeps the original content-based approach but upgrades:
- data acquisition for local environments (no `gsutil` dependency),
- preprocessing robustness (modern `pandas` / `scikit-learn` APIs),
- user profile construction (weighted by interaction strength),
- evaluation methodology (ranking metrics on a user-level holdout split).

## Files and Runtime Context

- Notebook: `Content_based_Recommender_Example.ipynb`
- Dataset directory: `hw2-dataset/`
  - `shared_articles.csv`
  - `users_interactions.csv`
- Python environment: local `.venv` in the same folder.

## What Was Changed

## 1) Local Dataset Download (Notebook Compatibility)

The original notebook used:
- `!gsutil cp -r gs://rec-sys-fmi/hw2-dataset .`

This was replaced with a Python downloader (`urllib.request`) that:
- creates `hw2-dataset/` if missing,
- downloads both CSV files from `https://storage.googleapis.com/rec-sys-fmi/hw2-dataset`,
- skips files that already exist.

Reason: local environments usually do not have `gsutil` installed.

## 2) API Compatibility Fixes

Updated legacy calls to current APIs:
- `articles_df.lang.value_counts()` -> `articles_df["lang"].value_counts()`
- `get_feature_names()` -> `get_feature_names_out()`
- safer text input for TF-IDF with `fillna("")`

Reason: avoid runtime errors on modern `pandas` and `scikit-learn`.

## 3) New Improved Recommender Pipeline

Added a separate, end-to-end pipeline (does not depend on intermediate fragile notebook state).

### 3.1 Interaction Strength Mapping

Events are converted to numeric preference strength:

- `VIEW`: 1.0
- `LIKE`: 2.0
- `BOOKMARK`: 3.0
- `FOLLOW`: 4.0
- `COMMENT CREATED`: 5.0

### 3.2 Article Filtering and Text Construction

From `shared_articles.csv`, keep:
- `eventType == "CONTENT SHARED"`
- `lang == "en"`

Use features:
- `contentId`, `title`, `text`, `url`

Document text:
- `document = title + " " + text`

### 3.3 Interaction Cleaning

From `users_interactions.csv`, keep:
- `personId`, `contentId`, `eventType`, `timestamp`
- only events present in the interaction-strength map
- only interactions with content that exists in filtered article set

Dedup logic:
- group by `["personId", "contentId"]`
- aggregate:
  - `eventStrength = max(eventStrength)`
  - `timestamp = max(timestamp)`

Cold-start reduction:
- keep users with at least `MIN_INTERACTIONS_PER_USER = 5`.

### 3.4 Train/Test Strategy

Temporal holdout per user:
- sort by `timestamp`,
- test = each user’s latest interaction,
- train = all other interactions.

This better matches recommendation usage compared to random global split.

### 3.5 Item Representation (TF-IDF)

`TfidfVectorizer` configuration:
- `analyzer="word"`
- `ngram_range=(1, 2)`
- `min_df=2`
- `max_df=0.8`
- `max_features=5000`
- English stopwords from NLTK

Result:
- sparse item-feature matrix `item_tfidf`
- `content_id -> matrix row index` mapping for fast lookups.

### 3.6 User Profile Construction

For each user:
- collect TF-IDF vectors of items seen in train,
- compute weighted average using `eventStrength`:

`profile_u = sum(strength_i * tfidf_i) / sum(strength_i)`

Also keep a per-user set of seen `contentId` to filter already-consumed items.

### 3.7 Recommendation Function

`recommend_for_user(person_id, top_n=10)`:
- cosine similarity between user profile and all item vectors,
- rank by descending similarity,
- exclude items seen in training,
- return top-N with `contentId`, `score`, `title`, `url`.

### 3.8 Evaluation Metrics

Implemented `evaluate_recommender(test_df, k=10)` with:
- `Recall@5`
- `Recall@10`
- `MRR@10`

For each user:
- relevant set = test items for that user (in this pipeline: one latest item),
- compare with predicted top-k list.

## Observed Run Output

A validation run in the same environment produced:
- `Recall@5 = 0.016560509554140127`
- `Recall@10 = 0.035668789808917196`
- `MRR@10 = 0.011260742088767567`

These values are baseline-quality for a pure content model with simple profile weighting.

## How to Reproduce

From folder `Week-02`:

1. Activate environment:
   - `source .venv/bin/activate`
2. Open and run notebook:
   - `Content_based_Recommender_Example.ipynb`
3. Execute cells in order, especially from:
   - `# My Content-Based Recommender (Improved)`
4. Confirm metrics dictionary cell output.

## Complexity Notes

Let:
- `U` = users with profiles,
- `I` = number of items,
- `F` = TF-IDF features.

Main cost centers:
- TF-IDF fitting: approx `O(total_tokens)` with sparse construction.
- Per-user scoring: cosine similarity over all items (`U` times over `I x F` sparse matrix).

For larger datasets, precomputing candidate pools or approximate nearest-neighbor search would be recommended.

## Known Limitations

- Purely content-based; no collaborative signal.
- No popularity/recency blending.
- One-item-per-user test may be high variance.
- No hyperparameter sweep for TF-IDF configuration.

## Next Technical Iterations (Optional)

- Add hybrid scoring: content similarity + item popularity prior.
- Use more robust temporal evaluation (multiple future interactions per user).
- Normalize interaction strengths with log scaling.
- Add `Precision@K`, `NDCG@K`, and coverage/diversity metrics.
