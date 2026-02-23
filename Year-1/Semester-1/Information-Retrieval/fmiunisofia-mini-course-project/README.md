# Mini Search Engine – Information Retrieval Project

## Overview

This project implements a **simple but extensible search engine** as part of the *Information Retrieval* mini-course assignment.  
The system indexes a collection of raw news articles stored as `.txt` files and provides multiple retrieval functionalities through a web-based interface.

The application is built on top of **Elasticsearch** and demonstrates core Information Retrieval concepts such as indexing, ranking, fuzzy matching, phrase search, multilingual retrieval, and document similarity.

---

## Key Features

### 1. Full-Text Search
- Keyword-based search over a corpus of news articles
- Uses Elasticsearch’s default **BM25** ranking model
- Supports relevance ranking and snippet highlighting

---

### 2. Multilingual Retrieval (English & Russian)
- Documents are indexed with **language-specific fields**:
  - `title_en`, `body_en` (English analyzer)
  - `title_ru`, `body_ru` (Russian analyzer)
- Users can select:
  - **English only**
  - **Russian only**
  - **All languages**
- Language-aware analyzers improve stemming, tokenization, and relevance scoring

---

### 3. Fuzzy Search (Typo Tolerance)
- Optional **fuzzy matching** to handle spelling mistakes
- Implemented using Elasticsearch’s built-in fuzzy term matching
- Based on **Levenshtein edit distance**
- Example:
  - `goverment` → `government`
- Activated via a UI checkbox and applied at query time

---

### 4. Exact Phrase Search
- Optional **exact phrase matching**
- Uses positional constraints (`match_phrase`)
- Improves precision by requiring terms to appear consecutively and in order
- Example:
  - `"election fraud"` matches only documents containing the exact phrase

---

### 5. Document Similarity – “More Like This”
- Content-based recommendation of similar articles
- Implemented using Elasticsearch’s **More Like This (MLT)** query
- The system:
  1. Extracts representative terms from a selected document
  2. Builds an internal query based on term statistics
  3. Retrieves and ranks similar documents using BM25
- Demonstrates document-to-document similarity (not machine learning)

---

### 6. Highlighted Search Results
- Query terms are highlighted in document snippets
- Helps users understand why a document was retrieved
- Language-aware highlighting (English / Russian)

---

### 7. Web-Based User Interface
- Implemented with **Flask**
- Supports:
  - Query input
  - Language selection
  - Fuzzy search toggle
  - Exact phrase toggle
  - “More like this” recommendations
- Clean and minimal UI designed for demonstration purposes

---

## Index Design

The Elasticsearch index uses a **single index with language-specific fields**.  
Each document contains:

- **Metadata**
  - `id` – unique document identifier
  - `path` – original file path
  - `language` – document language (`EN` / `RU`)
- **Content fields**
  - `title_en`, `body_en`
  - `title_ru`, `body_ru`

Language-specific analyzers ensure accurate tokenization and stemming while allowing unified querying across languages.

---

## Technologies Used

- **Python 3**
- **Elasticsearch 8.x**
- **Flask**
- **Docker** (for local Elasticsearch instance)
