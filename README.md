# INDICBRIDGE: Explainable Multilingual Information Retrieval System

> **Course:** CSD358 — Information Retrieval (Mid-Semester Hackathon)  
> **Track:** T5 — Multilingual and Indic-Language Search  
> **Institution:** Shiv Nadar University  
> **Team Members:**  
> - Dev Agarwal (Roll No: 2410110118)  
> - Kavya Pratap Singh Chauhan (Roll No: 2410110167)  
> - Suryansh Siddhu (Roll No: 2410110623)  
> - Harsh Sumrav (Roll No: 2410110142)  

---

## 1. Project Overview

**IndicBridge** is an explainable classical Information Retrieval (IR) system designed to bridge search across **Hindi (Devanagari script)**, **English (Latin script)**, and **Romanized Hindi / Hinglish (Latin script)** without relying on opaque black-box neural embeddings.

In Indian multilingual web search, users predominantly formulate queries in Romanized script (e.g., `"kal ka mausam"`, `"bhaarat desh"`), while authoritative corpus documents exist in Devanagari Hindi (`"कल का मौसम"`) or formal English (`"weather tomorrow"`). Traditional exact-token inverted index matching fails completely due to script divergence and non-standardized phonetic spelling variations.

IndicBridge addresses this challenge with a transparent **11-stage pipeline** integrating:
1. Custom in-memory **Inverted Index** tracking document frequencies, term frequencies, and exact in-document token positions.
2. Textbook **TF-IDF Vector Space Model** using sublinear term frequency scaling and Euclidean L2 vector normalization.
3. Bidirectional **cross-script transliteration normalization** (Romanized Hindi ⇄ Devanagari).
4. Custom **IndicSoundex** phonetic clustering mapping Indic acoustic consonant equivalents.
5. Convex **multi-signal score fusion** delivering real-time mathematical explanations.

---

## 2. 11-Stage IR Pipeline Architecture
[ User Query: Hindi / Hinglish / English ]
                                  │
                     Stage 1: Script & Lang Detect
                                  │
                     Stage 2: Unicode & Nukta Folding
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
      Stage 3: Transliteration        Stage 4: Spelling Normalize
      (Roman ⇄ Devanagari)            (Double Vowels / Dict Map)
                  └───────────────┬───────────────┘
                                  │
                     Stage 5: Stopword Removal
                                  │
                     Stage 6: Inflectional Stemming
                                  │
               Stage 7: Inverted Index Postings Lookup
                                  │
               Stage 8: Sublinear TF-IDF Vector Space
                                  │
               Stage 9: Cosine Similarity Scoring
                                  │
               Stage 10: Multi-Signal Score Fusion
                                  │
               Stage 11: Top-K Heap-Based Ranking
                                  │
                    [ Final Transparent Results ]
                    
                    ### Mathematical Formulations
- **Smoothed Inverse Document Frequency (IDF):**
  $$\text{idf}(t) = \ln\left( \frac{N + 1}{\text{df}(t) + 1} \right) + 1.0$$
- **Sublinear Term Frequency (TF):**
  $$\text{tf}(t, d) = 1 + \ln(\text{count}(t, d)) \quad \text{for count} > 0$$
- **Euclidean L2 Normalization:**
  $$\Vert{}\vec{d}\Vert{}_2 = \sqrt{\sum_{t \in d} \left( \text{tf}(t, d) \cdot \text{idf}(t) \right)^2}, \quad \vec{w}_d(t) = \frac{\text{tf}(t, d) \cdot \text{idf}(t)}{\Vert{}\vec{d}\Vert{}_2}$$
- **Cosine Similarity Dot-Product:**
  $$\text{Cosine}(q, d) = \sum_{t \in q \cap d} \vec{w}_q(t) \cdot \vec{w}_d(t)$$
- **Convex Score Fusion:**
  $$\text{Score}_{\text{final}}(q, d) = \alpha \cdot \text{Cosine}(q, d) + \beta \cdot \text{Translit}(q, d) + \gamma \cdot \text{Phonetic}(q, d) + \delta \cdot \text{Exact}(q, d)$$
  *Default Weights:* $\alpha = 0.70$, $\beta = 0.15$, $\gamma = 0.10$, $\delta = 0.05$ (strictly normalized to $1.00$).

---

## 3. Dataset & Provenance

The benchmark corpus consists of 32 curated parallel documents stored in `data/documents.jsonl` and 24 labeled evaluation queries in `data/evaluation_queries.json`.

### Sources:
- **Constitution & Fundamental Rights:** National Portal of India (`india.gov.in`) and NCERT civics summaries.
- **Weather & Monsoon Forecasts:** India Meteorological Department (`mausam.imd.gov.in`).
- **Aerospace & Space Science:** ISRO lunar mission archives on Chandrayaan (`isro.gov.in`).
- **Culture, Heritage, and Festivals:** Ministry of Culture (`indiaculture.gov.in`) and open-domain Hindi/English Wikipedia summaries.

The evaluation suite models the **FIRE (Forum for Information Retrieval Evaluation)** shared tasks and the **MIRACL** (Hindi subset) benchmark format.

---

## 4. Setup and Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Node.js 18+ and npm

### 1. Install Dependencies
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install