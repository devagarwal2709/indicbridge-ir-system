# INDICBRIDGE: Explainable Multilingual Information Retrieval System

> **Course:** CSD358 — Information Retrieval (Mid-Semester Hackathon)  
> **Track:** T5 — Multilingual and Indic-Language Search  
> **Institution:** Shiv Nadar University  
> **Team Members:**
>
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

```text
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
```

### Mathematical Formulations

- **Smoothed Inverse Document Frequency (IDF):**

  IDF(t) = ln((N + 1) / (df(t) + 1)) + 1.0

- **Sublinear Term Frequency (TF):**

  TF(t, d) = 1 + ln(count(t, d)), for count > 0

- **Euclidean L2 Normalization:**

  ‖d‖₂ = √[Σₜ∈d (TF(t, d) × IDF(t))²]

  Document weights are normalized as:

  w_d(t) = (TF(t, d) × IDF(t)) / ‖d‖₂

- **Cosine Similarity Dot-Product:**

  Cosine(q, d) = Σₜ∈q∩d w_q(t) × w_d(t)

- **Convex Score Fusion:**

  Score_final(q, d) = α × Cosine(q, d)
  + β × Translit(q, d)
  + γ × Phonetic(q, d)
  + δ × Exact(q, d)

  **Default Weights:**

  α = 0.70, β = 0.15, γ = 0.10, δ = 0.05

  The weights are strictly normalized to:

  α + β + γ + δ = 1.00
---

## 3. Dataset & Provenance

The benchmark corpus consists of **32 curated parallel documents** stored in `data/documents.jsonl` and **24 labeled evaluation queries** in `data/evaluation_queries.json`.

### Sources

- **Constitution & Fundamental Rights:** National Portal of India (`india.gov.in`) and NCERT civics summaries.
- **Weather & Monsoon Forecasts:** India Meteorological Department (`mausam.imd.gov.in`).
- **Aerospace & Space Science:** ISRO lunar mission archives on Chandrayaan (`isro.gov.in`).
- **Culture, Heritage, and Festivals:** Ministry of Culture (`indiaculture.gov.in`) and open-domain Hindi/English Wikipedia summaries.

The evaluation suite models the **FIRE (Forum for Information Retrieval Evaluation)** shared tasks and the **MIRACL** Hindi subset benchmark format.

---

## 4. Setup and Installation

### Prerequisites

- Python 3.10+ (tested on Python 3.13)
- Node.js 18+
- npm

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install Node dependencies
npm install
```

### 2. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

On Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

Update `.env` with any required local configuration.

### 3. Run the Backend

```bash
python app.py
```

### 4. Run the Frontend / Development Server

```bash
npm run dev
```

The exact frontend/backend commands may depend on the configuration in `package.json`.

---

## 5. Core System Architecture

IndicBridge combines classical Information Retrieval with multilingual normalization and explainable ranking.

### 5.1 Inverted Index

The system uses a custom in-memory `InvertedIndex`.

Each term maps to a postings list containing:

- Document ID
- Term frequency
- Exact token positions

Conceptually:

$$
\mathrm{Postings}(t)
=
[
\langle
\mathrm{doc\_id},
\mathrm{tf},
[p_1,p_2,\dots,p_n]
\rangle
]
$$

The positional information allows the system to expose exactly where a query term occurs inside a document.

### 5.2 TF-IDF Vector Space Model

Documents and queries are represented as weighted vectors.

The implementation uses:

- Smoothed IDF
- Sublinear TF
- L2 vector normalization
- Cosine similarity

This provides a classical and transparent lexical ranking mechanism.

### 5.3 Unicode and Nukta Normalization

Devanagari text is normalized before indexing and searching.

For example, compatible Unicode representations and nukta variants are folded into canonical forms to prevent visually equivalent characters from being treated as unrelated tokens.

### 5.4 Stopword Removal

Multilingual stopword lists are applied to remove low-information grammatical terms.

Examples include:

**Hindi:**

```text
का, के, में
```

**Hinglish:**

```text
ka, ke, hai
```

**English:**

```text
the, is, and
```

### 5.5 Stemming

The system applies inflectional stemming to reduce morphological variation.

Hindi processing uses the Ramanathan & Rao stemming approach, while English and Hinglish use standard suffix stripping.

---

## 6. Multilingual Search and Transliteration

### 6.1 Cross-Script Transliteration

Romanized Hindi queries can be mapped into Devanagari.

For example:

```text
kal ka mausam
```

is mapped to:

```text
कल का मौसम
```

This allows a Romanized query to retrieve documents written in native Devanagari Hindi.

### 6.2 Bidirectional Dual-Representation Indexing

Documents are indexed using both their native representation and their cross-script transliteration equivalents during index construction.

This enables cross-script candidate retrieval without requiring runtime external translation APIs.

The approach also keeps the search process deterministic and inspectable.

---

## 7. IndicSoundex Phonetic Matching

Standard English Soundex is not designed for Indic phonetic characteristics.

IndicBridge therefore introduces **IndicSoundex**, a custom phonetic matching mechanism designed around Indic consonant groups.

The system collapses acoustically similar spellings, including:

```text
bh ↔ b
kh ↔ k
sh ↔ s
```

This allows spelling variants such as:

```text
krishan
krishna
```

to map to the same phonetic key:

```text
K4320
```

This is particularly useful for Romanized Hindi, where users do not follow a standardized spelling convention.

---

## 8. Explainable Multi-Signal Score Fusion

IndicBridge combines multiple relevance signals:

1. Lexical cosine similarity
2. Transliteration similarity
3. Phonetic similarity
4. Exact-match score

The final score is:

$$
\mathrm{Score}_{\mathrm{final}}(q,d)
=
\alpha\cdot\mathrm{Cosine}(q,d)
+
\beta\cdot\mathrm{Translit}(q,d)
+
\gamma\cdot\mathrm{Phonetic}(q,d)
+
\delta\cdot\mathrm{Exact}(q,d)
$$

Default parameters are:

```text
α = 0.70
β = 0.15
γ = 0.10
δ = 0.05
```

with:

$$
\alpha+\beta+\gamma+\delta=1.00
$$

This bounded convex formulation ensures that the ranking remains controlled while allowing the relative importance of each signal to be tuned.

---

## 9. Novelty and Creativity

### 9.1 Acoustically Adapted IndicSoundex

Replaces standard English Soundex with 7 Indic-specific consonant clusters that collapse aspirated pairs ($bh \leftrightarrow b$, $kh \leftrightarrow k$) and sibilants ($sh \leftrightarrow s$), correctly matching spelling variants like *"krishan"* and *"krishna"* to key `K4320`.

### 9.2 Bidirectional Dual-Representation Indexing

Indexes documents with both native text and their cross-script transliteration equivalents during index construction, enabling instant sub-millisecond candidate retrieval without runtime translation APIs.

### 9.3 Bounded Convex Score Fusion

Dynamically combines lexical cosine, transliteration, phonetic, and exact-match scores under the strict constraint:

$$
\alpha+\beta+\gamma+\delta=1.0
$$

This prevents any single signal from dominating the ranking while allowing real-time parameter tuning.

### 9.4 White-Box Vector Explainability

Unlike opaque dense neural models, every ranked result decomposes the exact inner-product arithmetic:

$$
\vec{w}_q\cdot\vec{w}_d
$$

This allows the system to isolate high-IDF discriminators and postings positions directly on screen.

---

## 10. Explainability

A major design objective of IndicBridge is that the ranking process should be inspectable rather than treated as a black box.

For each retrieved result, the system can expose:

- Query terms that matched
- Document terms that contributed to the score
- TF values
- IDF values
- Vector weights
- Cosine similarity contributions
- Transliteration matches
- Phonetic matches
- Exact-match contributions
- Final fused score
- Relevant postings positions

This provides a direct connection between the underlying IR mathematics and the final search ranking.

---

## 11. Search Examples

### Example 1: English Query

```text
weather tomorrow
```

The system can retrieve relevant Hindi or English documents containing equivalent concepts.

### Example 2: Romanized Hindi Query

```text
kal ka mausam
```

The query can be transliterated into:

```text
कल का मौसम
```

allowing retrieval from Devanagari documents.

### Example 3: Spelling Variation

```text
krishan ji
```

The phonetic layer allows:

```text
krishan
```

and:

```text
krishna
```

to be treated as acoustically related forms.

---

## 12. Evaluation

IndicBridge was evaluated against an **Exact-Token TF-IDF Baseline** across **24 judged ground-truth queries** covering six categories:

- Devanagari Hindi
- English
- Hinglish
- Romanized Hindi
- Spelling Variants
- Cross-Script Queries

### Macro-Average Results

| Metric | Baseline (Exact TF-IDF) | IndicBridge (Multilingual) | Absolute Gain | Relative Gain (%) |
| --- | ---: | ---: | ---: | ---: |
| Precision@5 | 0.6111 | 0.5931 | -0.0180 | -2.95% |
| Precision@10 | 0.4960 | 0.5349 | +0.0389 | +7.84% |
| Recall@5 | 0.7222 | 0.7306 | +0.0084 | +1.16% |
| Recall@10 | 0.7472 | 0.8195 | +0.0723 | +9.68% |
| F1@5 | 0.6284 | 0.6342 | +0.0058 | +0.92% |
| MRR | 0.9792 | 1.0000 | +0.0208 | +2.12% |
| MAP | 0.7124 | 0.7650 | +0.0526 | +7.38% |

### Precision vs Recall Trade-Off

A key finding is the small trade-off between Precision@5 and Recall@10.

IndicBridge achieves:

- **+9.68% Recall@10**
- **+7.38% MAP**
- **1.0000 MRR**

while Precision@5 decreases slightly by **-2.95%**.

The recall improvement occurs because cross-script expansion broadens the candidate pool. This allows relevant documents written in different scripts to be retrieved, although it can occasionally introduce partial matches at lower ranks.

---

## 13. Architectural Ablation Study

| Config | Architecture Description | P@5 | R@5 | MRR | MAP |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | Baseline (Exact-token TF-IDF, no preprocessing) | 0.6111 | 0.7222 | 0.9792 | 0.7124 |
| 2 | TF-IDF + Unicode / Nukta Normalization + Stemming | 0.6292 | 0.7500 | 1.0000 | 0.7662 |
| 3 | Config 2 + Transliteration Expansion | 0.6056 | 0.7583 | 1.0000 | 0.7726 |
| 4 | Config 2 + IndicSoundex Phonetic Matching | 0.6375 | 0.7583 | 1.0000 | 0.7711 |
| 5 | **Full IndicBridge (All 11 Stages + Fused Scoring)** | **0.6056** | **0.7583** | **1.0000** | **0.7827** |

**Takeaway:** Normalization and transliteration provide the single largest boost to Recall and MAP, demonstrating the importance of bridging script divergence in multilingual Indian search.

---

## 14. Project Structure

```text
indicbridge-ir-system/
│
├── Screenshots/
│
├── config/
│   └── config.yaml
│
├── data/
│   ├── README.md
│   ├── documents.jsonl
│   └── evaluation_queries.json
│
├── docs/
│   ├── demo_script.md
│   ├── report_material.md
│   └── rubric_mapping.md
│
├── evaluation/
│   ├── ablation.py
│   ├── evaluate.py
│   └── results/
│       ├── ablation_results.csv
│       └── evaluation_results.json
│
├── scripts/
│
├── src/
│   ├── index/
│   ├── preprocessing/
│   └── ...
│
├── tests/
│
├── app.py
├── indicbridge.py
├── server.ts
├── index.html
├── metadata.json
├── package.json
├── package-lock.json
├── requirements.txt
├── pytest.ini
├── run_tests.py
├── tsconfig.json
├── vite.config.ts
├── .env.example
├── .gitignore
├── LICENSE
└── README.md
```

---

## 15. Testing

The project includes an automated test suite covering the major components of the system.

Run:

```bash
python run_tests.py
```

or, if using pytest directly:

```bash
pytest
```

The tests cover areas such as:

- Text normalization
- Transliteration
- Phonetic matching
- Inverted indexing
- TF-IDF calculations
- Search ranking
- Evaluation metrics

---

## 16. Performance and Design Goals

The prototype is intentionally designed as a transparent in-memory IR system.

The current benchmark contains **32 curated documents**, allowing:

- Fast query processing
- Direct postings inspection
- Easy debugging
- Transparent score calculations
- Real-time explanation generation

The architecture is designed to be extended to significantly larger corpora in the semester-project phase.

---

## 17. Current Limitations

### 17.1 Corpus Size

The current hackathon prototype indexes 32 curated documents in memory. This keeps the system lightweight and allows direct inspection of postings and ranking calculations.

### 17.2 Slang and Out-of-Vocabulary Terms

Unseen colloquial words may not exist in the transliteration dictionary and therefore rely on phonetic approximations.

### 17.3 Transliteration Ambiguity

Romanized Hindi does not have a standardized spelling system. A single Romanized form can sometimes correspond to multiple Devanagari representations.

### 17.4 Classical Ranking

The current system intentionally prioritizes classical IR techniques rather than dense neural retrieval. This improves explainability but may limit semantic matching for queries with little lexical overlap.

---

## 18. Future Roadmap

### 18.1 Scale to Larger Indic Datasets

Replace the current in-memory index with disk-backed **SPIMI (Single-Pass In-Memory Indexing)** to support substantially larger collections.

Potential datasets include:

- AI4Bharat IndicCorp
- MIRACL Hindi collection

### 18.2 Subword BPE Indexing

Integrate Byte-Pair Encoding (BPE) based subword tokenization to better handle:

- Morphological compounds
- Informal spelling
- Slang
- Out-of-vocabulary terms

### 18.3 Dense-Sparse Hybrid Retrieval

Combine the current classical inverted-index approach with dense multilingual transformer representations.

A future rank fusion layer can use **Reciprocal Rank Fusion (RRF)** to combine sparse lexical retrieval with dense semantic retrieval.

---

## 19. Work Division

All four team members contributed equally (25% each) across research, engineering, and evaluation.

### Dev Agarwal

**Core Inverted Index Architecture & Mathematical Foundations**

Implemented:

- `InvertedIndex`
- `PostingEntry`
- Positional indexing
- Sublinear TF-IDF
- L2 vector normalization
- Cosine dot-product calculation

Primary directory:

```text
src/index/
```

### Kavya Pratap Singh Chauhan

**Linguistic Preprocessing & IndicSoundex**

Developed:

- Unicode NFC normalization
- Devanagari nukta folding
- Hindi/Hinglish stopword lists
- 7-group IndicSoundex phonetic algorithm

Primary directory:

```text
src/preprocessing/
```

### Suryansh Siddhu

**Benchmark Evaluation Suite & Ablation Engine**

Responsible for:

- 24 ground-truth queries
- Precision@k
- Recall@k
- MRR
- MAP
- 5-configuration ablation study

Primary directory:

```text
evaluation/
```

### Harsh Sumrav

**System Integration, Interface & CLI Runner**

Engineered:

- Express/Vite bridge
- `server.ts`
- Unified CLI tool
- `indicbridge.py`
- Automated test suite
- Final documentation formatting and overall architecture

---

## 20. AI-Use Declaration

In compliance with CSD358 academic integrity guidelines:

**Google Gemini & OpenAI ChatGPT** were used as technical thought partners and coding assistants for:

- React/Tailwind frontend component boilerplate
- Debugging Windows UTF-8 (`cp1252`) subprocess piping in Node/Express
- Refining regular expressions for Devanagari Unicode codepoints

### Team Ownership

All core Information Retrieval implementations, including:

- Inverted index data structures
- Smoothed textbook IDF mathematics
- Vector space projections
- Score fusion weights
- Evaluation metrics

were designed, validated, and benchmarked directly by the team members.

---

## 21. References

1. Christopher D. Manning, Prabhakar Raghavan, and Hinrich Schütze. *Introduction to Information Retrieval*. Cambridge University Press, 2008.

2. FIRE (Forum for Information Retrieval Evaluation): Shared-Task Track on Cross-Lingual and Code-Mixed Indic Retrieval.

3. MIRACL Dataset: Multi-lingual Information Retrieval Across a Continuum of Languages — Hindi Subset.

4. AI4Bharat IndicCorp: Large-scale open corpora for Indian language processing.

5. Government Knowledge Portals:
   - National Portal of India
   - Indian Meteorological Department
   - Indian Space Research Organisation
   - Ministry of Culture

6. Open Domain Hindi & English Wikipedia: Wikipedia contributors, Wikimedia Foundation.

---

## 22. License

This project is distributed under the MIT License.

See the [LICENSE](LICENSE) file for details.

---

## 23. Repository Contents

The repository contains:

- Complete IndicBridge source code
- Multilingual benchmark dataset
- Evaluation queries
- Automated tests
- Ablation study implementation
- Evaluation results
- Documentation
- Hackathon project report
- Screenshots and demonstration material

---

**IndicBridge — Making multilingual Indian search transparent, explainable, and script-aware.**
