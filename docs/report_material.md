# IndicBridge: Academic Report Material
**Course:** CSD358 Information Retrieval  
**Track:** T5 — Multilingual and Indic-language Search  
**Project Title:** INDICBRIDGE: Explainable Multilingual Search for Hindi, Hinglish and English  

---

## 1. Problem Statement and Track Relevance (Track T5)
In Indian internet and digital communications, users search using a complex mixture of scripts and languages:
1. Standard Hindi written in Devanagari script (e.g., `कल का मौसम`)
2. English written in Latin script (e.g., `weather tomorrow`)
3. Romanized Hindi written phonetically in Latin script (e.g., `kal ka mausam`)
4. Code-mixed Hinglish queries interweaving English and Hindi vocabulary (e.g., `kal ka weather kaisa hoga`)
5. Orthographic and transliteration spelling variations (e.g., `bhaarat` vs `bharat`, `krishan` vs `krishna`).

Standard classical Information Retrieval systems rely strictly on exact token matching. When a user submits `kal ka mausam`, an exact-token TF-IDF engine fails completely to retrieve relevant Devanagari documents containing `कल का मौसम` or English documents discussing weather, leading to devastating vocabulary mismatch and zero recall.

**IndicBridge** bridges this cross-script gap while adhering strictly to foundational, transparent classical Information Retrieval principles.

---

## 2. Core Classical IR Mathematical Formulation
IndicBridge is founded entirely on the classical Vector Space Model (VSM) and inverted index architectures:

### 2.1 Smoothed Inverse Document Frequency (IDF)
$$\text{idf}(t) = \ln\left(\frac{N + 1}{\text{df}(t) + 1}\right) + 1$$
Where:
- $N$ is the total number of documents in the collection (32 in benchmark corpus)
- $\text{df}(t)$ is the document frequency of term $t$.

### 2.2 Sublinear Term Frequency (TF)
$$\text{tf}(t, d) = \begin{cases} 1 + \ln(\text{count}(t, d)) & \text{if } \text{count}(t, d) > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 2.3 TF-IDF Weight and L2 Vector Normalization
$$\vec{w}_d = (w_{t_1, d}, w_{t_2, d}, \dots, w_{t_V, d})^T \quad \text{where } w_{t, d} = \text{tf}(t, d) \cdot \text{idf}(t)$$
$$\|\vec{w}_d\|_2 = \sqrt{\sum_{t \in d} w_{t, d}^2}, \quad \hat{w}_{t, d} = \frac{w_{t, d}}{\|\vec{w}_d\|_2}$$

### 2.4 Cosine Similarity
$$\text{cosine}(\vec{q}, \vec{d}) = \sum_{t \in q \cap d} \hat{w}_{t, q} \cdot \hat{w}_{t, d}$$

---

## 3. Retrieval Pipeline Architecture (11 Stages)
IndicBridge executes an 11-stage pipeline for every incoming query:

1. **Stage 1 (Query Preservation):** Original query is preserved intact for user auditing.
2. **Stage 2 (Language & Script Identification):** Detects script (`Devanagari`, `Latin`, `Mixed`) and classifies language (`hi`, `en`, `hi-Latn`).
3. **Stage 3 (Unicode Normalization & Nukta Folding):** Normalizes to NFC and folds diacritic nuktas (`क़ -> क`, `फ़ -> फ`, `ज़ -> ज`).
4. **Stage 4 (Cross-Script Transliteration):** Bidirectional expansion between Latin and Devanagari (`kal` ➔ `कल`, `mausam` ➔ `मौसम`).
5. **Stage 5 (Romanized Spelling Regularization):** Maps phonetic spelling variants (`bhaarat` ➔ `bharat`, `krishan` ➔ `krishna`).
6. **Stage 6 (Stop-Word Handling):** Prunes high-frequency non-informative stop words across Hindi, Romanized Hindi, and English.
7. **Stage 7 (Lightweight Suffix Stemming):** Applies Ramanathan & Rao Hindi inflectional stemming and English/Hinglish suffix reduction.
8. **Stage 8 (Inverted Index Candidate Retrieval):** Uses inverted index postings lists to identify candidate documents in sub-millisecond time.
9. **Stage 9 (Auxiliary Transliteration & IndicSoundex Matching):** Evaluates cross-script token overlap and adapted phonetic distance.
10. **Stage 10 (Transparent Score Fusion):**
$$\text{final\_score} = \alpha \cdot \text{cosine} + \beta \cdot \text{translit} + \gamma \cdot \text{phonetic} + \delta \cdot \text{exact}$$
*(Default weights: $\alpha=0.70, \beta=0.15, \gamma=0.10, \delta=0.05$)*
11. **Stage 11 (Explainable Ranking):** Generates human-readable rationale cards showing matched terms, high-IDF terms, and component weights.

---

## 4. Empirical Evaluation Results

Evaluated on 24 ground-truth queries across 6 linguistic categories:

### 4.1 Overall Macro-Average Benchmark Metrics
| Metric | Baseline (Exact TF-IDF) | IndicBridge (Multilingual) | Absolute Gain | Relative Gain (%) |
| :--- | :--- | :--- | :--- | :--- |
| **Precision@5** | 0.6111 | 0.6056 | -0.0055 | -0.9% |
| **Precision@10** | 0.4960 | 0.5087 | +0.0127 | **+2.6%** |
| **Recall@5** | 0.7222 | 0.7583 | +0.0361 | **+5.0%** |
| **Recall@10** | 0.7472 | 0.8195 | +0.0723 | **+9.7%** |
| **F1@5** | 0.6284 | 0.6521 | +0.0237 | **+3.8%** |
| **MRR** | 0.9792 | 1.0000 | +0.0208 | **+2.1%** |
| **MAP** | 0.7124 | 0.7827 | +0.0703 | **+9.9%** |

### 4.2 Per-Category Performance Breakdown
| Category | Queries | P@5 | R@5 | MRR | MAP |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Devanagari Hindi** | 4 | 0.5667 | 0.6667 | 1.0000 | 0.6459 |
| **B: English** | 4 | 0.8000 | 0.6667 | 1.0000 | 0.6667 |
| **C: Hinglish Code-Mixed**| 4 | 0.5500 | 0.9167 | 1.0000 | 0.9524 |
| **D: Romanized Hindi** | 4 | 0.6000 | 0.8334 | 1.0000 | 0.8334 |
| **E: Spelling Variants** | 4 | 0.5500 | 0.7500 | 1.0000 | 0.7917 |
| **F: Transliterated** | 4 | 0.5667 | 0.7167 | 1.0000 | 0.8060 |

---

## 5. Architectural Component Ablation Study
Empirically measuring each pipeline component:

| Configuration | Description | P@5 | R@5 | F1@5 | MRR | MAP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. TF-IDF Only** | Exact token Baseline | 0.6111 | 0.7222 | 0.6284 | 0.9792 | 0.7124 |
| **2. TF-IDF + Normalization** | Unicode, nukta folding, stem | 0.6292 | 0.7500 | 0.6641 | 1.0000 | 0.7662 |
| **3. TF-IDF + Transliteration** | Cross-script expansion | 0.6056 | 0.7583 | 0.6521 | 1.0000 | 0.7726 |
| **4. TF-IDF + Phonetic Match** | IndicSoundex + fuzzy matching | 0.6375 | 0.7583 | 0.6724 | 1.0000 | 0.7711 |
| **5. Full IndicBridge** | All components + fusion | 0.6056 | 0.7583 | 0.6521 | 1.0000 | **0.7827** |

---

## 6. Honest Limitations
1. **Corpus Scale:** Demo benchmark contains 32 curated documents; larger collections (such as full Hindi Wikipedia or FIRE) require disk-backed inverted indexes.
2. **Heuristic Phonetic Mapping:** IndicSoundex collapses consonant alternations but cannot resolve deep semantic differences without lexical context.
3. **Short Query Ambiguity:** Very short 1-word queries in Romanized script (e.g., `mat`) can be ambiguous between Hindi and English.

---

## 7. AI Use Declaration
AI coding assistance was used during development for scaffolding, test construction, and documentation drafting. The team tested, verified, and debugged all IR mathematics, index algorithms, and evaluation scripts. The team takes full responsibility for the implementation and reported findings.
