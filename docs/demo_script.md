# IndicBridge: 5-8 Minute Live Video Demonstration Script
**Course:** CSD358 Information Retrieval  
**Track:** T5 — Multilingual and Indic-language Search  
**Rule:** The video demo must demonstrate the *live working system* and code, not static slides.

---

### [0:00 - 0:45] Problem Statement and Track 5 Context
- **Screen:** Show IndicBridge web workbench or terminal running `streamlit run app.py`.
- **Narration:**
  > "Hello, in this video we present **IndicBridge**, an explainable multilingual Information Retrieval system designed for Track T5 of CSD358 Information Retrieval.
  > In Indian digital search, users rarely use pure monolingual queries. They search using Devanagari Hindi, English, phonetic Romanized Hindi like *kal ka mausam*, and code-mixed Hinglish.
  > Standard TF-IDF search engines fail on these queries because of vocabulary mismatch across scripts. IndicBridge bridges this gap using classical IR principles, cross-script transliteration, adapted phonetic matching, and explainable score fusion."

---

### [0:45 - 2:00] System Architecture and IR Pipeline Walkthrough
- **Screen:** Show the "IR Pipeline Execution" diagnostic box on the Search Workbench.
- **Narration:**
  > "IndicBridge uses an 11-stage classical IR pipeline:
  > 1. Preservation of original query.
  > 2. Unicode normalization and nukta folding.
  > 3. Language and script identification.
  > 4. Cross-script transliteration between Devanagari and Latin.
  > 5. Romanized Hindi spelling normalization (mapping variants like *bhaarat* to *bharat*).
  > 6. Stop-word removal across three language dictionaries.
  > 7. Suffix stripping stemmers.
  > 8. Candidate document retrieval over our classical Inverted Index.
  > 9. Smoothed TF-IDF cosine similarity computation.
  > 10. Auxiliary transliteration and IndicSoundex phonetic scoring.
  > 11. Transparent score fusion: $\text{final} = \alpha \cdot \text{cosine} + \beta \cdot \text{translit} + \gamma \cdot \text{phonetic} + \delta \cdot \text{exact}$."

---

### [2:00 - 4:00] Live Search Demonstrations (4 Real Query Types)
- **Demo 1: Romanized Hindi Query**
  - **Query:** `kal ka mausam`
  - **Action:** Click Search.
  - **Observation:** Show that it retrieves doc_005 (Hinglish weather update) at Rank 1 AND doc_003 (Devanagari Hindi weather forecast) at Rank 2.
  - **Narration:**
    > "Notice that although we searched in Latin characters (*kal ka mausam*), the system transliterated and normalized the query to match both Latin and Devanagari documents!"

- **Demo 2: Hindi Devanagari Query**
  - **Query:** `कल का मौसम`
  - **Action:** Click Search.
  - **Observation:** High cosine similarity on doc_003 and doc_005.

- **Demo 3: English Query**
  - **Query:** `weather tomorrow`
  - **Action:** Click Search.
  - **Observation:** Matches doc_004 (English weather forecast) and doc_005.

- **Demo 4: Spelling Variation Query**
  - **Query:** `bhaarat desh` or `krishan ji`
  - **Action:** Click Search.
  - **Observation:** Show that *bhaarat* is canonicalized to *bharat* and IndicSoundex key matches *bharat* documents with high relevance.

---

### [4:00 - 5:00] Explainable IR Details & Inverted Index Postings
- **Screen:**
  1. Expand "Show IR Vector Details" on a search result card.
  2. Switch to the **IR Postings & Diagnostics** tab.
- **Narration:**
  > "IndicBridge is fully explainable. Every result clearly states *why* it was retrieved. Under IR details, we display the exact TF-IDF term dot-product contributions $w_{t,q} \cdot w_{t,d}$.
  > Now look at our **Inverted Index Diagnostics**. Let's enter the term `bharat`. We see its Document Frequency is 8, its collection size is 32, and the exact smoothed IDF is $\ln((32+1)/(8+1)) + 1 = 2.2993$.
  > We can directly inspect the postings list showing document IDs, Term Frequencies, and exact in-document token positions."

---

### [5:00 - 6:00] Baseline vs. Improved System Head-to-Head Comparison
- **Screen:** Click on **Baseline Comparison** tab. Run query `mera bharat`.
- **Narration:**
  > "Here is our mandatory Baseline comparison. The left column shows the exact TF-IDF Baseline. The right column shows IndicBridge.
  > On `mera bharat`, Baseline only retrieves documents with exact token matches. IndicBridge retrieves additional relevant documents from the Devanagari Hindi corpus (*मेरा भारत देश*).
  > As shown in the green banner, IndicBridge achieved an empirical retrieval delta of additional relevant documents that Baseline missed."

---

### [6:00 - 7:00] Empirical Benchmark Metrics & Ablation Study
- **Screen:** Switch to **Evaluation & Ablation** tab.
- **Narration:**
  > "We evaluated both systems across 24 ground-truth queries across 6 linguistic categories:
  > - Recall@10 increased by **+9.7%** (0.7472 to 0.8195).
  > - Mean Average Precision (MAP) increased by **+9.9%** (0.7124 to 0.7827).
  > - Mean Reciprocal Rank (MRR) achieved **1.0000**.
  > Below, our 5-stage ablation study isolates the contribution of each module: from pure TF-IDF, to normalization, to transliteration, to phonetic matching, up to full IndicBridge."

---

### [7:00 - 8:00] Honest Limitations and Future Work
- **Screen:** Show summary / terminal running `pytest` (showing 17/17 tests passing).
- **Narration:**
  > "Finally, we report our limitations:
  > 1. Our benchmark corpus consists of 32 demonstrative documents.
  > 2. IndicSoundex is a phonetic heuristic and does not perform deep semantic parsing.
  > In future work, we plan to extend this architecture to Bengali and Tamil, and integrate dense learned vector representations for hybrid sparse-dense ranking.
  > Thank you!"
