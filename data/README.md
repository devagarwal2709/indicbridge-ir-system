# IndicBridge Benchmark Corpus

## Dataset Overview
- **Label**: Demo corpus for local demonstration & benchmark evaluation
- **Course**: CSD358 Information Retrieval (Academic Mid-Sem Hackathon)
- **Track**: T5 — Multilingual and Indic-language Search
- **Format**: JSON Lines (`data/documents.jsonl`)

## Document Collection Statistics
- **Total Documents**: 32 curated documents
- **Languages / Scripts**:
  - Hindi in Devanagari script (`hi`): 14 documents
  - English in Latin script (`en`): 12 documents
  - Hinglish / Code-mixed (`hi-Latn` / `en-hi`): 6 documents
- **Domains Covered**:
  1. Indian Constitution, Law, and Fundamental Rights
  2. Weather, Climate, and Forecasting (Delhi, Monsoon, Himalayas)
  3. Space Exploration (ISRO, Chandrayaan, Mangalyaan)
  4. Culture, History, and Festivals (Diwali, Holi, Yoga, Heritage)
  5. Education, Universities, and Literacy
  6. Technology, Digital India, and Internet Access

## Schema
Each line in `data/documents.jsonl` contains a JSON object:
```json
{
  "id": "doc_001",
  "title": "भारत का संविधान और मौलिक अधिकार",
  "language": "hi",
  "text": "...",
  "source": "Academic Indian Civics Corpus (Open Public Records / NCERT adapted)",
  "url": "https://india.gov.in/my-government/constitution"
}
```

## Preprocessing & Curation Ethics
- Content is adapted from public domain educational summaries, NCERT open educational resources, government portal summaries, and open Wikipedia knowledge entries.
- No personal data, confidential material, or copyrighted private text is included.
- All documents are verified for genuine cross-script retrieval challenges (e.g. Hindi document on weather, English document on weather, and code-mixed document on weather).

## Limitations
- This corpus consists of 32 demonstrative documents designed to be fast, deterministically reproducible, and fully inspectable in memory without requiring external database setups.
- While it accurately evaluates cross-script IR challenges, full-scale production deployment should scale up to larger corpora (e.g., MIRACL Hindi subset, FIRE collections, or AI4Bharat IndicCorp) by pointing `config/config.yaml` to the expanded `.jsonl` path.
