"""
INDICBRIDGE: Explainable Multilingual Search for Hindi, Hinglish and English
CSD358 Information Retrieval Mid-Sem Academic Project (Track T5)
"""

import os
import sys
import time
import json
import pandas as pd
import streamlit as st

# Add repository root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils.config import load_config
from src.index.inverted_index import InvertedIndex
from src.index.tfidf_index import TFIDFIndex
from src.retrieval.baseline import BaselineRetriever
from src.retrieval.hybrid_retriever import IndicBridgeRetriever
from src.evaluation.evaluator import BenchmarkEvaluator

st.set_page_config(
    page_title="IndicBridge: Explainable Multilingual IR",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Academic Clarity
st.markdown("""
<style>
    .metric-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-hi { background-color: #fee2e2; color: #991b1b; }
    .badge-en { background-color: #dbeafe; color: #1e40af; }
    .badge-latn { background-color: #fef3c7; color: #92400e; }
    .pipeline-step {
        border-left: 3px solid #3b82f6;
        padding-left: 10px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading Inverted Index and Vector Models...")
def get_search_engines():
    config = load_config()
    cache_dir = config["paths"]["index_dir"]
    index_file = os.path.join(cache_dir, "inverted_index.json")

    # If index does not exist, build it
    if not os.path.exists(index_file):
        from scripts.build_index import build_and_save_index
        index, tfidf = build_and_save_index()
    else:
        index = InvertedIndex()
        index.load_index(cache_dir)
        tfidf = TFIDFIndex(index, use_sublinear_tf=config["index"]["use_sublinear_tf"])

    baseline = BaselineRetriever(index, tfidf)
    indicbridge = IndicBridgeRetriever(index, tfidf, config)
    return config, index, tfidf, baseline, indicbridge


config, index, tfidf, baseline, indicbridge = get_search_engines()

# SIDEBAR: Configuration & Pipeline Info
with st.sidebar:
    st.title("🇮🇳 INDICBRIDGE")
    st.caption("Explainable Multilingual IR for Hindi, Hinglish & English")
    st.markdown("---")

    mode = st.radio(
        "Navigation",
        ["🔍 Multilingual Search", "⚖️ Baseline Comparison", "🔬 IR Diagnostics & Postings", "📊 Benchmark Evaluation"],
        index=0
    )

    st.markdown("---")
    st.subheader("⚙️ Score Fusion Weights")
    st.caption("final_score = α·cosine + β·translit + γ·phonetic + δ·exact")

    alpha = st.slider("α (TF-IDF Cosine)", 0.0, 1.0, float(config["retrieval"]["alpha"]), 0.05)
    beta = st.slider("β (Transliteration)", 0.0, 1.0, float(config["retrieval"]["beta"]), 0.05)
    gamma = st.slider("γ (Phonetic / Fuzzy)", 0.0, 1.0, float(config["retrieval"]["gamma"]), 0.05)
    delta = st.slider("δ (Exact Match)", 0.0, 1.0, float(config["retrieval"]["delta"]), 0.05)
    indicbridge.set_weights(alpha, beta, gamma, delta)

    top_k = st.slider("Top-K Results", 1, 20, 10)

    st.markdown("---")
    st.caption(f"**Index Statistics:**\n- Documents: {index.num_docs}\n- Vocabulary Size: {len(index.get_vocabulary())}")


# =========================================================================
# TAB 1: MULTILINGUAL SEARCH
# =========================================================================
if mode == "🔍 Multilingual Search":
    st.header("Cross-Script & Cross-Variant Information Retrieval")
    st.write(
        "Search across Hindi (Devanagari), English, and Romanized Hindi (Hinglish) with full "
        "step-by-step explainability."
    )

    # Preset query chips
    preset_cols = st.columns(6)
    preset_query = None
    presets = [
        ("kal ka mausam", "Romanized Hindi"),
        ("कल का मौसम", "Hindi Devanagari"),
        ("weather tomorrow", "English"),
        ("mera bharat mahan", "Romanized Hindi"),
        ("bhaarat desh", "Spelling Variant"),
        ("chandrayaan mission", "Code-Mixed"),
    ]

    for col, (q_text, label) in zip(preset_cols, presets):
        if col.button(f"{q_text}\n({label})", key=f"btn_{q_text}"):
            preset_query = q_text

    query_input = st.text_input(
        "Enter Query (Hindi, Hinglish, English, or Romanized Hindi):",
        value=preset_query if preset_query else "kal ka mausam",
        key="search_input"
    )

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        run_search = st.button("🚀 Search IndicBridge", type="primary")

    if query_input and (run_search or preset_query):
        res = indicbridge.search(query_input, top_k=top_k)

        # IR Pipeline Panel
        st.markdown("### 🧭 IR Retrieval Pipeline Execution")
        pipe_cols = st.columns(4)
        pipe_cols[0].metric("Detected Script", res["detected_script"])
        pipe_cols[1].metric("Detected Language", res["detected_language"])
        pipe_cols[2].metric("Total Candidates", res["total_candidates"])
        pipe_cols[3].metric("Total Time", f"{res['timing_ms']['total']} ms")

        with st.expander("🔍 Click to view Full IR Preprocessing Pipeline Trace"):
            c1, c2 = st.columns(2)
            with c1:
                st.write("**1. Original Query:**", res["original_query"])
                st.write("**2. Normalized Query:**", res["normalized_query"])
                st.write("**3. Cross-Script Transliteration:**", res["transliterated_query"] or "None needed")
                st.write("**4. Removed Stop Words:**", res["removed_stopwords"] or "None")
            with c2:
                st.write("**5. Stemmed / Effective Tokens:**", res["stemmed_tokens"])
                st.write("**6. Retrieval Query Vector Terms:**", res["retrieval_query_terms"])
                st.write("**7. Timing Breakdown:**", res["timing_ms"])

        st.markdown("---")
        st.subheader(f"Search Results ({len(res['results'])} retrieved)")

        if not res["results"]:
            st.warning("No documents matched the query criteria.")
        else:
            for item in res["results"]:
                with st.container():
                    # Result Header
                    lang_badge = f"<span class='metric-badge badge-hi'>Hindi</span>" if item["language"] == "hi" else (
                        f"<span class='metric-badge badge-en'>English</span>" if item["language"] == "en" else
                        f"<span class='metric-badge badge-latn'>Hinglish</span>"
                    )
                    st.markdown(
                        f"### #{item['rank']} {item['title']} {lang_badge}",
                        unsafe_allow_html=True
                    )

                    st.write(item["text"])

                    # Score Breakdown Chips
                    sc_col1, sc_col2, sc_col3, sc_col4, sc_col5 = st.columns(5)
                    sc_col1.metric("Final Score", f"{item['final_score']:.4f}")
                    sc_col2.metric("TF-IDF Cosine", f"{item['cosine_score']:.4f}")
                    sc_col3.metric("Transliteration", f"{item['transliteration_score']:.4f}")
                    sc_col4.metric("Phonetic/Fuzzy", f"{item['phonetic_score']:.4f}")
                    sc_col5.metric("Exact Match", f"{item['exact_match_score']:.4f}")

                    # Human-readable explanation
                    st.info(f"💡 **Why this result was retrieved:** {item['explanation_summary']}")

                    # Deep IR Details
                    with st.expander(f"🔬 Show IR Details for Result #{item['rank']}"):
                        detail = item["explanation_details"]
                        c_left, c_right = st.columns(2)
                        with c_left:
                            st.write("**Matched Query Terms:**", item["matched_terms"])
                            st.write("**High-IDF Distinctive Terms:**", item["high_idf_terms"])
                            st.write("**Transliteration Matches:**", detail["translit_matches"] or "None")
                            st.write("**Phonetic Matches:**", detail["phonetic_matches"] or "None")
                        with c_right:
                            st.write("**TF-IDF Component Contributions ($w_{t,q} \\cdot w_{t,d}$):**")
                            st.json(item["term_contributions"])
                            st.write("**Component Weights Used:**")
                            st.json(detail["weights"])

                    st.markdown("---")


# =========================================================================
# TAB 2: BASELINE COMPARISON
# =========================================================================
elif mode == "⚖️ Baseline Comparison":
    st.header("Baseline vs. IndicBridge Comparison")
    st.write(
        "Compare pure exact-token TF-IDF (Baseline) against IndicBridge with cross-script transliteration, "
        "normalization, and adapted phonetic score fusion."
    )

    cmp_query = st.text_input(
        "Enter Comparison Query:",
        value="mera bharat",
        help="Try: 'kal ka mausam', 'mera bharat', 'bhaarat desh', 'krishan ji', 'delhi mausam'"
    )

    if st.button("⚖️ Run Comparative Search", type="primary"):
        b_res = baseline.search(cmp_query, top_k=top_k)
        i_res = indicbridge.search(cmp_query, top_k=top_k)

        st.subheader("Side-by-Side Results")
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"### 🛑 Baseline (Exact TF-IDF)\n*Retrieved: {len(b_res['results'])} in {b_res['elapsed_ms']} ms*")
            if not b_res["results"]:
                st.warning("No documents retrieved by baseline (exact vocabulary mismatch).")
            for r in b_res["results"]:
                st.markdown(f"**#{r['rank']} {r['title']}** (Score: `{r['final_score']:.4f}`)")
                st.caption(f"Matched: {r['matched_terms']}")
                st.write(r["text"][:140] + "...")
                st.divider()

        with c2:
            st.markdown(f"### 🚀 IndicBridge (Cross-Script & Phonetic)\n*Retrieved: {len(i_res['results'])} in {i_res['timing_ms']['total']} ms*")
            if not i_res["results"]:
                st.warning("No documents retrieved.")
            for r in i_res["results"]:
                st.markdown(f"**#{r['rank']} {r['title']}** (Score: `{r['final_score']:.4f}`)")
                st.caption(f"Matched: {r['matched_terms']} | Translit: {r['transliteration_score']:.2f} | Phonetic: {r['phonetic_score']:.2f}")
                st.write(r["text"][:140] + "...")
                st.divider()

        # Quantitative delta
        base_ids = set(r["doc_id"] for r in b_res["results"])
        ind_ids = set(r["doc_id"] for r in i_res["results"])
        novel_ids = ind_ids - base_ids

        st.success(
            f"**Retrieval Delta Analysis:** IndicBridge retrieved **{len(novel_ids)} additional relevant document(s)** "
            f"({', '.join(novel_ids) if novel_ids else 'None'}) that Baseline completely missed due to cross-script or spelling mismatch."
        )


# =========================================================================
# TAB 3: IR DIAGNOSTICS & POSTINGS VISUALIZER
# =========================================================================
elif mode == "🔬 IR Diagnostics & Postings":
    st.header("Inverted Index & Postings Diagnostics (Section 19)")
    st.write(
        "Directly inspect the internal inverted index data structures, vocabulary, Document Frequency (DF), "
        "and Term Frequency (TF) postings lists."
    )

    diag_term = st.text_input("Enter Index Term to Inspect:", value="bharat")

    if diag_term:
        clean = diag_term.strip().lower()
        diag = index.get_diagnostics(clean)
        idf_val = tfidf.get_idf(clean)

        c1, c2, c3 = st.columns(3)
        c1.metric("Inspected Term", f"'{clean}'")
        c2.metric("Document Frequency (DF)", f"{diag['document_frequency']} / {index.num_docs}")
        c3.metric("Computed IDF Weight", f"{idf_val:.4f}")

        st.markdown(f"**Textbook IDF Formulation:**")
        st.latex(r"idf(t) = \ln\left(\frac{N + 1}{df(t) + 1}\right) + 1")
        st.write(f"$= \\ln(({index.num_docs} + 1) / ({diag['document_frequency']} + 1)) + 1 = {idf_val:.4f}$")

        st.subheader("Postings List")
        if not diag["postings"]:
            st.warning(f"Term '{clean}' does not exist in the collection vocabulary.")
        else:
            postings_df = pd.DataFrame(diag["postings"])
            st.dataframe(postings_df, use_container_width=True)


# =========================================================================
# TAB 4: BENCHMARK EVALUATION & ABLATION
# =========================================================================
elif mode == "📊 Benchmark Evaluation":
    st.header("Benchmark Evaluation & Ablation Study")
    st.write(
        "Evaluate Baseline vs IndicBridge on the 24 ground-truth benchmark queries across all 6 categories."
    )

    col1, col2 = st.columns(2)
    with col1:
        run_eval_btn = st.button("▶️ Run Evaluation Benchmark", type="primary")
    with col2:
        run_ablation_btn = st.button("📊 Run Architectural Ablation Study")

    eval_json_path = os.path.join(config["paths"]["evaluation_results"], "evaluation_results.json")
    ablation_csv_path = os.path.join(config["paths"]["evaluation_results"], "ablation_results.csv")
    chart_path = os.path.join(config["paths"]["evaluation_results"], "ablation_chart.png")

    if run_eval_btn:
        with st.spinner("Executing benchmark across all 24 queries..."):
            evaluator = BenchmarkEvaluator(baseline, indicbridge, config["paths"]["evaluation_queries"])
            results = evaluator.compare(top_k=10)
            st.success("Benchmark completed successfully!")

    if os.path.exists(eval_json_path):
        with open(eval_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        st.subheader("Macro-Average Benchmark Comparison")
        comp = data["comparison"]
        m_df = pd.DataFrame([
            {
                "Metric": m.upper(),
                "Baseline": comp[m]["baseline"],
                "IndicBridge": comp[m]["indicbridge"],
                "Absolute Gain": comp[m]["absolute_gain"],
                "Relative Gain (%)": f"{comp[m]['relative_gain_pct']:+.1f}%"
            }
            for m in ["p@5", "p@10", "r@5", "r@10", "f1@5", "mrr", "map"]
        ])
        st.dataframe(m_df, use_container_width=True)

        st.subheader("Category-Wise Performance (IndicBridge)")
        cat_df = pd.DataFrame.from_dict(data["indicbridge"]["category_breakdown"], orient="index")
        st.dataframe(cat_df, use_container_width=True)

    if run_ablation_btn:
        with st.spinner("Executing 5-component ablation study..."):
            from evaluation.ablation import run_ablation
            run_ablation()
            st.success("Ablation study completed!")

    if os.path.exists(ablation_csv_path):
        st.subheader("Ablation Study: Component Impact")
        adf = pd.read_csv(ablation_csv_path)
        st.dataframe(adf, use_container_width=True)

    if os.path.exists(chart_path):
        st.image(chart_path, caption="IndicBridge Component Ablation Study Chart", use_container_width=True)
