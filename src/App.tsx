import React, { useState, useEffect } from "react";
import {
  Search,
  Scale,
  Database,
  BarChart3,
  Terminal,
  FileCode,
  Sparkles,
  Sliders,
  RefreshCw,
  CheckCircle2,
  AlertCircle,
  Cpu,
  Copy,
  Check,
  ChevronDown,
  ChevronUp,
  Play,
  Code2,
  ShieldCheck,
  SlidersHorizontal,
  ExternalLink,
  BookOpen,
} from "lucide-react";

interface SearchResult {
  doc_id: string;
  rank: number;
  title: string;
  language: string;
  text: string;
  url?: string;
  final_score: number;
  cosine_score: number;
  transliteration_score: number;
  phonetic_score: number;
  exact_match_score: number;
  matched_terms: string[];
  high_idf_terms?: string[];
  term_contributions?: Record<string, number>;
  explanation_summary?: string;
  explanation_details?: any;
}

interface SearchResponse {
  original_query: string;
  normalized_query?: string;
  transliterated_query?: string;
  detected_language?: string;
  detected_script?: string;
  confidence?: number;
  tokens?: string[];
  removed_stopwords?: string[];
  stemmed_tokens?: string[];
  retrieval_query_terms?: string[];
  total_candidates: number;
  timing_ms?: {
    query_processing: number;
    candidate_retrieval: number;
    ranking: number;
    total: number;
  };
  results: SearchResult[];
}

export default function App() {
  const [activeTab, setActiveTab] = useState<
    "search" | "compare" | "diagnostics" | "evaluation" | "cli" | "docs"
  >("search");

  // Search parameters
  const [query, setQuery] = useState("kal ka mausam");
  const [mode, setMode] = useState<"indicbridge" | "baseline">("indicbridge");
  const [topK, setTopK] = useState(8);
  const [alpha, setAlpha] = useState(0.7);
  const [beta, setBeta] = useState(0.15);
  const [gamma, setGamma] = useState(0.1);
  const [delta, setDelta] = useState(0.05);

  const [loading, setLoading] = useState(false);
  const [searchData, setSearchData] = useState<SearchResponse | null>(null);
  const [expandedDoc, setExpandedDoc] = useState<string | null>(null);

  // Compare state
  const [compareQuery, setCompareQuery] = useState("krishan ji");
  const [compareLoading, setCompareLoading] = useState(false);
  const [compareData, setCompareData] = useState<{
    baseline: SearchResponse;
    indicbridge: SearchResponse;
  } | null>(null);

  // Diagnostics state
  const [diagTerm, setDiagTerm] = useState("chandrayaan");
  const [diagLoading, setDiagLoading] = useState(false);
  const [diagData, setDiagData] = useState<any>(null);

  // Evaluation state
  const [evalLoading, setEvalLoading] = useState(false);
  const [evalData, setEvalData] = useState<any>(null);
  const [ablationData, setAblationData] = useState<any[]>([]);

  // CLI Playground state
  const [cliCommand, setCliCommand] = useState(
    'python indicbridge.py search "kal ka mausam" --explain'
  );
  const [cliLoading, setCliLoading] = useState(false);
  const [cliOutput, setCliOutput] = useState<{
    stdout: string;
    stderr: string;
    exitCode: number;
    elapsedMs: number;
  } | null>(null);
  const [copiedCmd, setCopiedCmd] = useState<string | null>(null);

  useEffect(() => {
    executeSearch("kal ka mausam", "indicbridge");
    loadEvaluationData();
  }, []);

  const executeSearch = async (
    q: string = query,
    m: "indicbridge" | "baseline" = mode
  ) => {
    if (!q.trim()) return;
    setLoading(true);
    try {
      const res = await fetch("/api/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: q,
          mode: m,
          top_k: topK,
          alpha,
          beta,
          gamma,
          delta,
        }),
      });
      const data = await res.json();
      if (data && Array.isArray(data.results)) {
        setSearchData(data);
      } else {
        setSearchData({
          original_query: q,
          total_candidates: 0,
          results: [],
          timing_ms: { query_processing: 0, candidate_retrieval: 0, ranking: 0, total: 0 },
        });
      }
    } catch (err) {
      console.error(err);
      setSearchData({
        original_query: q,
        total_candidates: 0,
        results: [],
        timing_ms: { query_processing: 0, candidate_retrieval: 0, ranking: 0, total: 0 },
      });
    } finally {
      setLoading(false);
    }
  };

  const executeCompare = async (q: string = compareQuery) => {
    if (!q.trim()) return;
    setCompareLoading(true);
    try {
      const [baseRes, indRes] = await Promise.all([
        fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ query: q, mode: "baseline", top_k: topK }),
        })
          .then((r) => r.json())
          .catch(() => ({ results: [] })),
        fetch("/api/search", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            query: q,
            mode: "indicbridge",
            top_k: topK,
            alpha,
            beta,
            gamma,
            delta,
          }),
        })
          .then((r) => r.json())
          .catch(() => ({ results: [] })),
      ]);
      setCompareData({
        baseline:
          baseRes && Array.isArray(baseRes.results)
            ? baseRes
            : { original_query: q, total_candidates: 0, results: [] },
        indicbridge:
          indRes && Array.isArray(indRes.results)
            ? indRes
            : { original_query: q, total_candidates: 0, results: [] },
      });
    } catch (err) {
      console.error(err);
    } finally {
      setCompareLoading(false);
    }
  };

  const executeDiagnostics = async (term: string = diagTerm) => {
    if (!term.trim()) return;
    setDiagLoading(true);
    try {
      const res = await fetch(`/api/diagnostics?term=${encodeURIComponent(term)}`);
      const data = await res.json();
      setDiagData(data);
    } catch (err) {
      console.error(err);
    } finally {
      setDiagLoading(false);
    }
  };

  const loadEvaluationData = async () => {
    setEvalLoading(true);
    try {
      const [eRes, aRes] = await Promise.all([
        fetch("/api/evaluation").then((r) => r.json()),
        fetch("/api/ablation").then((r) => r.json()),
      ]);
      setEvalData(eRes);
      if (aRes && aRes.rows) {
        setAblationData(aRes.rows);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setEvalLoading(false);
    }
  };

  const executeCliCommand = async (cmd: string = cliCommand) => {
    if (!cmd.trim()) return;
    setCliLoading(true);
    try {
      const res = await fetch("/api/cli", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd }),
      });
      const data = await res.json();
      setCliOutput(data);
    } catch (err: any) {
      setCliOutput({
        stdout: "",
        stderr: err.message || "Failed to execute CLI command",
        exitCode: 1,
        elapsedMs: 0,
      });
    } finally {
      setCliLoading(false);
    }
  };

  const handleWeightChange = (key: "alpha" | "beta" | "gamma" | "delta", val: number) => {
    const current = { alpha, beta, gamma, delta };
    current[key] = val;
    const remaining = Math.max(0, 1.0 - val);
    const otherKeys = (["alpha", "beta", "gamma", "delta"] as const).filter((k) => k !== key);
    const otherSum = otherKeys.reduce((acc, k) => acc + current[k], 0);

    if (otherSum > 0) {
      otherKeys.forEach((k) => {
        current[k] = parseFloat(((current[k] / otherSum) * remaining).toFixed(2));
      });
    }

    setAlpha(current.alpha);
    setBeta(current.beta);
    setGamma(current.gamma);
    setDelta(current.delta);
  };

  const applyWeightPreset = (preset: "default" | "translit" | "phonetic" | "lexical") => {
    if (preset === "default") {
      setAlpha(0.7);
      setBeta(0.15);
      setGamma(0.1);
      setDelta(0.05);
    } else if (preset === "translit") {
      setAlpha(0.45);
      setBeta(0.35);
      setGamma(0.15);
      setDelta(0.05);
    } else if (preset === "phonetic") {
      setAlpha(0.45);
      setBeta(0.15);
      setGamma(0.35);
      setDelta(0.05);
    } else if (preset === "lexical") {
      setAlpha(0.9);
      setBeta(0.05);
      setGamma(0.0);
      setDelta(0.05);
    }
  };

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedCmd(text);
    setTimeout(() => setCopiedCmd(null), 2000);
  };

  const queryPresets = [
    { text: "kal ka mausam", cat: "Romanized Hindi", note: "Devanagari weather alignment" },
    { text: "कल का मौसम", cat: "Hindi (Devanagari)", note: "Cross-script match with English/Hinglish" },
    { text: "weather tomorrow", cat: "English", note: "Matches English forecast and Hinglish blogs" },
    { text: "mera bharat mahan", cat: "Romanized Hindi", note: "Matches Devanagari 'मेरा भारत देश'" },
    { text: "bhaarat desh", cat: "Spelling Variant", note: "Normalizes 'aa' variant to root 'bharat'" },
    { text: "krishan ji", cat: "Phonetic Variant", note: "IndicSoundex aligns with 'krishna'" },
    { text: "chandrayaan space mission", cat: "Code-Mixed", note: "Retrieves Hindi & English ISRO docs" },
    { text: "delhi mausam", cat: "Cross-Script", note: "Bilingual mixture: Latin and Devanagari" },
  ];

  return (
    <div className="min-h-screen bg-stone-100/70 text-stone-900 flex flex-col font-sans selection:bg-amber-100 selection:text-amber-900">
      {/* Warm Header */}
      <header className="bg-white/90 backdrop-blur-md border-b border-stone-200 sticky top-0 z-50 shadow-xs">
        <div className="max-w-7xl mx-auto px-4 py-3 flex flex-wrap items-center justify-between gap-4">
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-600 via-orange-600 to-amber-700 text-white flex items-center justify-center font-black text-sm tracking-tight shadow-sm ring-2 ring-amber-100">
              IN
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-extrabold text-lg text-stone-900 tracking-tight">
                  INDICBRIDGE
                </h1>
                <span className="text-[10px] bg-amber-100/80 text-amber-900 font-bold px-2.5 py-0.5 rounded-full border border-amber-200 uppercase tracking-wider">
                  Track T5 • Indic Search
                </span>
              </div>
              <p className="text-[11.5px] text-stone-600 font-medium leading-none mt-1">
                Explainable Multilingual Search for Hindi, Hinglish & English
              </p>
            </div>
          </div>

          {/* Warm Navigation Pill */}
          <nav className="flex items-center gap-1 bg-stone-100 p-1.5 rounded-xl border border-stone-200/60 text-xs font-semibold">
            <button
              onClick={() => setActiveTab("search")}
              className={`px-3.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all ${
                activeTab === "search"
                  ? "bg-white text-amber-800 shadow-xs font-bold"
                  : "text-stone-600 hover:text-stone-900 hover:bg-white/50"
              }`}
            >
              <Search className="w-3.5 h-3.5 text-amber-600" />
              <span>Search</span>
            </button>
            <button
              onClick={() => {
                setActiveTab("compare");
                if (!compareData) executeCompare();
              }}
              className={`px-3.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all ${
                activeTab === "compare"
                  ? "bg-white text-amber-800 shadow-xs font-bold"
                  : "text-stone-600 hover:text-stone-900 hover:bg-white/50"
              }`}
            >
              <Scale className="w-3.5 h-3.5 text-amber-600" />
              <span>Comparison</span>
            </button>
            <button
              onClick={() => {
                setActiveTab("diagnostics");
                if (!diagData) executeDiagnostics();
              }}
              className={`px-3.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all ${
                activeTab === "diagnostics"
                  ? "bg-white text-amber-800 shadow-xs font-bold"
                  : "text-stone-600 hover:text-stone-900 hover:bg-white/50"
              }`}
            >
              <Database className="w-3.5 h-3.5 text-amber-600" />
              <span>Postings & IDF</span>
            </button>
            <button
              onClick={() => setActiveTab("evaluation")}
              className={`px-3.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all ${
                activeTab === "evaluation"
                  ? "bg-white text-amber-800 shadow-xs font-bold"
                  : "text-stone-600 hover:text-stone-900 hover:bg-white/50"
              }`}
            >
              <BarChart3 className="w-3.5 h-3.5 text-amber-600" />
              <span>Benchmarks</span>
            </button>
            <button
              onClick={() => {
                setActiveTab("cli");
                if (!cliOutput) executeCliCommand();
              }}
              className={`px-3.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all ${
                activeTab === "cli"
                  ? "bg-stone-900 text-amber-300 shadow-xs font-bold"
                  : "text-stone-600 hover:text-stone-900 hover:bg-white/50"
              }`}
            >
              <Terminal className="w-3.5 h-3.5" />
              <span>Terminal CLI</span>
            </button>
            <button
              onClick={() => setActiveTab("docs")}
              className={`px-3.5 py-1.5 rounded-lg flex items-center gap-1.5 transition-all ${
                activeTab === "docs"
                  ? "bg-white text-amber-800 shadow-xs font-bold"
                  : "text-stone-600 hover:text-stone-900 hover:bg-white/50"
              }`}
            >
              <BookOpen className="w-3.5 h-3.5 text-amber-600" />
              <span>Architecture & Docs</span>
            </button>
          </nav>
        </div>
      </header>

      {/* Main Body */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 space-y-6">
        {/* ========================================================================= */}
        {/* TAB 1: SEARCH WORKBENCH */}
        {/* ========================================================================= */}
        {activeTab === "search" && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
            {/* Sidebar Controls */}
            <aside className="lg:col-span-1 bg-white p-5 rounded-2xl border border-stone-200/80 shadow-xs space-y-5 h-fit">
              <div>
                <h3 className="text-xs font-extrabold uppercase tracking-wider text-stone-600 flex items-center gap-1.5 mb-2.5">
                  <Sliders className="w-3.5 h-3.5 text-amber-600" />
                  Retrieval Mode
                </h3>
                <div className="grid grid-cols-2 gap-2 text-xs font-semibold">
                  <button
                    onClick={() => {
                      setMode("indicbridge");
                      executeSearch(query, "indicbridge");
                    }}
                    className={`py-2 px-2.5 rounded-xl border text-center transition-all ${
                      mode === "indicbridge"
                        ? "bg-amber-50 border-amber-500 text-amber-900 font-bold shadow-xs"
                        : "border-stone-200 text-stone-600 hover:bg-stone-50"
                    }`}
                  >
                    IndicBridge (Full)
                  </button>
                  <button
                    onClick={() => {
                      setMode("baseline");
                      executeSearch(query, "baseline");
                    }}
                    className={`py-2 px-2.5 rounded-xl border text-center transition-all ${
                      mode === "baseline"
                        ? "bg-stone-200/80 border-stone-400 text-stone-900 font-bold shadow-xs"
                        : "border-stone-200 text-stone-600 hover:bg-stone-50"
                    }`}
                  >
                    Exact Baseline
                  </button>
                </div>
              </div>

              {/* Dynamic Weights */}
              {mode === "indicbridge" && (
                <div className="space-y-3.5 pt-3 border-t border-stone-100">
                  <div className="flex items-center justify-between">
                    <h3 className="text-xs font-extrabold uppercase tracking-wider text-stone-600 flex items-center gap-1">
                      <SlidersHorizontal className="w-3 h-3 text-amber-600" />
                      Score Fusion
                    </h3>
                    <span className="text-[10.5px] text-amber-900 font-mono font-bold bg-amber-50 px-2 py-0.5 rounded-md border border-amber-200">
                      Sum: {(alpha + beta + gamma + delta).toFixed(2)}
                    </span>
                  </div>

                  {/* Weight Presets */}
                  <div className="grid grid-cols-2 gap-1.5 text-[10px] font-semibold">
                    <button
                      onClick={() => applyWeightPreset("default")}
                      className="px-2 py-1 bg-stone-100 hover:bg-amber-50 hover:text-amber-800 text-stone-700 rounded-lg border border-stone-200 text-left transition-colors"
                    >
                      Balanced Standard
                    </button>
                    <button
                      onClick={() => applyWeightPreset("translit")}
                      className="px-2 py-1 bg-stone-100 hover:bg-amber-50 hover:text-amber-800 text-stone-700 rounded-lg border border-stone-200 text-left transition-colors"
                    >
                      Transliteration Pri
                    </button>
                    <button
                      onClick={() => applyWeightPreset("phonetic")}
                      className="px-2 py-1 bg-stone-100 hover:bg-amber-50 hover:text-amber-800 text-stone-700 rounded-lg border border-stone-200 text-left transition-colors"
                    >
                      Phonetic / Fuzzy
                    </button>
                    <button
                      onClick={() => applyWeightPreset("lexical")}
                      className="px-2 py-1 bg-stone-100 hover:bg-amber-50 hover:text-amber-800 text-stone-700 rounded-lg border border-stone-200 text-left transition-colors"
                    >
                      Strict Lexical
                    </button>
                  </div>

                  {/* Sliders */}
                  <div className="space-y-2.5 text-xs pt-1">
                    <div>
                      <div className="flex justify-between text-stone-700 mb-0.5">
                        <span className="font-medium">α (TF-IDF Cosine):</span>
                        <span className="font-mono font-bold text-amber-700">{alpha.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={alpha}
                        onChange={(e) => handleWeightChange("alpha", parseFloat(e.target.value))}
                        className="w-full accent-amber-600 h-1.5 bg-stone-200 rounded-lg cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-stone-700 mb-0.5">
                        <span className="font-medium">β (Transliteration):</span>
                        <span className="font-mono font-bold text-orange-700">{beta.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={beta}
                        onChange={(e) => handleWeightChange("beta", parseFloat(e.target.value))}
                        className="w-full accent-orange-600 h-1.5 bg-stone-200 rounded-lg cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-stone-700 mb-0.5">
                        <span className="font-medium">γ (Phonetic / Fuzzy):</span>
                        <span className="font-mono font-bold text-purple-700">{gamma.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={gamma}
                        onChange={(e) => handleWeightChange("gamma", parseFloat(e.target.value))}
                        className="w-full accent-purple-600 h-1.5 bg-stone-200 rounded-lg cursor-pointer"
                      />
                    </div>

                    <div>
                      <div className="flex justify-between text-stone-700 mb-0.5">
                        <span className="font-medium">δ (Exact Match):</span>
                        <span className="font-mono font-bold text-emerald-700">{delta.toFixed(2)}</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.05"
                        value={delta}
                        onChange={(e) => handleWeightChange("delta", parseFloat(e.target.value))}
                        className="w-full accent-emerald-600 h-1.5 bg-stone-200 rounded-lg cursor-pointer"
                      />
                    </div>
                  </div>

                  <button
                    onClick={() => executeSearch()}
                    className="w-full py-2 bg-amber-50 hover:bg-amber-100/80 text-amber-900 text-xs font-bold rounded-xl border border-amber-200/80 transition-all flex items-center justify-center gap-1.5 shadow-2xs"
                  >
                    <RefreshCw className="w-3.5 h-3.5 text-amber-700" />
                    Re-Rank with Weights
                  </button>
                </div>
              )}

              {/* Top-K Selector */}
              <div className="pt-3 border-t border-stone-100">
                <div className="flex justify-between text-xs text-stone-700 mb-1">
                  <span className="font-medium">Top-K Documents:</span>
                  <span className="font-mono font-bold text-amber-700">{topK}</span>
                </div>
                <input
                  type="range"
                  min="2"
                  max="16"
                  step="1"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full accent-amber-600 h-1.5 bg-stone-200 rounded-lg cursor-pointer"
                />
              </div>

              {/* VSM Mathematical Foundations */}
              <div className="p-3.5 bg-stone-50 rounded-xl text-[11px] text-stone-700 leading-relaxed border border-stone-200/80 space-y-1.5">
                <div className="font-bold flex items-center gap-1.5 text-stone-900">
                  <ShieldCheck className="w-3.5 h-3.5 text-amber-600 shrink-0" />
                  Vector Space Math:
                </div>
                <p className="font-mono text-[10px] text-stone-600">
                  ||d||₂ = √(∑ w(t,d)²)
                </p>
                <p className="text-[10.5px] text-stone-600">
                  Cosine dot product evaluates unit vectors: <code>q̂ · d̂</code>.
                </p>
              </div>
            </aside>

            {/* Right Main Search and Results */}
            <div className="lg:col-span-3 space-y-5">
              {/* Search Box */}
              <div className="bg-white p-5 rounded-2xl border border-stone-200/80 shadow-xs space-y-3.5">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    executeSearch();
                  }}
                  className="flex gap-2.5"
                >
                  <div className="relative flex-1">
                    <Search className="w-4 h-4 absolute left-3.5 top-3.5 text-stone-400" />
                    <input
                      type="text"
                      value={query}
                      onChange={(e) => setQuery(e.target.value)}
                      placeholder="Type query in Hindi (कल का मौसम), Hinglish (kal ka mausam), or English..."
                      className="w-full pl-10 pr-4 py-2.5 bg-stone-50/50 border border-stone-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/50 focus:border-amber-500 transition-all font-medium"
                    />
                  </div>
                  <button
                    type="submit"
                    disabled={loading}
                    className="bg-amber-600 hover:bg-amber-700 text-white font-bold px-6 py-2.5 rounded-xl text-sm flex items-center gap-2 transition-all disabled:opacity-50 shadow-xs"
                  >
                    {loading ? (
                      <RefreshCw className="w-4 h-4 animate-spin" />
                    ) : (
                      <Sparkles className="w-4 h-4" />
                    )}
                    Search
                  </button>
                </form>

                {/* Preset Chips */}
                <div className="space-y-1.5 pt-1">
                  <div className="flex items-center justify-between text-[11px] text-stone-600">
                    <span className="font-bold uppercase tracking-wider">
                      Evaluation Query Presets:
                    </span>
                    <span>Click any chip to search</span>
                  </div>
                  <div className="flex flex-wrap gap-1.5">
                    {queryPresets.map((p, idx) => (
                      <button
                        key={idx}
                        onClick={() => {
                          setQuery(p.text);
                          executeSearch(p.text, mode);
                        }}
                        className={`text-xs px-3 py-1 rounded-lg border transition-all text-left flex items-center gap-1.5 ${
                          query === p.text
                            ? "bg-amber-50 border-amber-400 text-amber-900 font-semibold ring-1 ring-amber-300"
                            : "bg-stone-50 hover:bg-stone-100 border-stone-200 text-stone-700"
                        }`}
                        title={p.note}
                      >
                        <span className="font-medium">{p.text}</span>
                        <span className="text-[9.5px] px-1.5 py-0.2 rounded bg-white text-stone-600 border border-stone-200">
                          {p.cat}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* 11-Stage IR Execution Trace Card */}
              {searchData && (
                <div className="bg-white p-5 rounded-2xl border border-stone-200/80 shadow-xs space-y-4 text-xs">
                  <div className="flex items-center justify-between flex-wrap gap-2 border-b border-stone-100 pb-3">
                    <div className="flex items-center gap-2">
                      <Cpu className="w-4 h-4 text-amber-600" />
                      <span className="font-extrabold text-stone-900">
                        11-Stage IR Pipeline Execution Trace
                      </span>
                    </div>
                    {searchData.timing_ms && (
                      <div className="flex items-center gap-3 text-stone-600 text-[11px]">
                        <span>Preprocessing: <b>{searchData.timing_ms.query_processing} ms</b></span>
                        <span>Postings: <b>{searchData.timing_ms.candidate_retrieval} ms</b></span>
                        <span>Ranking: <b>{searchData.timing_ms.ranking} ms</b></span>
                        <span className="text-amber-900 font-extrabold bg-amber-50 px-2.5 py-0.5 rounded-md border border-amber-200">
                          Total: {searchData.timing_ms.total} ms
                        </span>
                      </div>
                    )}
                  </div>

                  {/* Metric Badges */}
                  <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                    <div className="bg-stone-50/80 p-3 rounded-xl border border-stone-200/60">
                      <span className="text-stone-500 block text-[10px] uppercase font-bold">
                        Detected Script
                      </span>
                      <span className="font-extrabold text-stone-900 text-sm">
                        {searchData.detected_script || "Unknown"}
                      </span>
                    </div>
                    <div className="bg-stone-50/80 p-3 rounded-xl border border-stone-200/60">
                      <span className="text-stone-500 block text-[10px] uppercase font-bold">
                        Classified Language
                      </span>
                      <span className="font-extrabold text-stone-900 text-sm">
                        {searchData.detected_language || "Unknown"}
                      </span>
                    </div>
                    <div className="bg-stone-50/80 p-3 rounded-xl border border-stone-200/60">
                      <span className="text-stone-500 block text-[10px] uppercase font-bold">
                        Candidates Evaluated
                      </span>
                      <span className="font-extrabold text-stone-900 text-sm">
                        {searchData.total_candidates} documents
                      </span>
                    </div>
                    <div className="bg-stone-50/80 p-3 rounded-xl border border-stone-200/60">
                      <span className="text-stone-500 block text-[10px] uppercase font-bold">
                        Top-K Ranked
                      </span>
                      <span className="font-extrabold text-amber-700 text-sm">
                        {Array.isArray(searchData.results) ? searchData.results.length : 0} results
                      </span>
                    </div>
                  </div>

                  {/* Transliteration Callout Banner */}
                  {searchData.transliterated_query && (
                    <div className="text-[11px] bg-amber-50/70 border border-amber-200 p-3 rounded-xl text-amber-950 flex items-center justify-between gap-2">
                      <div className="flex items-center gap-2">
                        <span className="font-bold text-amber-800">
                          Cross-Script Transliteration Normalization:
                        </span>
                        <span className="font-mono bg-white px-2 py-0.5 rounded border border-amber-300 font-semibold">
                          "{searchData.original_query}" ➔ "{searchData.transliterated_query}"
                        </span>
                      </div>
                      <span className="text-[10px] text-amber-800 bg-amber-100/90 px-2 py-0.5 rounded-full font-bold">
                        Dual Script Matches
                      </span>
                    </div>
                  )}

                  {/* All 11 Stages rendered clearly */}
                  <div className="flex flex-wrap gap-1.5 pt-1 text-[10.5px]">
                    <span className="px-2.5 py-1 rounded-md bg-stone-100 text-stone-700 border border-stone-200">
                      1. Script Detection ({searchData.detected_script})
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-stone-100 text-stone-700 border border-stone-200">
                      2. Unicode NFC Normalization
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-stone-100 text-stone-700 border border-stone-200">
                      3. Nukta Folding
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-stone-100 text-stone-700 border border-stone-200">
                      4. Transliteration Expansion
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-stone-100 text-stone-700 border border-stone-200">
                      5. Romanized Regularization
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-stone-100 text-stone-700 border border-stone-200">
                      6. Stopwords Filtered ({searchData.removed_stopwords?.length ? searchData.removed_stopwords.join(", ") : "None"})
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-stone-100 text-stone-700 border border-stone-200">
                      7. Suffix Stemming
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-amber-50 text-amber-900 border border-amber-200 font-semibold">
                      8. Postings Lookup
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-orange-50 text-orange-900 border border-orange-200 font-semibold">
                      9. Sublinear TF-IDF Cosine
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-purple-50 text-purple-900 border border-purple-200 font-semibold">
                      10. IndicSoundex Alignment
                    </span>
                    <span className="px-2.5 py-1 rounded-md bg-emerald-50 text-emerald-900 border border-emerald-200 font-semibold">
                      11. Score Fusion
                    </span>
                  </div>
                </div>
              )}

              {/* Results Stream */}
              <div className="space-y-4">
                {loading ? (
                  <div className="bg-white p-14 rounded-2xl border border-stone-200 text-center space-y-3 shadow-xs">
                    <RefreshCw className="w-8 h-8 animate-spin text-amber-600 mx-auto" />
                    <p className="text-sm font-bold text-stone-800">
                      Running 11-Stage IR Vector Retrieval & Cross-Script Fusion...
                    </p>
                    <p className="text-xs text-stone-500">
                      Computing TF-IDF vector projections, IndicSoundex keys, and cosine dot products
                    </p>
                  </div>
                ) : searchData && Array.isArray(searchData.results) && searchData.results.length > 0 ? (
                  searchData.results.map((res) => {
                    const isExpanded = expandedDoc === res.doc_id;
                    const langBadge =
                      res.language === "hi"
                        ? "bg-rose-50 text-rose-800 border-rose-200"
                        : res.language === "en"
                        ? "bg-blue-50 text-blue-800 border-blue-200"
                        : "bg-amber-50 text-amber-800 border-amber-200";

                    return (
                      <div
                        key={res.doc_id}
                        className="bg-white rounded-2xl border border-stone-200 shadow-xs hover:border-amber-300 transition-all p-5 space-y-4"
                      >
                        {/* Result Item Header */}
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex items-start gap-3">
                            <span className="w-7 h-7 rounded-lg bg-amber-50 text-amber-800 font-extrabold text-xs flex items-center justify-center shrink-0 border border-amber-200">
                              #{res.rank}
                            </span>
                            <div>
                              <div className="flex items-center gap-2 flex-wrap">
                                <h4 className="font-extrabold text-base text-stone-900 hover:text-amber-800 transition-colors">
                                  {res.title}
                                </h4>
                                <span className={`text-[10px] font-extrabold uppercase px-2.5 py-0.5 rounded-full border ${langBadge}`}>
                                  {res.language === "hi"
                                    ? "Hindi (Devanagari)"
                                    : res.language === "en"
                                    ? "English"
                                    : "Hinglish / Romanized"}
                                </span>
                              </div>
                              <span className="text-[11px] text-stone-500 font-mono">
                                Doc ID: {res.doc_id}
                              </span>
                            </div>
                          </div>

                          <div className="text-right shrink-0 bg-amber-50/80 border border-amber-200 px-3 py-1.5 rounded-xl">
                            <span className="text-[9.5px] text-amber-800 uppercase font-black block">
                              Fused Score
                            </span>
                            <span className="text-base font-black text-amber-900 font-mono">
                              {res.final_score.toFixed(4)}
                            </span>
                          </div>
                        </div>

                        {/* Document Content */}
                        <p className="text-sm text-stone-700 leading-relaxed bg-stone-50/70 p-3.5 rounded-xl border border-stone-200/60 font-sans">
                          {res.text}
                        </p>

                        {/* Progress Indicators */}
                        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 text-xs bg-stone-50/60 p-3 rounded-xl border border-stone-200/60">
                          <div className="space-y-1">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-stone-500">TF-IDF Cosine:</span>
                              <span className="font-mono font-bold text-stone-800">
                                {res.cosine_score.toFixed(3)}
                              </span>
                            </div>
                            <div className="w-full bg-stone-200 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="bg-amber-600 h-full rounded-full"
                                style={{ width: `${Math.min(100, res.cosine_score * 100)}%` }}
                              />
                            </div>
                          </div>

                          <div className="space-y-1">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-stone-500">Transliteration:</span>
                              <span className="font-mono font-bold text-stone-800">
                                {res.transliteration_score.toFixed(3)}
                              </span>
                            </div>
                            <div className="w-full bg-stone-200 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="bg-orange-500 h-full rounded-full"
                                style={{ width: `${Math.min(100, res.transliteration_score * 100)}%` }}
                              />
                            </div>
                          </div>

                          <div className="space-y-1">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-stone-500">Phonetic:</span>
                              <span className="font-mono font-bold text-stone-800">
                                {res.phonetic_score.toFixed(3)}
                              </span>
                            </div>
                            <div className="w-full bg-stone-200 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="bg-purple-600 h-full rounded-full"
                                style={{ width: `${Math.min(100, res.phonetic_score * 100)}%` }}
                              />
                            </div>
                          </div>

                          <div className="space-y-1">
                            <div className="flex justify-between text-[11px]">
                              <span className="text-stone-500">Exact Match:</span>
                              <span className="font-mono font-bold text-stone-800">
                                {res.exact_match_score.toFixed(3)}
                              </span>
                            </div>
                            <div className="w-full bg-stone-200 h-1.5 rounded-full overflow-hidden">
                              <div
                                className="bg-emerald-600 h-full rounded-full"
                                style={{ width: `${Math.min(100, res.exact_match_score * 100)}%` }}
                              />
                            </div>
                          </div>
                        </div>

                        {/* Explainable Summary */}
                        {res.explanation_summary && (
                          <div className="p-3 bg-amber-50/70 border border-amber-200/80 rounded-xl text-xs text-amber-950 flex items-start gap-2.5">
                            <Sparkles className="w-4 h-4 text-amber-700 shrink-0 mt-0.5" />
                            <div>
                              <span className="font-extrabold text-amber-900">
                                Explainable Retrieval Rationale:
                              </span>{" "}
                              {res.explanation_summary}
                            </div>
                          </div>
                        )}

                        {/* Expand Details Trigger */}
                        <div className="pt-1 flex items-center justify-between">
                          <button
                            onClick={() =>
                              setExpandedDoc(isExpanded ? null : res.doc_id)
                            }
                            className="text-xs text-amber-800 hover:text-amber-950 font-bold flex items-center gap-1.5 transition-colors"
                          >
                            {isExpanded ? (
                              <>
                                <ChevronUp className="w-3.5 h-3.5" /> Hide Mathematical Vector Breakdown
                              </>
                            ) : (
                              <>
                                <ChevronDown className="w-3.5 h-3.5" /> Show Mathematical Vector Breakdown & IDF Values
                              </>
                            )}
                          </button>

                          {res.url && (
                            <a
                              href={res.url}
                              target="_blank"
                              rel="noreferrer"
                              className="text-[11px] text-stone-500 hover:text-stone-700 flex items-center gap-1"
                            >
                              Source Record <ExternalLink className="w-3 h-3" />
                            </a>
                          )}
                        </div>

                        {/* Mathematical Drawer */}
                        {isExpanded && res.explanation_details && (
                          <div className="p-4 bg-stone-900 text-stone-100 rounded-xl text-xs space-y-3 font-mono shadow-inner">
                            <div className="flex items-center justify-between border-b border-stone-800 pb-2">
                              <span className="font-bold text-amber-400 flex items-center gap-1.5">
                                <Code2 className="w-4 h-4" /> Vector Dot-Product Math for {res.doc_id}
                              </span>
                              <span className="text-[10px] text-stone-400">
                                L2 Unit Normalized
                              </span>
                            </div>

                            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                              <div>
                                <span className="text-stone-400 block mb-1">
                                  Matched Terms ({res.matched_terms.length}):
                                </span>
                                <div className="flex flex-wrap gap-1">
                                  {res.matched_terms.map((t, i) => (
                                    <span
                                      key={i}
                                      className="bg-stone-800 px-2 py-0.5 rounded text-amber-300 border border-stone-700"
                                    >
                                      {t}
                                    </span>
                                  ))}
                                </div>
                              </div>

                              {res.high_idf_terms && (
                                <div>
                                  <span className="text-stone-400 block mb-1">
                                    High-IDF Discriminative Terms:
                                  </span>
                                  <div className="flex flex-wrap gap-1">
                                    {res.high_idf_terms.map((t, i) => (
                                      <span
                                        key={i}
                                        className="bg-purple-900/60 text-purple-300 px-2 py-0.5 rounded border border-purple-700"
                                      >
                                        {t}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>

                            {res.term_contributions && (
                              <div>
                                <span className="text-stone-400 block mb-1">
                                  Term Dot-Product Contributions (w_q · w_d):
                                </span>
                                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2">
                                  {Object.entries(res.term_contributions).map(
                                    ([term, val]) => (
                                      <div
                                        key={term}
                                        className="bg-stone-800/90 p-2 rounded border border-stone-700 text-[11px]"
                                      >
                                        <span className="text-stone-400">{term}:</span>{" "}
                                        <b className="text-emerald-400">{val.toFixed(4)}</b>
                                      </div>
                                    )
                                  )}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })
                ) : (
                  <div className="bg-white p-12 rounded-2xl border border-stone-200 text-center space-y-2 shadow-xs">
                    <AlertCircle className="w-8 h-8 text-amber-600 mx-auto" />
                    <p className="text-sm font-bold text-stone-800">
                      No documents matched the query criteria
                    </p>
                    <p className="text-xs text-stone-500">
                      Try selecting a preset query above or lowering the minimum similarity threshold.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 2: BASELINE COMPARISON */}
        {/* ========================================================================= */}
        {activeTab === "compare" && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-stone-200 shadow-xs space-y-4">
              <div className="flex items-center justify-between">
                <div>
                  <h2 className="text-base font-extrabold text-stone-900 flex items-center gap-2">
                    <Scale className="w-5 h-5 text-amber-600" />
                    Baseline vs. IndicBridge Head-to-Head Comparison
                  </h2>
                  <p className="text-xs text-stone-600 mt-1">
                    Direct empirical comparison: Exact-token TF-IDF (Baseline) vs. Full 11-Stage Multilingual IndicBridge.
                  </p>
                </div>
              </div>

              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  executeCompare();
                }}
                className="flex gap-2.5"
              >
                <input
                  type="text"
                  value={compareQuery}
                  onChange={(e) => setCompareQuery(e.target.value)}
                  placeholder="Enter query to compare (e.g. 'krishan ji', 'kal ka mausam')..."
                  className="flex-1 px-4 py-2.5 border border-stone-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/50"
                />
                <button
                  type="submit"
                  disabled={compareLoading}
                  className="bg-amber-600 hover:bg-amber-700 text-white font-bold px-6 py-2.5 rounded-xl text-sm flex items-center gap-2 shadow-xs"
                >
                  {compareLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Scale className="w-4 h-4" />}
                  Run Comparison
                </button>
              </form>
            </div>

            {compareData && (
              <div className="space-y-4">
                {(() => {
                  const baseResults = compareData.baseline?.results || [];
                  const indResults = compareData.indicbridge?.results || [];
                  const baseIds = new Set(baseResults.map((r) => r.doc_id));
                  const indIds = new Set(indResults.map((r) => r.doc_id));
                  const additional = Array.from(indIds).filter((id) => !baseIds.has(id));

                  return (
                    <div className="p-4 bg-emerald-50 border border-emerald-200 rounded-xl text-xs text-emerald-950 flex items-start gap-3">
                      <CheckCircle2 className="w-5 h-5 text-emerald-600 shrink-0 mt-0.5" />
                      <div>
                        <span className="font-extrabold text-sm text-emerald-900 block mb-0.5">
                          Empirical Retrieval Delta Analysis:
                        </span>
                        IndicBridge retrieved{" "}
                        <b className="text-emerald-800 underline">
                          {additional.length} additional relevant document(s)
                        </b>{" "}
                        ({additional.join(", ") || "None in Top-K"}) that the exact Baseline completely missed due to cross-script or spelling mismatch.
                      </div>
                    </div>
                  );
                })()}

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  {/* Baseline Column */}
                  <div className="space-y-3">
                    <div className="p-4 bg-stone-100 rounded-xl border border-stone-200">
                      <h3 className="font-extrabold text-stone-900 text-sm">
                        Baseline (Exact-Token TF-IDF)
                      </h3>
                      <p className="text-[11px] text-stone-600 mt-1">
                        Tokenizes raw string only. Fails on script divergence.
                      </p>
                      <span className="text-xs font-bold text-stone-800 mt-2 block">
                        Retrieved: {compareData.baseline?.results?.length || 0} documents
                      </span>
                    </div>

                    {(compareData.baseline?.results || []).map((r) => (
                      <div key={r.doc_id} className="bg-white p-4 rounded-xl border border-stone-200 shadow-xs space-y-2 text-xs">
                        <div className="flex justify-between items-start">
                          <span className="font-bold text-stone-900">#{r.rank} {r.title}</span>
                          <span className="font-mono text-stone-700 font-extrabold">{r.final_score.toFixed(4)}</span>
                        </div>
                        <p className="text-stone-600 line-clamp-2 leading-relaxed">{r.text}</p>
                      </div>
                    ))}
                  </div>

                  {/* IndicBridge Column */}
                  <div className="space-y-3">
                    <div className="p-4 bg-amber-50/80 rounded-xl border border-amber-200">
                      <h3 className="font-extrabold text-amber-950 text-sm">
                        IndicBridge (Cross-Script & Phonetic)
                      </h3>
                      <p className="text-[11px] text-amber-800 mt-1">
                        11-Stage fused pipeline with Unicode normalizer and IndicSoundex.
                      </p>
                      <span className="text-xs font-bold text-amber-900 mt-2 block">
                        Retrieved: {compareData.indicbridge?.results?.length || 0} documents
                      </span>
                    </div>

                    {(compareData.indicbridge?.results || []).map((r) => (
                      <div key={r.doc_id} className="bg-white p-4 rounded-xl border border-amber-200 shadow-xs space-y-2 text-xs">
                        <div className="flex justify-between items-start">
                          <span className="font-bold text-stone-900">#{r.rank} {r.title}</span>
                          <span className="font-mono text-amber-700 font-extrabold">{r.final_score.toFixed(4)}</span>
                        </div>
                        <p className="text-stone-600 line-clamp-2 leading-relaxed">{r.text}</p>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 3: IR DIAGNOSTICS & POSTINGS */}
        {/* ========================================================================= */}
        {activeTab === "diagnostics" && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-stone-200 shadow-xs space-y-4">
              <h2 className="text-base font-extrabold text-stone-900 flex items-center gap-2">
                <Database className="w-5 h-5 text-amber-600" />
                Inverted Index Postings & IDF Diagnostics
              </h2>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  executeDiagnostics();
                }}
                className="flex gap-2.5"
              >
                <input
                  type="text"
                  value={diagTerm}
                  onChange={(e) => setDiagTerm(e.target.value)}
                  placeholder="Enter term to inspect (e.g. bharat, mausam, chandrayaan, संविधान)..."
                  className="flex-1 px-4 py-2.5 border border-stone-300 rounded-xl text-sm focus:outline-none focus:ring-2 focus:ring-amber-500/50 font-mono"
                />
                <button
                  type="submit"
                  disabled={diagLoading}
                  className="bg-amber-600 hover:bg-amber-700 text-white font-bold px-6 py-2.5 rounded-xl text-sm shadow-xs"
                >
                  {diagLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                  Inspect Term
                </button>
              </form>
            </div>

            {diagData && (
              <div className="bg-white p-6 rounded-2xl border border-stone-200 shadow-xs space-y-4">
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                  <div className="bg-stone-50 p-3.5 rounded-xl border border-stone-200/60">
                    <span className="text-[10px] text-stone-500 uppercase font-bold">Term</span>
                    <span className="text-base font-extrabold text-stone-900 block font-mono">"{diagData.term}"</span>
                  </div>
                  <div className="bg-stone-50 p-3.5 rounded-xl border border-stone-200/60">
                    <span className="text-[10px] text-stone-500 uppercase font-bold">Document Frequency</span>
                    <span className="text-base font-extrabold text-stone-900 block font-mono">
                      {diagData.document_frequency} / {diagData.collection_size} docs
                    </span>
                  </div>
                  <div className="bg-stone-50 p-3.5 rounded-xl border border-stone-200/60">
                    <span className="text-[10px] text-stone-500 uppercase font-bold">Calculated IDF</span>
                    <span className="text-base font-extrabold text-amber-700 block font-mono">{diagData.idf || "N/A"}</span>
                  </div>
                </div>

                <h3 className="font-extrabold text-sm text-stone-900 pt-2">
                  Postings List for "{diagData.term}" ({diagData.postings.length} entries)
                </h3>
                <div className="overflow-x-auto border border-stone-200 rounded-xl">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-stone-50 text-stone-600 border-b border-stone-200">
                      <tr>
                        <th className="p-3">Doc ID</th>
                        <th className="p-3">Title</th>
                        <th className="p-3">Language</th>
                        <th className="p-3">TF</th>
                        <th className="p-3">Positions</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-stone-100 font-mono">
                      {diagData.postings.map((p: any) => (
                        <tr key={p.doc_id} className="hover:bg-stone-50/60">
                          <td className="p-3 font-bold text-amber-800">{p.doc_id}</td>
                          <td className="p-3 font-sans font-medium text-stone-900">{p.title}</td>
                          <td className="p-3 text-stone-500 uppercase text-[10px]">{p.language}</td>
                          <td className="p-3 font-bold text-stone-800">{p.tf}</td>
                          <td className="p-3 text-stone-500 text-[11px]">[{p.positions.slice(0, 8).join(", ")}]</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 4: BENCHMARK & ABLATION */}
        {/* ========================================================================= */}
        {activeTab === "evaluation" && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-stone-200 shadow-xs flex items-center justify-between">
              <div>
                <h2 className="text-base font-extrabold text-stone-900 flex items-center gap-2">
                  <BarChart3 className="w-5 h-5 text-amber-600" />
                  Academic Benchmark Evaluation & Ablation
                </h2>
                <p className="text-xs text-stone-600 mt-1">
                  Empirical metrics across 24 test queries and 5 pipeline ablation configurations.
                </p>
              </div>
              <button
                onClick={loadEvaluationData}
                disabled={evalLoading}
                className="text-xs bg-amber-50 hover:bg-amber-100 text-amber-900 px-3.5 py-2 rounded-xl font-bold flex items-center gap-1.5 transition-colors border border-amber-200"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${evalLoading ? "animate-spin" : ""}`} />
                Refresh
              </button>
            </div>

            {/* Macro Metrics Comparison Table */}
            {evalData && evalData.comparison && (
              <div className="bg-white p-6 rounded-2xl border border-stone-200 shadow-xs space-y-4">
                <h3 className="font-extrabold text-sm text-stone-900">Macro-Average Metrics Comparison</h3>
                <div className="overflow-x-auto border border-stone-200 rounded-xl">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-stone-50 text-stone-600 border-b border-stone-200">
                      <tr>
                        <th className="p-3">Metric</th>
                        <th className="p-3">Baseline (Exact)</th>
                        <th className="p-3">IndicBridge (Multilingual)</th>
                        <th className="p-3">Absolute Gain</th>
                        <th className="p-3">Gain (%)</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-stone-100 font-mono">
                      {["p@5", "p@10", "r@5", "r@10", "f1@5", "mrr", "map"].map((m) => {
                        const c = evalData.comparison[m];
                        if (!c) return null;
                        const isGain = c.relative_gain_pct >= 0;
                        return (
                          <tr key={m} className="hover:bg-stone-50/60">
                            <td className="p-3 font-sans font-extrabold text-stone-900 uppercase">{m}</td>
                            <td className="p-3 text-stone-600">{c.baseline.toFixed(4)}</td>
                            <td className="p-3 text-amber-800 font-bold">{c.indicbridge.toFixed(4)}</td>
                            <td className="p-3 text-stone-700">{c.absolute_gain >= 0 ? "+" : ""}{c.absolute_gain.toFixed(4)}</td>
                            <td className="p-3">
                              <span className={`px-2.5 py-0.5 rounded-full text-[11px] font-bold ${
                                isGain ? "bg-emerald-100 text-emerald-800" : "bg-stone-100 text-stone-700"
                              }`}>
                                {isGain ? "+" : ""}{c.relative_gain_pct.toFixed(1)}%
                              </span>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Ablation Table */}
            {ablationData && ablationData.length > 0 && (
              <div className="bg-white p-6 rounded-2xl border border-stone-200 shadow-xs space-y-4">
                <h3 className="font-extrabold text-sm text-stone-900">
                  Architectural Component Ablation Study (5 Configurations)
                </h3>
                <div className="overflow-x-auto border border-stone-200 rounded-xl">
                  <table className="w-full text-xs text-left">
                    <thead className="bg-stone-50 text-stone-600 border-b border-stone-200">
                      <tr>
                        <th className="p-3">Configuration</th>
                        <th className="p-3">Description</th>
                        <th className="p-3">P@5</th>
                        <th className="p-3">R@5</th>
                        <th className="p-3">MRR</th>
                        <th className="p-3">MAP</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-stone-100 font-mono">
                      {ablationData.map((row, idx) => (
                        <tr key={idx} className={idx === ablationData.length - 1 ? "bg-amber-50/70 font-bold text-amber-950" : "hover:bg-stone-50/60"}>
                          <td className="p-3 font-sans font-extrabold text-stone-900">{row.Configuration}</td>
                          <td className="p-3 font-sans text-stone-500 text-[11px]">{row.Description}</td>
                          <td className="p-3 text-stone-700">{row["P@5"]?.toFixed?.(4) ?? row["P@5"]}</td>
                          <td className="p-3 text-amber-800 font-bold">{row["R@5"]?.toFixed?.(4) ?? row["R@5"]}</td>
                          <td className="p-3 text-stone-700">{row.MRR?.toFixed?.(4) ?? row.MRR}</td>
                          <td className="p-3 text-emerald-700 font-bold">{row.MAP?.toFixed?.(4) ?? row.MAP}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 5: CLI PLAYGROUND */}
        {/* ========================================================================= */}
        {activeTab === "cli" && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-stone-200 shadow-xs space-y-4">
              <h2 className="text-base font-extrabold text-stone-900 flex items-center gap-2">
                <Terminal className="w-5 h-5 text-amber-600" />
                Interactive CLI Playground
              </h2>
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  executeCliCommand();
                }}
                className="flex gap-2.5"
              >
                <div className="relative flex-1 font-mono text-xs">
                  <span className="absolute left-3.5 top-3 text-amber-500 font-bold">$</span>
                  <input
                    type="text"
                    value={cliCommand}
                    onChange={(e) => setCliCommand(e.target.value)}
                    className="w-full pl-8 pr-4 py-2.5 bg-stone-900 text-amber-200 rounded-xl font-mono text-xs focus:outline-none focus:ring-2 focus:ring-amber-500"
                  />
                </div>
                <button
                  type="submit"
                  disabled={cliLoading}
                  className="bg-amber-600 hover:bg-amber-700 text-white font-bold px-6 py-2.5 rounded-xl text-xs flex items-center gap-2 shadow-xs"
                >
                  {cliLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-3.5 h-3.5 fill-current" />}
                  Execute CLI
                </button>
              </form>
            </div>

            {/* Terminal Screen */}
            <div className="bg-stone-950 rounded-2xl border border-stone-800 shadow-xl overflow-hidden font-mono text-xs">
              <div className="bg-stone-900 px-4 py-2.5 border-b border-stone-800 flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <span className="w-3 h-3 rounded-full bg-rose-500 inline-block" />
                  <span className="w-3 h-3 rounded-full bg-amber-500 inline-block" />
                  <span className="w-3 h-3 rounded-full bg-emerald-500 inline-block" />
                  <span className="text-stone-400 text-[11px] ml-2">bash — {cliCommand}</span>
                </div>
                {cliOutput && (
                  <span className="text-[10px] text-stone-400">Exit: {cliOutput.exitCode} | Latency: {cliOutput.elapsedMs} ms</span>
                )}
              </div>
              <div className="p-5 text-stone-200 overflow-x-auto min-h-[320px] max-h-[500px] overflow-y-auto leading-relaxed">
                {cliLoading ? (
                  <div className="text-amber-400 flex items-center gap-2">
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Executing on host machine...
                  </div>
                ) : cliOutput ? (
                  <div>
                    {cliOutput.stdout && <pre className="whitespace-pre-wrap text-emerald-400 font-mono">{cliOutput.stdout}</pre>}
                    {cliOutput.stderr && <pre className="whitespace-pre-wrap text-rose-400 font-mono mt-2">{cliOutput.stderr}</pre>}
                  </div>
                ) : (
                  <p className="text-stone-500">Run any command above to inspect terminal stream output.</p>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ========================================================================= */}
        {/* TAB 6: ARCHITECTURE & DOCUMENTATION */}
        {/* ========================================================================= */}
        {activeTab === "docs" && (
          <div className="space-y-6">
            <div className="bg-white p-6 rounded-2xl border border-stone-200 shadow-xs space-y-3">
              <h2 className="text-lg font-extrabold text-stone-900 flex items-center gap-2">
                <BookOpen className="w-5 h-5 text-amber-600" />
                System Architecture & Course Project Specifications
              </h2>
              <p className="text-xs text-stone-600 leading-relaxed">
                Academic documentation of the IndicBridge information retrieval system designed for CSD358 Track T5 (Multilingual and Indic-language search).
              </p>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-5 text-xs">
              <div className="p-5 bg-white rounded-2xl border border-stone-200 shadow-xs space-y-2.5">
                <h3 className="font-extrabold text-stone-900 text-sm">1. Corpus & Data Provenance</h3>
                <p className="text-stone-600 leading-relaxed">
                  The dataset contains 32 curated documents spanning Hindi in Devanagari script, English in Latin script, and Romanized Hindi (Hinglish). Content is adapted from public knowledge portals (india.gov.in, IMD bulletins, and ISRO lunar archives) to benchmark genuine cross-script retrieval challenges.
                </p>
              </div>

              <div className="p-5 bg-white rounded-2xl border border-stone-200 shadow-xs space-y-2.5">
                <h3 className="font-extrabold text-stone-900 text-sm">2. 11-Stage IR Pipeline</h3>
                <p className="text-stone-600 leading-relaxed">
                  Combines script classification, NFC normalization, Devanagari nukta folding, bidirectional transliteration expansion, and IndicSoundex phonetic group matching with classical smoothed TF-IDF cosine ranking over an in-memory inverted index.
                </p>
              </div>

              <div className="p-5 bg-white rounded-2xl border border-stone-200 shadow-xs space-y-2.5">
                <h3 className="font-extrabold text-stone-900 text-sm">3. Transparent Score Fusion</h3>
                <p className="text-stone-600 leading-relaxed font-mono text-[11px] bg-stone-50 p-2.5 rounded-lg border border-stone-200">
                  final_score = α·cosine + β·translit + γ·phonetic + δ·exact
                </p>
                <p className="text-stone-600 leading-relaxed">
                  All signal components are scaled to [0, 1] before computing the linear combination, allowing real-time explanation of why each result was retrieved.
                </p>
              </div>

              <div className="p-5 bg-white rounded-2xl border border-stone-200 shadow-xs space-y-2.5">
                <h3 className="font-extrabold text-stone-900 text-sm">4. Benchmark Evaluation & Ablation</h3>
                <p className="text-stone-600 leading-relaxed">
                  Tested over 24 ground-truth queries across 6 linguistic categories. An architectural ablation study quantifies the incremental recall and MAP gains provided by normalization, transliteration, and phonetic matching against the raw token baseline.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}