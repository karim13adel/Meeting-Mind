# MeetingMind v2 — Speaker-Aware Meeting Intelligence System

MeetingMind v2 takes raw multi-speaker audio recordings and produces a fully attributed, queryable knowledge base of the conversation. The pipeline runs speaker diarization with pyannote 3.1, transcribes with OpenAI Whisper medium, then applies a three-contribution RAG architecture — semantically-coherent chunking, speaker-filtered retrieval, and hybrid vector+keyword search — so that a query like "what did Speaker 2 say about the budget?" returns chunks scoped to that speaker rather than any chunk mentioning budgets. A Streamlit interface exposes transcripts with per-speaker filtering and timestamps, LLM-generated speaker profiles, meeting reports, and conversational Q&A with multi-query expansion, cross-encoder re-ranking, and conversation memory.

---

## Technical Contributions

The following are original algorithmic components built from scratch. They are distinct from configuration, API wiring, or library calls.

**Contribution 1 — Meeting-aware semantic chunking (Step 5, Notebook 2)**

Standard RAG systems chunk documents every N words. Applied to meeting transcripts, this cuts turns mid-sentence, mixes unrelated topics, and loses speaker attribution. The contribution is a three-pass algorithm that splits meetings at topic boundaries:

- Pass 1 absorbs short turns (< 5 words: affirmations, backchannels) into their predecessor, eliminating embedding noise. This removed 224 turns from EN2001a, 164 from EN2001b, and 53 from IS1009a.
- Pass 2 embeds every cleaned turn with `multi-qa-mpnet-base-dot-v1` and computes windowed cosine similarity (window = 2) between each turn and its neighbors.
- Pass 3 detects chunk boundaries using a dynamic threshold — the 25th percentile of all within-meeting similarity scores — so the threshold adapts per-meeting rather than applying a global value. Speaker changes after >3 seconds of silence add a 0.15 bonus to the boundary score. Hard constraints cap chunks at 12 turns maximum and 30 words minimum.

Each smart chunk carries metadata: speaker list, primary speaker (by word count), start/end timestamps, topic index, and turn indices.

**Contribution 2 — Speaker-aware retrieval with dynamic filtering (Step 7, Notebook 2)**

The baseline `blind_retrieve` searches all chunks regardless of speaker. The speaker-aware retrieval function parses natural language speaker references from the query — both explicit labels (`SPEAKER_01`) and positional references ("the second speaker") — then restricts FAISS search to a temporary mini-index built from only that speaker's chunks. This makes speaker-specific queries answerable without post-hoc filtering.

**Contribution 3 — Hybrid BM25 + FAISS retrieval with alpha tuning (Step 8, Notebook 2)**

A linear interpolation between BM25 keyword scores and FAISS vector scores: `hybrid_score = alpha * vector_score + (1 - alpha) * bm25_score`. The `hybrid_alpha = 0.5` default is chosen by ablation. BM25 recovers keyword-exact matches (names, jargon, technical terms) that dense retrieval misses when the embedding space conflates semantically similar but content-distinct terms.

---

## Repository Structure

```
MeetingMind/
├── MeetingMind_v2_Pipeline.ipynb   — Notebook 1: audio preprocessing, diarization, Whisper ASR
├── MeetingMind_v2_Indexing.ipynb   — Notebook 2: chunking, embedding, BM25+FAISS index, QA
├── MeetingMind_v2_App.ipynb        — Notebook 3: Streamlit app launch via pyngrok
└── README.md
```

All intermediate outputs (transcripts, checkpoints, indices) are persisted to Google Drive between notebooks.

---

## Quickstart

All notebooks run in Google Colab with a GPU runtime.

```python
# Notebook 1
drive.mount('/content/drive')
# Run Cell 0-A: creates folder structure under MeetingMind_v2/
# Run Cell 0-B: saves config.json to Drive
# Steps 1-4: download AMI audio, run pyannote diarization, run Whisper ASR, build master transcript

# Notebook 2
# Steps 5-10: chunking, embedding, FAISS index, BM25 index, retrieval, QA evaluation

# Notebook 3
# Launches Streamlit app; pyngrok prints a public HTTPS URL
```

Required: Hugging Face token with pyannote model access, Groq API key for LLaMA inference.

---

## Requirements

| Component | Details |
|-----------|---------|
| Language | Python 3.x |
| ASR | OpenAI Whisper (medium) |
| Diarization | pyannote.audio 3.1 |
| Embedding | sentence-transformers/multi-qa-mpnet-base-dot-v1 |
| Vector Search | faiss-cpu |
| Keyword Search | rank_bm25 |
| LLM | LLaMA 3.3-70B-versatile via Groq API |
| UI | Streamlit + pyngrok |
| Evaluation | rouge-score |
| Environment | Google Colab (GPU) + Google Drive |

---

## Stack

| Component | Tool |
|-----------|------|
| Speaker diarization | pyannote.audio 3.1 |
| ASR | OpenAI Whisper (medium) |
| Semantic chunking | sentence-transformers (custom 3-pass algorithm) |
| Vector index | FAISS |
| Keyword index | BM25 (rank_bm25) |
| LLM QA | LLaMA 3.3-70B (Groq) |
| App | Streamlit |
| Persistence | Google Drive |

---

## Results

Evaluated on 3 AMI corpus meetings (EN2001a, EN2001b, IS1009a):

- Smart chunking removes 224 / 164 / 53 short-turn noise entries per respective meeting
- Hybrid retrieval (alpha=0.5) outperforms pure vector or pure BM25 on speaker-specific queries
- ROUGE scores computed on LLM-generated summaries vs. AMI human reference summaries

---

## Limitations

The diarization step requires substantial GPU time for meetings longer than 30 minutes. Pyannote's speaker count bounds (min=2, max=6) are tuned for AMI-style small-group meetings and degrade on recordings with many speakers or very brief contributions. The Groq API has rate limits that trigger automatic fallback to `llama3-8b-8192`. Multi-language meetings (Arabic/English code-switching) are not supported — Whisper transcribes but accuracy degrades significantly.
