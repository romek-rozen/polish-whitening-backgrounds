# GOTCHAS

Sharp edges that aren't obvious from the code. Read before opening an issue.

## 1. Background granularity must match index granularity

ZCA whitening estimates the covariance Σ of "typical" embeddings in your
target distribution. The background is only valid if the vectors you
whiten at inference time come from the **same distribution** the
background was fitted on.

Concretely: **the unit fed to the embedder during fit must match the
unit fed to the embedder at query/index time.**

| Your retrieval indexes… | Use a background fitted on… |
|---|---|
| Full documents (one vector per doc) | Full documents (`_doc_`) |
| Fixed-size chunks (one vector per 512-token chunk) | Chunks of comparable length (`_chunks_`) |
| Article sections (one vector per H2/H3-scale section) | Sections (`_segments_`) |
| Paragraphs (one vector per blank-line paragraph) | Paragraphs (`_paragraphs_`) |
| Short keyword phrases (1–5 words) | Keyword phrases (`_kw_`) |
| Sentences | Sentences (not shipped — refit) |

Why it matters:

- A doc-level Σ sees one embedding per long, multi-topic text. Topics
  average out → tighter cluster, smaller principal directions.
- A paragraph-level Σ sees many embeddings per document, each more
  topically focused → more spread, different eigenstructure.
- Whitening with the wrong-granularity Σ is still a valid linear
  transform, but it stops being the *isotropisation* you wanted —
  some directions get over-corrected, others under-corrected.

This repo ships **five granularities** side-by-side:

- `*_doc_mrl*` — one embedding per FineWeb-2 / wiki / oasst document,
  truncated to 30k tokens (50 042 docs).
- `*_chunks_mrl*` — one embedding per 512-token chunk with 64-token
  overlap, produced by `lib.chunker` (129 181 chunks from the same
  50 042 docs).
- `*_segments_mrl*` — one embedding per **article section** (up to
  1024 tokens, no overlap), produced by `lib.segmenter` (73 692
  segments from the same 50 042 docs).  Built for internal-linking
  retrieval: match section-of-article-A → sections-of-article-B and
  aggregate per target doc (max / top-k mean over its segments), so
  both sides of the cosine live in the same whitened space.  Don't
  score a `_segments_`-whitened query against `_doc_`-whitened
  targets — different W, incomparable spaces; represent targets by
  their segments instead.
- `*_paragraphs_mrl*` — one embedding per **blank-line paragraph**,
  produced by `lib.paragrapher` (196 759 paragraphs from the same
  50 042 docs): split strictly on blank lines (one paragraph = one
  row, never merged across paragraphs), oversize >512-token paragraphs
  subdivided at sentence boundaries, `merge_tiny(min_chars=120)` folds
  heading-only / one-line fragments forward.  A distinct length band
  between `kw` and `chunks` (median ~490 chars, roughly half a chunk)
  — empirically: whitening paragraph embeddings with a `_chunks_`
  background leaves ~7× the residual anisotropy (top_ev/mean ≈ 13.5)
  versus a proper `_paragraphs_` background (≈ 2.0) on Qwen3-4B, so
  this contract applies to it just as much.
- `*_kw_mrl*` — one embedding per short keyword-like phrase
  (1–5 words), for keyword grouping / clustering.

Pick the variant whose granularity matches what your retriever
indexes. Don't whiten chunk embeddings with a `_doc_` background or
vice versa.

Segments vs chunks — why both exist: chunks are fixed 512-token RAG
windows cut mid-thought with overlap; segments are variable-length,
topically whole sections with **no overlap**.  The segmenter
(`lib.segmenter`) tries markdown headings (`\n## `, `\n### `) before
paragraph boundaries, so on real markdown articles it splits exactly
at H2/H3; on the plain-text fit corpus those separators never match
and it falls through to paragraph packing.  Same splitter at fit and
inference time — the §6 rule applies to segments too
(`merge_tiny` floor is 300 chars there, not 100).

## 2. MRL backgrounds are *not* the same as truncating a full-dim background

If you have a full-dim background `W_2560` and slice it to
`W[:1024, :1024]`, that is **wrong**. The ZCA matrix mixes all input
dimensions into every output dimension; slicing it gives you a partial
projection, not the whitening matrix for the truncated subspace.

The right thing is what `scripts/fit_zca.py --truncate-to 1024` does:
slice each embedding to the first 1024 components, **L2-renormalise
row-wise**, then re-fit μ and Σ from scratch. That's the Σ of the
truncated-and-renormalised distribution — and that is what `loader.py`
expects.

## 3. Zero-vector placeholders are not bugs

If the OpenRouter API repeatedly rejects a document (context overflow,
provider error, etc.), the embed script writes a zero vector at that
row's position so chunk row N always maps to corpus row N. Skipped doc
IDs land in `data/skipped_<slug>.jsonl`.

The ZCA fit treats zero vectors like any other input. In practice
they're <0.1% of the corpus and don't visibly perturb the eigenvalues.
If you want them excluded, mask them in `lib.zca.fit` before the
mean/cov pass — but check the skipped log first to make sure you're not
hiding a real bug.

## 4. Provider routing: per-token price varies wildly across OpenRouter providers

OpenRouter's automatic routing for `qwen/qwen3-embedding-*` can send
requests to providers whose per-token rate differs by several×. If
cost matters, set `--ignore-providers` to exclude the expensive ones
and watch `data/cost_report_<slug>.json` — the per-token rate is the
only place you'll see the difference between routings.

## 5. The 32k context cap is real and pre-flight truncation is required

OpenRouter returns HTTP 200 with an error body when a single doc
exceeds the model context. The script tokenises every doc with the
model's own `tokenizer.json` and truncates to 30 000 tokens before
sending. If you bypass `lib.tokenizer.resolve_and_apply_token_cap`,
expect ~0.1% of long docs to silently turn into zero vectors.

## 6. Paragraph splitter (chunks variant) uses sentence boundaries + overlap

The `_doc_` backgrounds embed **whole documents**. The `_chunks_`
backgrounds embed **chunks** for paragraph-level retrieval, produced
by `lib.chunker`. The chunker respects three rules:

1. **Split on sentence boundaries, never mid-word.** Cutting a word
   in half (`Aminokwa|sy`) destroys the subword tokenisation and
   produces a meaningless embedding for the boundary token. Greedy-
   accumulate full sentences (regex over `. `, `? `, `! ` followed by
   uppercase) until the chunk hits the token target.

2. **Use sliding-window overlap.** Each chunk shares its last
   ~1-2 sentences with the start of the next chunk (target overlap
   ≈ 64 tokens for a 512-token window). If a fact straddles a chunk
   boundary in the source, the overlap guarantees it lives whole in
   at least one chunk — query-time recall stays intact.

3. **Don't try to "understand" lists or tables.** Heuristics that try
   to detect markdown-pipe tables or bullet blocks fire false-positives
   on stray punctuation and false-negatives on real tables that came
   out of trafilatura without pipes. Instead: if a line doesn't end
   with sentence-final punctuation (`.`, `!`, `?`), keep it glued to
   the next line. Lists and tables naturally end up as one chunk that
   way, without any explicit detection.

Defaults that fit Qwen3-Embedding's 32k context but match typical
RAG pipelines: `chunk_size_tokens=512`, `chunk_overlap_tokens=64`.

**Concrete recipe** (shipped, lives in `scripts/lib/chunker.py`):

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter
from lib.tokenizer import load_tokenizer

tok = load_tokenizer("Qwen/Qwen3-Embedding-4B")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=64,
    length_function=lambda s: len(
        tok.encode(s, add_special_tokens=False).ids
    ),
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    keep_separator=True,    # chunks end at sentence-final punctuation
)
chunks = splitter.split_text(doc_text)
```

The recursive-fallback separator list is what makes this robust:
the splitter tries `\n\n` first (paragraph boundary), only descends
to single `\n` if a paragraph is too big, then to sentence-final
punctuation, then to whitespace, then to characters (last resort,
inside a single token-dense word). In practice we never reach the
last fallback on natural prose.

**Critical:** whatever chunker you ship in production, the `_chunks_`
background must use the **same chunker on the corpus**. Different
splitters → different Σ → wrong whitening. See §1.

**Why the post-process merge step exists:** LangChain's
``RecursiveCharacterTextSplitter`` has a wart — when a very short
paragraph (e.g. a Wikipedia section header like "Życiorys" between
two ``\n\n`` separators) sits between two longer ones, the splitter
emits it as a **standalone chunk *without* applying overlap**.
You'd end up with a chunk that's literally just the header — useless
for retrieval, and contributing a degenerate embedding to the
whitening Σ.

``lib.chunker.merge_tiny`` forward-merges any chunk shorter than
100 chars into the next chunk (or, for the last-in-doc, into the
previous chunk).  Net effect: section headers become the **opening
line** of the next paragraph — which matches how authors actually
intended them.  A query like "Maria Skłodowska życiorys" then hits
the chunk that starts with "Życiorys\n\n[biography text...]" and
also contains the answer.

Verified on the `pl_mixed50k` corpus: pre-merge 130 900 chunks with
756 (0.6 %) under 30 chars; post-merge 129 181 chunks, **all ≥ 100 chars**.

## 7. Resume is by chunk file, not by row count

`embed_via_openrouter.py` resumes from the highest `chunk_NNNN.npy`
on disk. If you kill mid-chunk, the partial batch is lost and the
script restarts at the last *completed* chunk. That's fine for a 1000-
row chunk size but expensive at, say, 10 000 — keep `--chunk-size`
modest.
