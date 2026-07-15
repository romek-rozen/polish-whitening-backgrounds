# scripts/lib/

Reusable helpers behind the thin CLI scripts in [`../`](../). If a
function could be useful from more than one script (or a test), it
lives here. Rules for what belongs here are in [`AGENTS.md`](AGENTS.md).

## Modules

| Module | What it does |
|---|---|
| `zca.py` | The core: `fit()` (ZCA whitening Σ → `W_A`, `mu_A`, `eigvals_A` + diagnostics) and `write_meta()`. |
| `tokenizer.py` | Model ↔ tokenizer mapping (Qwen3 HF tokenizers, OpenAI `tiktoken cl100k_base`), token counting, truncation caps. |
| `chunker.py` | Fixed-size token chunking (recursive splitter, tiny-fragment merge, overlap stripping) for the `chunks` granularity. |
| `paragrapher.py` | Blank-line paragraph splitting for the `paragraphs` granularity. |
| `segmenter.py` | Heading-first section splitting (≤1024 tokens, no overlap) for the `segments` granularity. |
| `openrouter_client.py` | HTTP client for the embeddings API (OpenRouter + OpenAI), adaptive batching, retries. |
| `chunk_store.py` | On-disk embedding store: `chunk_NNNN.npy` files + `manifest.jsonl`, resumable. |
| `dotenv.py` | Minimal `.env` loader (no third-party dependency). |

## Conventions

- Each module does `logger = logging.getLogger(__name__)` — messages
  show up tagged `lib.<module>`. Logging is configured once in the
  calling CLI script's `main()`, never here.
- No secrets in code — API keys arrive via env / `.env`.

See [`AGENTS.md`](AGENTS.md) for the cut rules (what graduates from a
CLI script into `lib/`) and the full contract of each helper.
