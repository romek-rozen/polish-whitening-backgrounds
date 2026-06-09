# Polskie tła ZCA whitening dla Qwen3-Embedding-4B

🇬🇧 **English:** [README.md](./README.md)

Gotowe artefakty whiteningu (`W_A.npy`, `mu_A.npy`, `eigvals_A.npy`) do
podpięcia w siteFocus / dowolnym pipelinie retrievalu używającym
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
na tekstach polskich. Oszczędzasz sobie ~godziny próbkowania korpusu,
50k embeddingów i ZCA SVD — klonujesz, ładujesz, używasz.

Liczba teł w repo: **5**  ·  Licencja: [CC-BY-4.0](LICENSE)

## Szybki start

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds
```

```python
from loader import load_background, list_backgrounds

print(list_backgrounds())
# ['corpus205_n3155', 'polish_mixed_50k_v1', 'polish_mixed_50k_v1_mrl1024',
#  'polish_mixed_50k_v1_mrl1536', 'polish_smoke_1500']

bg = load_background("polish_mixed_50k_v1_mrl1024")
print(bg.dim, bg.W.shape, bg.mu.shape)
# 1024 (1024, 1024) (1024,)

# Wybielanie batcha L2-znormalizowanych embeddingów Qwen3 (obciętych do bg.dim jeśli MRL).
import numpy as np
x = np.random.randn(8, bg.dim).astype("float32")
x /= np.linalg.norm(x, axis=1, keepdims=True)
x_white = bg.apply(x)         # równoważne (x - bg.mu) @ bg.W
```

Jedyną zależnością runtime jest `numpy`. Bez `git lfs`, bez
zewnętrznych pobrań — każdy plik leży wprost w repo (największy
~25 MB, łącznie ~88 MB).

## Które tło wybrać?

| Kiedy | Wybierz |
|---|---|
| Produkcja, pełny wymiar Qwen3 (2560) | `polish_mixed_50k_v1` |
| MRL obcięte do 1024 wymiarów | `polish_mixed_50k_v1_mrl1024` |
| MRL obcięte do 1536 wymiarów | `polish_mixed_50k_v1_mrl1536` |
| Smoke / unit-testy | `polish_smoke_1500` (**NIE** używaj na produkcji — niedostateczny rank) |
| Bootstrap (legacy) | `corpus205_n3155` — zostawione do reprodukcji starych runów |

Pełna tabela z `n_fit`, deficytem rangi, stosunkiem największej do
średniej wartości własnej i datą fitu znajduje się w
[`REGISTRY.md`](REGISTRY.md). Te same dane w
[`registry.json`](registry.json) gdy ładujesz programowo.

## Co to jest tło MRL-truncated?

[`Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
to model trenowany z Matryoshka Representation Learning — pierwsze
`N < 2560` komponentów każdego wektora stanowi sam w sobie poprawny
embedding (po L2-renormalizacji). Tła `_mrlN` w tym repo to refit ZCA
na takich obciętych + renormalizowanych wektorach, więc transformacja
whiteningu zgadza się z tym co Twój pipeline faktycznie widzi przy
inferencji. Łącz je **wyłącznie** z wektorami sliced + renormalised w
ten sam sposób:

```python
x_full = embed("...")                     # (2560,) z Qwen3
x_1024 = x_full[:1024]
x_1024 /= np.linalg.norm(x_1024)
bg = load_background("polish_mixed_50k_v1_mrl1024")
x_white = bg.apply(x_1024[None])[0]       # wybielenie w przestrzeni MRL-1024
```

Mieszanie wektorów MRL-1024 z tłem 2560-D (`polish_mixed_50k_v1`) jest
niezdefiniowane — średnie / kowariancje nie są kompatybilne.

## Pochodzenie danych

Wszystkie tła zostały dopasowane na zbalansowanym miksie polskich
korpusów tekstowych:

| Źródło | Liczba dokumentów | Uwagi |
|---|---:|---|
| Wikipedia PL | 20 000 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) konfiguracja `20231101.pl` |
| mC4 PL | 20 000 | [`allenai/c4`](https://huggingface.co/datasets/allenai/c4) konfiguracja `pl` |
| KLEJ | 5 000 | podzbiory NKJP-NER, DYK, CDSC-R |
| OASST PL | 156 | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) przefiltrowane `lang == 'pl'` |

Każdy `*.meta.json` zapisuje dokładne `sample_size_actual`,
`corpus_fingerprint_sha256`, seed i diagnostyczne wartości własne.

## Struktura repo

```
backgrounds/<name>/
  W_A.npy           # (dim, dim) float32  — zastosowanie: (x - mu) @ W
  mu_A.npy          # (dim,)    float32
  eigvals_A.npy     # (dim,)    float32   — diagnostyka, niepotrzebne przy apply
  <name>.meta.json  # pochodzenie + diagnostyka
REGISTRY.md         # czytelny indeks
registry.json       # to samo, w wersji do parsowania
loader.py           # loader (tylko numpy, ~60 linii)
LICENSE             # CC-BY-4.0
README.md           # wersja angielska
README.pl.md        # ten plik
```

## Jak zostały zbudowane

`polish_mixed_50k_v1` (rodzic 2560-D) został dopasowany od zera:
próbka mixu jak wyżej (seed=42), embedding każdego dokumentu modelem
Qwen3 przy `max_chars_per_doc=1800`, fit ZCA w dwóch przejściach po
chunkach (`μ = E[x]`, `Σ = E[(x-μ)(x-μ)ᵀ]`), a potem
`W = U · diag(1/√(S + ε)) · Uᵀ` z `SVD(Σ)`, gdzie `ε=1e-6`.

Tła pochodne `_mrl*` zostały dofitowane w sekundach z zapisanych
chunków rodzica — bez ponownego embedowania. Algorytm: weź każdy chunk,
obetnij do pierwszych `N < 2560` kolumn, L2-renormalizuj wierszowo,
ponownie dopasuj ZCA na obciętym zbiorze. Wynik jest deterministyczny
dla zadanego rodzica.

Tła `_qwen3-*-nocap` zostały zbudowane przez API OpenRouter (bez
potrzeby lokalnego GPU) i **bez** capa 1800 znaków — przepis w
następnej sekcji. Stosują precyzyjne obcinanie po tokenach (domyślnie
30 000 tokenów, ~2k zapasu pod oknem 32k Qwen3) liczone tokenizerem
modelu pobranym z HuggingFace. W 45 156-dokumentowym miksie tylko ~25
dokumentów przekracza ten limit; reszta przechodzi bez zmian.

## Zbudować od zera (lub dopasować dla własnego modelu)

Katalog `scripts/` zawiera kompletny pipeline który możesz odpalić z
dowolnym kluczem OpenRouter i dla dowolnego modelu embeddującego
wspieranego przez OpenRouter. Wall-time: ~2-4 h na model, koszt API
~$2-5 dla 45k polskich dokumentów (zależnie od rozkładu długości).

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds

# 1. Zainstaluj minimalne zależności
pip install -r requirements.txt

# 2. Podaj swój klucz OpenRouter (https://openrouter.ai/keys)
cp .env.example .env
$EDITOR .env             # wklej OPENROUTER_API_KEY=sk-or-...

# 3. End-to-end: korpus → embed (4B + 8B) → fit → index
bash scripts/run_full.sh
```

Co robi każdy skrypt:

| Skrypt | Zastosowanie |
|---|---|
| `scripts/build_corpus.py` | Próbkuje mix polski (wiki + mc4 + klej + oasst) z seed=42. Zapisuje `data/corpus.parquet`. Default: brak capa. |
| `scripts/embed_via_openrouter.py` | Embedduje `corpus.parquet` przez OpenRouter. Wstępne, precyzyjne obcinanie po tokenach pod okno kontekstu modelu (domyślnie 30 000 tokenów, tokenizer Qwen3 pobierany z HF — zmiana przez `--max-tokens-per-doc` i `--tokenizer-repo`). Adaptacyjny batch (start 16, połowa przy 429/5xx, rośnie po seriach sukcesów). Idempotentny: resume z najwyższego istniejącego chunka. Pisze `data/chunks_<slug>/*.npy` plus per-call `cost_report_<slug>.json`. |
| `scripts/fit_zca.py` | Dwa streamingowe pass-y (μ, Σ) po chunkach + SVD. Pisze `backgrounds/<name>/{W_A.npy, mu_A.npy, eigvals_A.npy, *.meta.json}`. |
| `scripts/index_backgrounds.py` | Regeneruje `REGISTRY.md` + `registry.json`. Wywoływane przez `run_full.sh`. |
| `scripts/run_full.sh` | Orchestrator. Idempotentny — bezpieczny do ponownego uruchomienia. |

`data/` jest w `.gitignore` (korpus + chunki są odtwarzalne). Tylko
finalne artefakty `backgrounds/<name>/` trafiają do repo.

Aby dopasować tylko jeden model:

```bash
MODELS="qwen/qwen3-embedding-8b" bash scripts/run_full.sh
```

Aby zachować cap na poziomie korpusu (np. dla repro starszego 1800-znakowego
buildu) — wartość trafia do `build_corpus.py`:

```bash
MAX_CHARS=1800 NAME_PREFIX=polish_mixed_50k_cap1800 bash scripts/run_full.sh
```

Aby zaostrzyć lub poluzować limit tokenów per-doc w kroku embed (domyślnie
30 000, ~2k zapasu pod oknem 32k Qwen3):

```bash
python scripts/embed_via_openrouter.py \
  --model qwen/qwen3-embedding-4b \
  --max-tokens-per-doc 28000
```

Ustaw `--max-tokens-per-doc 0` żeby wyłączyć limit; dokumenty
przekraczające kontekst modelu wywołają wtedy HTTP 200 + body z błędem
od providera i zostaną pominięte (z zero-wektorem jako placeholderem —
żeby wiersz N w chunku dalej odpowiadał wierszowi N w korpusie).

## Licencja

[CC-BY-4.0](LICENSE). Darmowe użycie, dzielenie się i adaptacja przy
zachowaniu atrybucji. Bez gwarancji.

## Cytowanie

Jeżeli korzystasz z tych teł w publikacji, prosimy zacytować model
Qwen3-Embedding-4B oraz odesłać do tego repo, żeby inni mogli też je
znaleźć:

```
@misc{polish-whitening-backgrounds,
  author = {Rozenberger, Roman},
  title  = {Polish ZCA whitening backgrounds for Qwen3-Embedding-4B},
  year   = {2026},
  url    = {https://github.com/romek-rozen/polish-whitening-backgrounds}
}
```
