# Polskie tła ZCA whitening dla Qwen3-Embedding (4B & 8B)

🇬🇧 **English:** [README.md](./README.md)

Gotowe artefakty whiteningu (`W_A.npy`, `mu_A.npy`, `eigvals_A.npy`) do
podpięcia w siteFocus / dowolnym pipelinie retrievalu używającym
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
lub [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
na tekstach polskich. Oszczędzasz sobie próbkowanie korpusu, 50k
embeddingów i ZCA SVD — klonujesz, ładujesz, używasz.

Licencja: [CC-BY-4.0](LICENSE)

> **Status (2026-06-10):** wszystkie 22 tła gotowe i w repo. Dwa
> modele × dwie granularności × siatka MRL. Korpus to `pl_mixed50k`
> — 22 500 Wikipedia + 27 500 FineWeb-2 PL + 42 wątki oasst =
> **50 042 dokumentów** (akapity ≥500 znaków, ~46 M tokenów).
> Granularność `chunks` to 129 181 chunków po 512 tokenów z
> 64-tokenowym overlapem (RecursiveCharacterTextSplitter + merge
> sub-100-char chunków + strip overlap fragments).
>
> | Model | Granularność | Refity MRL |
> |---|---|---|
> | Qwen3-Embedding-4B | `doc` | `qwen3_4b_pl_mixed50k_doc_mrl{2560, 1536, 1024, 768, 512}` |
> | Qwen3-Embedding-4B | `chunks` | `qwen3_4b_pl_mixed50k_chunks_mrl{2560, 1536, 1024, 768, 512}` |
> | Qwen3-Embedding-8B | `doc` | `qwen3_8b_pl_mixed50k_doc_mrl{4096, 3072, 2048, 1024, 768, 512}` |
> | Qwen3-Embedding-8B | `chunks` | `qwen3_8b_pl_mixed50k_chunks_mrl{4096, 3072, 2048, 1024, 768, 512}` |
>
> Wcześniejsze `polish_mixed_50k_v1{,_mrl1024,_mrl1536}`,
> `corpus205_n3155` i `polish_smoke_1500` zostały wycofane (inny
> korpus, brak tagu granularności w nazwie) — sięgnij do historii
> gita jeśli ich potrzebujesz. Aktualny stan w
> [`REGISTRY.md`](REGISTRY.md).

> ⚠️ **Granularność ma znaczenie.** Warianty `doc` są fitowane na
> **całych dokumentach** (jeden embedding na doc z FineWeb-2 / wiki
> / oasst); warianty `chunks` — na 129 181 chunkach po 512 tokenów
> z 64-tokenowym overlapem. Dopasuj granularność tła do
> granularności tego, co rzeczywiście trzymasz w indeksie
> retrievalowym. Dlaczego mieszanie granularności po cichu psuje
> whitening: [GOTCHAS.md §1](GOTCHAS.md#1-background-granularity-must-match-index-granularity).

## Po co whitening?

Współczesne modele embeddingowe (Qwen3 też) produkują wektory
**anizotropowe** — podobieństwo cosinusowe jest skoszone w stronę
kilku dominujących kierunków w przestrzeni, przez co dystans
cosinusowy robi się ciasny: większość par wygląda na "podobne" nawet
gdy w rzeczywistości nie są. Na tym polskim korpusie stosunek
największej wartości własnej kowariancji embeddingów do średniej
mierzy się w dziesiątkach (vs. ~1× dla idealnego rozkładu
izotropowego).

**Transformacja ZCA whitening** przywraca równowagę przestrzeni:

```
x_white = (x - μ) @ W       gdzie  Σ = U S Uᵀ,
                                   W = U · diag(1 / √(S + ε)) · Uᵀ
```

Po jej zastosowaniu każdy kierunek niesie porównywalną wariancję, a
dystans cosinusowy zachowuje się znacznie bliżej teoretycznego
ideału. W retrievalu zwykle przekłada się to na:

- realnie lepsze **recall@k** na trudnych zapytaniach z polisemią /
  klastrami tematycznymi, zwłaszcza przy krótkich query na długie
  dokumenty,
- znacznie czystsze sygnały do **klasteryzacji / deduplikacji** —
  "monokultura top eigenvalue" przestaje sklejać niepowiązanych
  dokumentów,
- naprawę dobrze znanego problemu **"wszystkie cosinusy wyglądają
  jak 0.7"**.

Robi się to tylko raz na kombinację (model, korpus, język) — stąd
pre-fitting i dystrybucja jako statycznego artefaktu.

## Szybki start

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds
```

```python
from loader import load_background, list_backgrounds

print(list_backgrounds())
# Zwraca 22 nazwy — 4B/8B × doc/chunks × siatka MRL:
# ['qwen3_4b_pl_mixed50k_doc_mrl2560', '..._mrl1536', '..._mrl1024', '..._mrl768', '..._mrl512',
#  'qwen3_4b_pl_mixed50k_chunks_mrl2560', '..._mrl1536', '..._mrl1024', '..._mrl768', '..._mrl512',
#  'qwen3_8b_pl_mixed50k_doc_mrl4096', '..._mrl3072', '..._mrl2048', '..._mrl1024', '..._mrl768', '..._mrl512',
#  'qwen3_8b_pl_mixed50k_chunks_mrl4096', '..._mrl3072', '..._mrl2048', '..._mrl1024', '..._mrl768', '..._mrl512']

# Dopasuj tło do faktycznie używanej kombinacji (model + slice wymiaru).
bg = load_background("qwen3_4b_pl_mixed50k_doc_mrl1024")
print(bg.dim, bg.W.shape, bg.mu.shape)
# 1024 (1024, 1024) (1024,)

# Wybielanie batcha L2-znormalizowanych embeddingów Qwen3.
import numpy as np
x = np.random.randn(8, bg.dim).astype("float32")
x /= np.linalg.norm(x, axis=1, keepdims=True)
x_white = bg.apply(x)         # równoważne (x - bg.mu) @ bg.W
```

Jedyną zależnością runtime jest `numpy`. Bez `git lfs`, bez
zewnętrznych pobrań — wszystkie 22 tła leżą wprost w repo.

## End-to-end: użycie w pipelinie retrievalu

Tak wygląda realny przepływ cosinusowego retrievalu w produkcji na
indeksie Qwen3-4B. Krok whiteningu wpada **zaraz po L2-renormalizacji,
przed dot-productem** — reszta pipeline'u nie zmienia się ani trochę.

```python
import numpy as np
from loader import load_background
# Cokolwiek już używasz do Qwen3 — lokalnie, vLLM, OpenRouter itp.
from your_pipeline import embed_qwen3_4b

# 1. Załaduj raz na starcie.
bg = load_background("qwen3_4b_pl_mixed50k_doc_mrl1024")

def encode(texts):
    """Embed → MRL slice → L2 renorm → ZCA whiten."""
    x = embed_qwen3_4b(texts)             # (n, 2560) float32
    x = x[:, :bg.dim]                     # MRL slice do 1024
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return bg.apply(x)                    # (n, 1024) po whiteningu

# 2. Zaindeksuj dokumenty raz.
doc_vecs = encode(documents)              # (N, 1024)

# 3. Przy zapytaniu enkoduj query tak samo.
q_vec = encode([query])                   # (1, 1024)
scores = q_vec @ doc_vecs.T               # (1, N) cosine, po whiteningu
topk = np.argpartition(-scores[0], 10)[:10]
```

Co jest ważne w tym wzorcu:

- **Wybielaj obie strony identycznie** — wektory query i wektory
  dokumentów muszą przejść przez ten sam `bg.apply`. Mieszanie
  wybielonych i surowych daje bezsensowne wyniki.
- **Para (model, wymiar, tło)** — `mrl1024` z tła 4B pasuje wyłącznie
  do embeddingów 4B obciętych do 1024. `mrl1024` z 8B ma ten sam
  kształt ale statystyki μ i Σ są zupełnie inne — nie są wymienne.
- **Transformacja jest dokładna i bezstratna** — `bg.apply` to obrót
  + skalowanie per-oś; nie wyrzuca informacji, tylko przerozdziela
  wariancję na osie.

## Tła MRL

Zarówno Qwen3-Embedding-4B jak i 8B to modele trenowane z Matryoshka
Representation Learning — pierwsze `N < D` komponentów każdego wektora
stanowi sam w sobie poprawny embedding (po L2-renormalizacji). Dla
każdego modelu repo dostarcza osobny refit ZCA dla każdego popularnego
`N`, więc whitening zgadza się z tym co Twój pipeline faktycznie
podaje do indeksu przy inferencji:

| Model | Wymiar natywny | Dostępne refity MRL |
|---|---:|---|
| Qwen3-Embedding-4B | 2560 | `mrl{2560, 1536, 1024, 768, 512}` |
| Qwen3-Embedding-8B | 4096 | `mrl{4096, 3072, 2048, 1024, 768, 512}` |

Lista wymiarów dla 8B trzyma się kanonicznych targetów MRL Qwen3
(potęgi dwójki plus 768 i 3072); off-grid rozmiary jak 2560 / 1536 są
pominięte dla 8B bo model nie był trenowany MRL przy tych wymiarach —
slice matematycznie działa, ale recall byłby gorszy niż przy
wytrenowanych wymiarach.

Łącz każde z nich **wyłącznie** z wektorami sliced + renormalised w
ten sam sposób:

```python
x_full = embed("...")                     # (2560,) z Qwen3-4B
x_1024 = x_full[:1024]                    # MRL slice
x_1024 /= np.linalg.norm(x_1024)          # renorm do unit L2
bg = load_background("qwen3_4b_pl_mixed50k_doc_mrl1024")
x_white = bg.apply(x_1024[None])[0]       # wybielenie w przestrzeni MRL-1024
```

Potrzebujesz wymiaru którego nie dostarczamy (np. 256 albo 2048 dla
4B)? Refit zajmuje kilka sekund na zapisanych chunkach — przepis w
[Zbudować od zera](#zbudować-od-zera-lub-dopasować-dla-własnego-modelu)
poniżej.

## Pochodzenie danych

Korpus v2 to zbalansowany miks polskich tekstów (źródła czysto-zdaniowe
KLEJ zastąpione większą ilością akapitów, zaszumiony mC4 zamieniony na
pre-cleaned FineWeb-2):

| Źródło | Liczba dokumentów | Uwagi |
|---|---:|---|
| Wikipedia PL | 22 500 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) konfiguracja `20231101.pl` |
| FineWeb-2 PL | 22 500 | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) konfiguracja `pol_Latn` — polski web crawl wyciągnięty przez trafilatura, filtrowany językowo/jakościowo, dedup minhashem już u źródła |
| OASST PL | ~42 | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) przefiltrowane `lang == 'pl'` (cel 5 000; ~42 dokumentów przebija próg 500 znaków w publicznym dumpcie) |

Faktyczny korpus `pl_mixed50k`: **50 042 dokumentów, ~46 M tokenów,
fingerprint `6e9e965ffbb6dbe6…`**. Wszystkie źródła wymuszają
minimum 500 znaków per dokument (akapit, nie zdanie). Seed = 42,
streaming shuffle, deterministycznie. Dla wariantów `chunks` ten
sam korpus jest cięty przez `lib.chunker`
(RecursiveCharacterTextSplitter, chunk 512 tok / overlap 64 tok,
merge sub-100-char chunków forward, strip overlap fragments) i
daje **129 181 chunków** (~47.5 M tokenów po embed).

Wcześniejsze buildy (zachowane w historii gita) zawierały dodatkowo
**KLEJ** (NKJP-NER + DYK + CDSC-R) i używały **mC4** zamiast
FineWeb-2. KLEJ został usunięty bo median długości to 78 znaków —
pojedyncze zdania przesuwają rozkład embeddingów daleko od typowego
celu retrievalu (akapity). mC4 zamieniony bo jego surowy tekst niesie
boilerplate (menu, breadcrumbs, timestampy) z naiwnej ekstrakcji
HTML→tekst — i nie da się tego naprawić downstream (HTML już dawno
wyrzucony). FineWeb-2 dostarcza tekst już wyciągnięty przez
[trafilatura](https://trafilatura.readthedocs.io).

Każdy `*.meta.json` zapisuje dokładne `sample_size_actual`,
`corpus_fingerprint_sha256`, seed i diagnostyczne wartości własne.

## Struktura repo

```
backgrounds/<name>/                   # 22 katalogi
  W_A.npy           # (dim, dim) float32  — zastosowanie: (x - mu) @ W
  mu_A.npy          # (dim,)    float32
  eigvals_A.npy     # (dim,)    float32   — diagnostyka, niepotrzebne przy apply
  <name>.meta.json  # pochodzenie + diagnostyka
REGISTRY.md         # czytelny indeks, autogenerowany
registry.json       # to samo, w wersji do parsowania
loader.py           # loader tylko numpy (patrz Szybki start)
scripts/            # pipeline korpus + embed + fit + index
LICENSE             # CC-BY-4.0
README.md           # wersja angielska
README.pl.md        # ten plik
```

## Jak zostały zbudowane

Próbka mixu jak wyżej (seed=42), embedding każdego dokumentu (oraz
każdego chunka) przez OpenRouter na `Qwen/Qwen3-Embedding-4B` i
`Qwen/Qwen3-Embedding-8B`, fit ZCA w dwóch streamingowych
przejściach po chunkach (`μ = E[x]`, `Σ = E[(x-μ)(x-μ)ᵀ]`), a potem
`W = U · diag(1/√(S + ε)) · Uᵀ` z `SVD(Σ)`, gdzie `ε=1e-6`. Bez
GPU. Łączny koszt OpenRouter dla wszystkich 22 teł: **~$2.77**
(4B doc $0.92, 8B doc $0.43, 4B chunks $0.95, 8B chunks $0.48 —
routing OpenRoutera z `--ignore-providers siliconflow`, bo
SiliconFlow jest 4× droższy).

Per-dokumentowy kontekst egzekwowany jest precyzyjnie na etapie
embed: każdy doc przechodzi przez tokenizer modelu (pobierany z HF —
ten sam `tokenizer.json` dla 4B i 8B, byte-identyczny) i jest
obcinany do **30 000 tokenów** jeśli trzeba (~2k zapasu pod oknem
32k Qwen3). Dla wariantu `chunks` chunker robi twardy split przed
embed (target 512 tok), więc cap nigdy nie pyka.

Te same chunki embedów są potem fitowane pięć razy (4B) i sześć
razy (8B) na każdą granularność — raz na każdy wymiar MRL — przez
**refit od zera**: slice chunka do `N` kolumn, L2-renormalizacja
wierszowa, ponowne obliczenie μ i Σ, świeże SVD. (Nie liczymy
pełnego `W` raz i nie slice'ujemy go — to dałoby złe statystyki.)
Cała siatka MRL dla jednej granularności jednego modelu zajmuje
poniżej dwóch minut na CPU po zakończeniu embed.

Tabela diagnostyki (`top_ev_ratio_pre` / `rank_deficient_eigvals`,
od najwyższego do najniższego wymiaru MRL):

| Tło | mrl4096 | mrl3072 | mrl2560 | mrl2048 | mrl1536 | mrl1024 | mrl768 | mrl512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 4B doc    |   —    |   —    | 91.7/17 |   —    | 59.4/2 | 42.6/2 | 33.2/2 | 24.6/2 |
| 4B chunks |   —    |   —    | 86.1/17 |   —    | 55.6/3 | 40.3/2 | 31.5/2 | 23.9/2 |
| 8B doc    | 157.6/24 | 119.0/3 |   —    | 80.5/1 |   —    | 38.8/0 | 29.4/0 | 20.6/0 |
| 8B chunks | 153.9/21 | 117.1/3 |   —    | 79.1/1 |   —    | 38.7/0 | 28.7/0 | 20.4/0 |

Chunki są nieznacznie mniej anizotropowe niż dokumenty przy tym
samym wymiarze (np. 86.1 vs 91.7 dla 4B mrl2560), bo 129k chunków
próbkuje przestrzeń embedding bardziej równomiernie niż 50k całych
dokumentów.

## Zbudować od zera (lub dopasować dla własnego modelu)

Katalog `scripts/` zawiera kompletny pipeline który możesz odpalić z
dowolnym kluczem OpenRouter i dla dowolnego modelu embeddującego
wspieranego przez OpenRouter. Wall-time: ~1-3 h na model na
granularność, koszt API ~$0.4-1 na model dla 50k polskich
dokumentów / 129k chunków (~46-48 M tokenów po $0.01-0.02 / M w
zależności od providera kierowanego przez OpenRouter).

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds

# 1. Zainstaluj minimalne zależności (numpy + pyarrow + datasets + requests + tokenizers + trafilatura).
pip install -r requirements.txt

# 2. Podaj swój klucz OpenRouter (https://openrouter.ai/keys).
cp .env.example .env
$EDITOR .env             # wklej OPENROUTER_API_KEY=sk-or-...

# 3. End-to-end: korpus(_chunks) → embed (4B + 8B) → fit (22 tła) → index.
bash scripts/run_full.sh
```

Co robi każdy skrypt:

| Skrypt | Zastosowanie |
|---|---|
| `scripts/build_corpus.py` | Próbkuje mix polski (wiki + FineWeb-2 PL + oasst) z seed=42 i progiem 500 znaków na akapit. Zapisuje `data/corpus.parquet`. Default: brak górnego capa. |
| `scripts/build_corpus_chunks.py` | Tnie `data/corpus.parquet` przez `lib.chunker` (512 tok / 64 tok overlap, merge sub-100-char, strip overlap fragments). Zapisuje `data/corpus_chunks.parquet` (129 181 chunków). |
| `scripts/embed_via_openrouter.py` | Embedduje `corpus.parquet` przez OpenRouter. Wstępne, precyzyjne obcinanie po tokenach pod okno kontekstu modelu (domyślnie 30 000 tokenów, tokenizer Qwen3 pobierany z HF — zmiana przez `--max-tokens-per-doc` i `--tokenizer-repo`). Adaptacyjny batch (start 16, połowa przy 429/5xx, rośnie po seriach sukcesów). Idempotentny: resume z najwyższego istniejącego chunka. Pisze `data/chunks_<slug>/*.npy` plus per-call `cost_report_<slug>.json`. |
| `scripts/fit_zca.py` | Dwa streamingowe pass-y (μ, Σ) po chunkach + SVD. Opcjonalne `--truncate-to N` obcina każdy chunk do `N` kolumn i ponownie renormalizuje przed fitem, do refitów MRL. Pisze `backgrounds/<name>/{W_A.npy, mu_A.npy, eigvals_A.npy, *.meta.json}`. |
| `scripts/index_backgrounds.py` | Regeneruje `REGISTRY.md` + `registry.json`. Wywoływane przez `run_full.sh`. |
| `scripts/run_full.sh` | Orchestrator: korpus → embed na każdy model → fit przy każdym wymiarze z `DIMS_<MODEL>` → index. Idempotentny — bezpieczny do ponownego uruchomienia. |

`data/` jest w `.gitignore` (korpus + chunki są odtwarzalne). Tylko
finalne artefakty `backgrounds/<name>/` trafiają do repo.

Aby dopasować tylko jeden model:

```bash
MODELS="qwen/qwen3-embedding-8b" bash scripts/run_full.sh
```

Aby zmienić listę wymiarów MRL dla modelu (default: 4B = 2560/1536/1024/768/512,
8B = 4096/3072/2048/1024/768/512):

```bash
DIMS_4B="2560 1024" bash scripts/run_full.sh   # tylko dwa fity dla 4B
```

Aby zaostrzyć lub poluzować limit tokenów per-doc w kroku embed:

```bash
python scripts/embed_via_openrouter.py \
  --model qwen/qwen3-embedding-4b \
  --max-tokens-per-doc 28000
```

Ustaw `--max-tokens-per-doc 0` żeby wyłączyć limit; dokumenty
przekraczające kontekst modelu wywołają wtedy HTTP 200 + body z
błędem od providera i zostaną pominięte (z zero-wektorem jako
placeholderem — żeby wiersz N w chunku dalej odpowiadał wierszowi N
w korpusie).

## Licencja

[CC-BY-4.0](LICENSE). Darmowe użycie, dzielenie się i adaptacja przy
zachowaniu atrybucji. Bez gwarancji.

## Cytowanie

Jeżeli korzystasz z tych teł w publikacji, prosimy zacytować
Qwen3-Embedding oraz odesłać do tego repo, żeby inni mogli też je
znaleźć:

```
@misc{polish-whitening-backgrounds,
  author = {Rozenberger, Roman},
  title  = {Polish ZCA whitening backgrounds for Qwen3-Embedding (4B & 8B)},
  year   = {2026},
  url    = {https://github.com/romek-rozen/polish-whitening-backgrounds}
}
```
