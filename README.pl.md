# Polskie tła ZCA whitening dla Qwen3-Embedding (4B & 8B)

🇬🇧 **English:** [README.md](./README.md)

Gotowe artefakty whiteningu (`W_A.npy`, `mu_A.npy`, `eigvals_A.npy`) do
podpięcia w siteFocus / dowolnym pipelinie retrievalu używającym
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B)
lub [`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
na tekstach polskich. Oszczędzasz sobie próbkowanie korpusu, 45k
embeddingów i ZCA SVD — klonujesz, ładujesz, używasz.

Liczba teł w repo: **11**  ·  Licencja: [CC-BY-4.0](LICENSE)

> **Uwaga (2026-06-09):** korpus przebudowany jako **v2** — wiki 22.5k
> + FineWeb-2 PL 22.5k + oasst ~42 = 45 042 dokumentów, tylko akapity
> (≥500 znaków), precyzyjne obcinanie po tokenach pod oknem 32k Qwen3.
> Wcześniejsze `polish_mixed_50k_v1{,_mrl1024,_mrl1536}`,
> `corpus205_n3155`, `polish_smoke_1500` zostały usunięte z `main`.
> Sięgnij do historii gita jeśli ich potrzebujesz.

## Po co whitening?

Współczesne modele embeddingowe (Qwen3 też) produkują wektory
**anizotropowe** — podobieństwo cosinusowe jest skoszone w stronę
kilku dominujących kierunków w przestrzeni, przez co dystans cosinusowy
robi się ciasny: większość par wygląda na "podobne" nawet gdy w
rzeczywistości nie są. Konkretnie na naszym korpusie stosunek
największej wartości własnej kowariancji embeddingów do średniej
to **~50–100×** (vs. ~1× dla idealnego rozkładu izotropowego).

**Transformacja ZCA whitening** przywraca równowagę przestrzeni:

```
x_white = (x - μ) @ W       gdzie  Σ = U S Uᵀ,
                                   W = U · diag(1 / √(S + ε)) · Uᵀ
```

Po jej zastosowaniu każdy kierunek niesie porównywalną wariancję, a
dystans cosinusowy zachowuje się znacznie bliżej teoretycznego ideału.
W retrievalu zwykle przekłada się to na:

- realnie lepsze **recall@k** na trudnych zapytaniach z polisemią /
  klastrami tematycznymi, zwłaszcza przy krótkich query na długie
  dokumenty,
- znacznie czystsze sygnały do **klasteryzacji / deduplikacji** —
  "monokultura top eigenvalue" przestaje sklejać niepowiązanych
  dokumentów,
- naprawę dobrze znanego problemu **"wszystkie cosinusy wyglądają jak
  0.7"**.

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
# ['polish_mixed_50k_v2_qwen3-4b_mrl2560',
#  'polish_mixed_50k_v2_qwen3-4b_mrl1536',  '..._mrl1024', '..._mrl768', '..._mrl512',
#  'polish_mixed_50k_v2_qwen3-8b_mrl4096',
#  'polish_mixed_50k_v2_qwen3-8b_mrl3072',  '..._mrl2048', '..._mrl1024', '..._mrl768', '..._mrl512']

# Dopasuj tło do faktycznie używanej kombinacji (model + slice wymiaru).
bg = load_background("polish_mixed_50k_v2_qwen3-4b_mrl1024")
print(bg.dim, bg.W.shape, bg.mu.shape)
# 1024 (1024, 1024) (1024,)

# Wybielanie batcha L2-znormalizowanych embeddingów Qwen3.
import numpy as np
x = np.random.randn(8, bg.dim).astype("float32")
x /= np.linalg.norm(x, axis=1, keepdims=True)
x_white = bg.apply(x)         # równoważne (x - bg.mu) @ bg.W
```

Jedyną zależnością runtime jest `numpy`. Bez `git lfs`, bez
zewnętrznych pobrań — każdy plik leży wprost w repo.

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
bg = load_background("polish_mixed_50k_v2_qwen3-4b_mrl1024")

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
  do embeddingów 4B obciętych do 1024. `mrl1024` z 8B ma ten sam kształt,
  ale statystyki μ i Σ są zupełnie inne.
- **Transformacja jest dokładna i bezstratna przy pełnym wymiarze** —
  `bg.apply` to obrót + skalowanie per-oś; nie wyrzuca informacji,
  tylko przerozdziela wariancję na osie.

## Które tło wybrać?

| Kiedy | Wybierz | Wymiar |
|---|---|---:|
| Qwen3-Embedding-4B, natywny | `polish_mixed_50k_v1_qwen3-4b_nocap` | 2560 |
| Qwen3-Embedding-8B, natywny | `polish_mixed_50k_v1_qwen3-8b_nocap` | 4096 |

Pełna tabela z `n_fit`, deficytem rangi, stosunkiem największej do
średniej wartości własnej i datą fitu znajduje się w
[`REGISTRY.md`](REGISTRY.md). Te same dane w
[`registry.json`](registry.json) gdy ładujesz programowo.

Potrzebujesz wersji MRL (np. 1024 lub 1536 wymiarów dla 4B)?
Uruchom `scripts/fit_zca.py` na zapisanych chunkach — patrz
[Zbudować od zera](#zbudować-od-zera-lub-dopasować-dla-własnego-modelu)
poniżej.

## Tła MRL

Zarówno Qwen3-Embedding-4B jak i 8B to modele trenowane z Matryoshka
Representation Learning — pierwsze `N < D` komponentów każdego wektora
stanowi sam w sobie poprawny embedding (po L2-renormalizacji). Dla
każdego modelu repo ma osobny refit ZCA dla wszystkich popularnych
`N`, więc whitening zgadza się z tym co Twój pipeline faktycznie
podaje do indeksu przy inferencji:

| Model | Wymiar natywny | Dostępne refity MRL |
|---|---:|---|
| Qwen3-Embedding-4B | 2560 | `mrl{2560, 1536, 1024, 768, 512}` |
| Qwen3-Embedding-8B | 4096 | `mrl{4096, 3072, 2048, 1024, 768, 512}` |

Łącz każde z nich **wyłącznie** z wektorami sliced + renormalised w
ten sam sposób:

```python
x_full = embed("...")                     # (2560,) z Qwen3-4B
x_1024 = x_full[:1024]                    # MRL slice
x_1024 /= np.linalg.norm(x_1024)          # renorm do unit L2
bg = load_background("polish_mixed_50k_v2_qwen3-4b_mrl1024")
x_white = bg.apply(x_1024[None])[0]       # wybielenie w przestrzeni MRL-1024
```

Mieszanie wektorów MRL-1024 z tłem pełnowymiarowym jest niezdefiniowane
— średnie / kowariancje nie są kompatybilne. Podobnie `mrl1024` z tła
4B **nie** jest wymienne z `mrl1024` z tła 8B mimo że kształty się
zgadzają — bazowe statystyki są zupełnie inne.

Potrzebujesz wymiaru którego nie dostarczamy (np. 256, albo 2048 dla
4B)? Refit zajmuje kilka sekund na zapisanych chunkach — przepis w
[Zbudować od zera](#zbudować-od-zera-lub-dopasować-dla-własnego-modelu)
poniżej.

## Pochodzenie danych

Wszystkie tła zostały dopasowane na zbalansowanym miksie polskich
korpusów tekstowych (v2 — usunięto źródła czysto-zdaniowe, mC4
zastąpione przez pre-cleaned FineWeb-2):

| Źródło | Liczba dokumentów | Uwagi |
|---|---:|---|
| Wikipedia PL | 22 500 | [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia) konfiguracja `20231101.pl` |
| FineWeb-2 PL | 22 500 | [`HuggingFaceFW/fineweb-2`](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) konfiguracja `pol_Latn` — polski web crawl wyciągnięty przez trafilatura, filtrowany językowo/jakościowo, dedup minhashem już u źródła |
| OASST PL | ~156 | [`OpenAssistant/oasst1`](https://huggingface.co/datasets/OpenAssistant/oasst1) przefiltrowane `lang == 'pl'` (cel 5 000, w praktyce ~156) |

Wszystkie źródła wymuszają minimum 500 znaków per dokument (akapit,
nie zdanie). Seed = 42, streaming shuffle, deterministycznie.

Wcześniejsze buildy (zachowane w historii gita) zawierały dodatkowo
**KLEJ** (NKJP-NER + DYK + CDSC-R) i używały **mC4** zamiast FineWeb-2.
KLEJ został usunięty bo median długości to 78 znaków — pojedyncze
zdania przesuwają rozkład embeddingów daleko od typowego celu retrievalu
(akapity). mC4 zamieniony bo jego surowy tekst niesie boilerplate (menu,
breadcrumbs, timestampy) z naiwnej ekstrakcji HTML→tekst — i nie da się
tego naprawić downstream (HTML już dawno wyrzucony). FineWeb-2 dostarcza
tekst już wyciągnięty przez [trafilatura](https://trafilatura.readthedocs.io).

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

Próbka mixu jak wyżej (seed=42), embedding każdego dokumentu przez
OpenRouter na `Qwen/Qwen3-Embedding-{4B,8B}`, fit ZCA w dwóch
streamingowych przejściach po chunkach (`μ = E[x]`,
`Σ = E[(x-μ)(x-μ)ᵀ]`), a potem
`W = U · diag(1/√(S + ε)) · Uᵀ` z `SVD(Σ)`, gdzie `ε=1e-6`. Bez GPU;
łączny koszt API: ~$1.2 na oba modele dla mixu 45 156 dokumentów.

Suffix `_nocap` oznacza brak twardego limitu znaków na etapie budowy
korpusu. Per-dokumentowy kontekst egzekwowany jest precyzyjnie na
etapie embed: każdy doc przechodzi przez tokenizer modelu (ten sam
`tokenizer.json` dla 4B i 8B, sha256 `83cdf8c3a34f6886…`) i jest
obcinany do **30 000 tokenów** jeśli trzeba (~2k zapasu pod oknem 32k
Qwen3). Tylko ~25 z 45 156 dokumentów przekracza ten limit; reszta
przechodzi bez zmian. Pełny przepis w następnej sekcji.

## Zbudować od zera (lub dopasować dla własnego modelu)

Katalog `scripts/` zawiera kompletny pipeline który możesz odpalić z
dowolnym kluczem OpenRouter i dla dowolnego modelu embeddującego
wspieranego przez OpenRouter. Wall-time: ~1-3 h na model, koszt API
~$0.5-1 na model dla 45k polskich dokumentów (~38 M tokenów po
$0.01-0.02 / M w zależności od providera kierowanego przez OpenRouter).

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
| `scripts/build_corpus.py` | Próbkuje mix polski (wiki + FineWeb-2 PL + oasst) z seed=42 i progiem 500 znaków na akapit. Zapisuje `data/corpus.parquet`. Default: brak górnego capa. |
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
