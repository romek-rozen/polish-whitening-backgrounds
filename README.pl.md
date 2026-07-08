# Polskie tła ZCA whitening dla Qwen3-Embedding i OpenAI text-embedding-3

🇬🇧 **English:** [README.md](./README.md)

Gotowe artefakty whiteningu (`W_A.npy`, `mu_A.npy`, `eigvals_A.npy`) do
podpięcia w siteFocus / dowolnym pipelinie retrievalu lub klastrowania
używającym
[`Qwen/Qwen3-Embedding-4B`](https://huggingface.co/Qwen/Qwen3-Embedding-4B),
[`Qwen/Qwen3-Embedding-8B`](https://huggingface.co/Qwen/Qwen3-Embedding-8B),
`text-embedding-3-small` lub `text-embedding-3-large`
na tekstach polskich. Oszczędzasz sobie próbkowanie korpusu, 50k
embeddingów i ZCA SVD — klonujesz, ładujesz, używasz.

Licencja: [CC-BY-4.0](LICENSE)

> **Status (2026-07-08):** **103 tła w repo** — cztery modele × do
> pięciu granularności × pełna siatka MRL/dimensions. Korpus to
> `pl_mixed50k` — 22 500 Wikipedia + 27 500 FineWeb-2 PL + 42 wątki
> oasst = **50 042 dokumentów** (akapity ≥500 znaków, ~46 M tokenów).
> Granularność `chunks` to 129 181 chunków po 512 tokenów z
> 64-tokenowym overlapem (`lib.chunker`). Granularność `segments`
> tnie te same dokumenty na **73 692 sekcje artykułów** po maksymalnie
> 1024 tokeny, **bez overlapu** (`lib.segmenter`) — fitowana pod
> retrieval do **linkowania wewnętrznego**, gdzie dopasowujesz sekcję
> artykułu A do sekcji kandydujących artykułów docelowych.
> Granularność `paragraphs` tnie te same dokumenty na **196 759
> akapitów** wg pustych linii (`lib.paragrapher`, mediana ~490
> znaków) — najmniejsza jednostka strukturalna powyżej zdania,
> odrębne pasmo długości między `kw` a `chunks` (mniej więcej pół
> chunka). Granularność `kw` to
> **50 000 polskich fraz keyword-podobnych** (1–5 słów) wydobytych z
> tego samego korpusu — fitowana pod **grupowanie / klastrowanie
> krótkich fraz wyszukiwania** (np. listy słów kluczowych Google
> Ads), gdzie tła dokumentowe po cichu nie pasują.
>
> | Model | Granularności | Refity MRL |
> |---|---|---|
> | Qwen3-Embedding-4B | `doc`, `chunks`, `segments`, `kw`, `paragraphs` | `qwen3_4b_pl_mixed50k_{doc,chunks,segments,kw,paragraphs}_mrl{2560, 1536, 1024, 768, 512}` |
> | Qwen3-Embedding-8B | `doc`, `chunks`, `segments`, `kw`, `paragraphs` | `qwen3_8b_pl_mixed50k_{doc,chunks,segments,kw,paragraphs}_mrl{4096, 3072, 2048, 1024, 768, 512}` |
> | text-embedding-3-small | `doc`, `chunks`, `kw`, `paragraphs` | `te3small_pl_mixed50k_{doc,chunks,kw,paragraphs}_mrl{1536, 1024, 768, 512, 256}` |
> | text-embedding-3-large | `doc`, `chunks`, `kw`, `paragraphs` | `te3large_pl_mixed50k_{doc,chunks,kw,paragraphs}_mrl{3072, 2048, 1536, 1024, 768, 512, 256}` |
>
> Wcześniejsze `polish_mixed_50k_v1{,_mrl1024,_mrl1536}`,
> `corpus205_n3155` i `polish_smoke_1500` zostały wycofane (inny
> korpus, brak tagu granularności w nazwie) — sięgnij do historii
> gita jeśli ich potrzebujesz. Aktualny stan w
> [`REGISTRY.md`](REGISTRY.md).

> ⚠️ **Granularność ma znaczenie.** Warianty `doc` są fitowane na
> **całych dokumentach** (jeden embedding na doc z FineWeb-2 / wiki
> / oasst); warianty `chunks` — na 129 181 chunkach po 512 tokenów
> z 64-tokenowym overlapem; warianty `segments` — na sekcjach
> artykułów (≤1024 tokeny, bez overlapu); warianty `paragraphs` — na
> akapitach wg pustych linii (mediana ~490 znaków, odrębne pasmo
> między `kw` a `chunks`); warianty `kw` — na 50 000
> krótkich fraz (1–5 słów). Dopasuj granularność tła do granularności tego, co
> rzeczywiście trzymasz w indeksie / klastrujesz. Dlaczego mieszanie
> granularności po cichu psuje whitening:
> [GOTCHAS.md §1](GOTCHAS.md#1-background-granularity-must-match-index-granularity).

## Po co whitening?

Współczesne modele embeddingowe (Qwen3 też) produkują wektory
**anizotropowe** — podobieństwo cosinusowe jest skoszone w stronę
kilku dominujących kierunków w przestrzeni, przez co dystans
cosinusowy robi się ciasny: większość par wygląda na "podobne" nawet
gdy w rzeczywistości nie są. Na tym polskim korpusie stosunek
największej wartości własnej kowariancji embeddingów do średniej
mierzy się w dziesiątkach (vs. ~1× dla idealnego rozkładu
izotropowego) — a dla **krótkich fraz keywordowych jest znacznie
gorzej**: 81× dla Qwen3-4B, 150× dla Qwen3-8B, 40× dla
text-embedding-3-small. To jest dokładnie ten mechanizm, przez który
"każdy keyword wygląda podobnie do każdego" i grupowanie słów
kluczowych się sypie.

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
# Zwraca 103 nazwy — 4 modele × {doc, chunks, segments, kw, paragraphs} × siatka MRL, np.:
# ['qwen3_4b_pl_mixed50k_doc_mrl2560',  … , 'qwen3_4b_pl_mixed50k_segments_mrl512',
#  'qwen3_8b_pl_mixed50k_doc_mrl4096',  … , 'qwen3_8b_pl_mixed50k_segments_mrl512',
#  'te3small_pl_mixed50k_doc_mrl1536',  … , 'te3small_pl_mixed50k_kw_mrl256',
#  'te3large_pl_mixed50k_doc_mrl3072',  … , 'te3large_pl_mixed50k_kw_mrl256']

# Dopasuj tło do faktycznie używanej kombinacji (model + granularność + slice wymiaru).
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
zewnętrznych pobrań — wszystkie 103 tła leżą wprost w repo.

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

## Linkowanie wewnętrzne (granularność `segments`)

Tła `_segments_` są fitowane na **sekcjach artykułów** — do 1024
tokenów, cięte na nagłówkach markdown / granicach akapitów, bez
overlapu — to naturalna jednostka dla **sugestii linkowania
wewnętrznego**: z której sekcji artykułu A powinienem linkować i do
którego artykułu B?

Kluczowe ograniczenie: obie strony cosinusa muszą być wybielone tym
**samym** tłem. Więc nie porównuj segmentu z wektorem `_doc_` —
reprezentuj artykuły *docelowe* również przez ich sekcje, dopasowuj
segment→segment i agreguj per artykuł docelowy:

```python
import numpy as np
from loader import load_background
from scripts.lib.segmenter import make_segmenter, segment_text

bg = load_background("qwen3_4b_pl_mixed50k_segments_mrl1024")
splitter = make_segmenter("qwen/qwen3-embedding-4b")   # ten sam splitter co przy fitcie

def encode_segments(article_text):
    segs = segment_text(splitter, article_text)
    x = embed_qwen3_4b(segs)[:, :bg.dim]              # MRL slice
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return segs, bg.apply(x)                          # (n_seg, 1024) po whiteningu

# Zaindeksuj każdy kandydujący artykuł docelowy po jego segmentach (raz).
targets = {url: encode_segments(text) for url, text in site_articles.items()}

# Dla każdej sekcji edytowanego artykułu wyranguj cele linków.
src_segs, src_vecs = encode_segments(new_article)
for i, seg in enumerate(src_segs):
    scores = {url: float((src_vecs[i] @ vecs.T).max())   # najlepsze dopasowanie sekcji
              for url, (_, vecs) in targets.items()}
    best = max(scores, key=scores.get)
    print(f"sekcja {i} → linkuj do {best} ({scores[best]:.3f})")
```

`max()` po segmentach celu mówi Ci nie tylko *który* artykuł
zalinkować, ale i *która jego sekcja* faktycznie pasuje — przydatne
przy doborze anchor tekstu. Użyj średniej z top-k zamiast max, jeśli
chcesz faworyzować cele pasujące szeroko, a nie na jednej sekcji.

Jeśli Twój CMS już tnie artykuły na H2/H3, możesz podać te sekcje
wprost zamiast `segment_text` — separatory segmentera z nagłówkiem na
pierwszym miejscu emulują dokładnie tę strukturę (o to właśnie chodzi:
jednostki z czasu fitu i czasu inferencji muszą się zgadzać, patrz
[GOTCHAS.md §1](GOTCHAS.md#1-background-granularity-must-match-index-granularity)).

## Grupowanie / klastrowanie słów kluczowych (granularność `kw`)

Tła `_kw_` służą do innej roboty niż retrieval: **grupowania i
klastrowania krótkich fraz wyszukiwania** — list keywordów Google
Ads, zapytań z GSC, search terms reports. Tła dokumentowe tu nie
pasują, bo rozkład embeddingów 3-wyrazowej frazy nie ma nic
wspólnego z rozkładem 2000-znakowego akapitu (a anizotropia jest
znacznie gorsza — patrz stosunki wartości własnych wyżej).

```python
import numpy as np
from loader import load_background
from sklearn.cluster import AgglomerativeClustering

bg = load_background("te3small_pl_mixed50k_kw_mrl1536")

keywords = ["buty do biegania", "buty biegowe damskie",
            "kredyt hipoteczny kalkulator", ...]
x = embed_openai(keywords)                # (n, 1536), L2-znormalizowane
x_white = bg.apply(x)
x_white /= np.linalg.norm(x_white, axis=1, keepdims=True) + 1e-12

# Klastrowanie cosinusowe na wybielonych wektorach — klastry
# przestają być sklejane przez monokulturę dominującego kierunku.
labels = AgglomerativeClustering(
    n_clusters=None, distance_threshold=0.55,
    metric="cosine", linkage="average",
).fit_predict(x_white)
```

Korpus fitu `kw` to 50 000 polskich fraz keyword-podobnych (1–5
słów) wydobytych z `pl_mixed50k` — szczegóły w [Pochodzeniu
danych](#pochodzenie-danych). Przybliża **kształt** przestrzeni
embeddingów krótkich fraz, nie konkretną niszę — działa dla
keywordów z dowolnej branży. Jeśli chcesz rozkład fitu jeszcze
bliższy swoim kontom, domieszaj własne eksporty keywordów do
`data/corpus_keywords.parquet` i refituj
(`bash scripts/run_kw_fits.sh` — minuty pracy, grosze kosztu).

## Tła MRL

Qwen3-Embedding-4B/8B to modele trenowane z Matryoshka
Representation Learning — pierwsze `N < D` komponentów każdego wektora
stanowi sam w sobie poprawny embedding (po L2-renormalizacji). Modele
OpenAI text-embedding-3 mają ten sam mechanizm jako parametr API
`dimensions` (skrócenie + L2-renormalizacja). Dla każdego modelu repo
dostarcza osobny refit ZCA dla każdego popularnego `N`, więc
whitening zgadza się z tym co Twój pipeline faktycznie podaje do
indeksu przy inferencji:

| Model | Wymiar natywny | Dostępne refity MRL |
|---|---:|---|
| Qwen3-Embedding-4B | 2560 | `mrl{2560, 1536, 1024, 768, 512}` |
| Qwen3-Embedding-8B | 4096 | `mrl{4096, 3072, 2048, 1024, 768, 512}` |
| text-embedding-3-small | 1536 | `mrl{1536, 1024, 768, 512, 256}` |
| text-embedding-3-large | 3072 | `mrl{3072, 2048, 1536, 1024, 768, 512, 256}` |

Lista wymiarów dla 8B trzyma się kanonicznych targetów MRL Qwen3
(potęgi dwójki plus 768 i 3072); off-grid rozmiary jak 2560 / 1536 są
pominięte dla 8B bo model nie był trenowany MRL przy tych wymiarach —
slice matematycznie działa, ale recall byłby gorszy niż przy
wytrenowanych wymiarach. Dla modeli OpenAI `mrl<N>` pasuje zarówno do
wektorów pobranych z `dimensions=N`, jak i do natywnych wektorów
obciętych lokalnie do `N` + L2-renormalizowanych — per dokumentacja
OpenAI to to samo.

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

Dla granularności `segments` te same dokumenty przechodzą przez
`lib.segmenter`: ten sam rekurencyjny splitter, ale z separatorami z
nagłówkami markdown na pierwszym miejscu (`\n## `, `\n### ` — na
korpusie plain-text schodzą do pakowania akapitów po `\n\n`), capem
1024 tokenów, **bez overlapu** i progiem `merge_tiny` 300 znaków, co
daje **73 692 sekcje** (średnio 1.47 na dokument, wszystkie ≥300
znaków).

Dla granularności `paragraphs` skrypt
`scripts/build_corpus_paragraphs.py` przepuszcza te same dokumenty
przez `lib.paragrapher`: tnie ściśle po pustych liniach (`\n\n`), więc
jeden akapit to dokładnie jeden wiersz (nigdy nie sklejany między
akapitami), akapity zbyt duże (>512 tokenów) dzieli na granicach zdań,
a `merge_tiny(min_chars=120)` dokleja fragmenty typu sam-nagłówek /
jedna-linijka do przodu — co daje **196 759 akapitów** (3.93 na
dokument, mediana 488 znaków, ~32 M tokenów). To najmniejsza
jednostka strukturalna powyżej zdania, zajmująca odrębne pasmo
długości między `kw` a `chunks` (mniej więcej pół chunka). Zapisuje
`data/corpus_paragraphs.parquet`.

Dla granularności `kw` skrypt `scripts/build_corpus_keywords.py`
wydobywa z tego samego korpusu **50 000 fraz keyword-podobnych**:
lowercase n-gramy (1–5 słów), które nie zaczynają się ani nie kończą
stopwordem (328-słowa lista
[stopwords-iso/stopwords-pl](https://github.com/stopwords-iso/stopwords-pl))
ani gołą liczbą, 3–60 znaków, częstość dokumentowa ≥ 3, próbkowane
jednostajnie w obrębie kubełka długości przy realistycznym mixie
listy keywordów (10% 1-słowo / 35% 2-słowa / 30% 3-słowa / 15%
4-słowa / 10% 5-słów), seed = 42. Próbkowanie jednostajne (nie
ważone częstością) unika nadreprezentacji webowego boilerplate'u.

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
backgrounds/<name>/                   # 103 katalogi
  W_A.npy           # (dim, dim) float32  — zastosowanie: (x - mu) @ W
  mu_A.npy          # (dim,)    float32
  eigvals_A.npy     # (dim,)    float32   — diagnostyka, niepotrzebne przy apply
  <name>.meta.json  # pochodzenie + diagnostyka
REGISTRY.md         # czytelny indeks, autogenerowany
registry.json       # to samo, w wersji do parsowania
loader.py           # loader tylko numpy (patrz Szybki start)
lib/chunker.py      # RecursiveCharacterTextSplitter używany przez build_corpus_chunks.py
lib/segmenter.py    # wariant sekcyjny używany przez build_corpus_segments.py
lib/paragrapher.py  # splitter akapitów wg pustych linii używany przez build_corpus_paragraphs.py
scripts/            # pipeline korpus + embed + fit + index
LICENSE             # CC-BY-4.0
README.md           # wersja angielska
README.pl.md        # ten plik
```

## Jak zostały zbudowane

Próbka mixu jak wyżej (seed=42), embedding każdego dokumentu /
chunka / segmentu / akapitu / frazy — modele Qwen przez OpenRouter, modele OpenAI przez
`api.openai.com` (ten sam skrypt, `--base-url` + `--api-key-env`) —
potem fit ZCA w dwóch streamingowych przejściach po chunkach
(`μ = E[x]`, `Σ = E[(x-μ)(x-μ)ᵀ]`), a potem
`W = U · diag(1/√(S + ε)) · Uᵀ` z `SVD(Σ)`, gdzie `ε=1e-6`. Bez GPU.
Koszty: rodziny Qwen doc+chunks **~$2.77** przez OpenRouter (4B doc
$0.92, 8B doc $0.43, 4B chunks $0.95, 8B chunks $0.48 — routing z
`--ignore-providers siliconflow`, bo SiliconFlow jest 4× droższy);
rodziny Qwen `segments` **~$1.39** (4B $0.93, 8B $0.46 — po 46.3 M
tokenów); rodziny OpenAI doc+chunks **~$14** przez API OpenAI (~95 M tokenów ×
$0.02/M dla 3-small i $0.13/M dla 3-large); cztery rodziny `kw` to
**grosze** (~0.4 M tokenów każda).

Per-dokumentowy kontekst egzekwowany jest precyzyjnie na etapie
embed: każdy doc przechodzi przez tokenizer modelu — `tokenizer.json`
Qwen3 z HF (byte-identyczny dla 4B i 8B) albo `tiktoken`
`cl100k_base` dla modeli OpenAI — i jest obcinany do **30 000
tokenów** dla Qwen (~2k zapasu pod oknem 32k) lub **8 191 tokenów**
dla OpenAI (ich twardy limit inputu; obcięło 326 z 50 042 doków).
Chunki i frazy keywordowe nie zbliżają się do żadnego z limitów.

Te same chunki embedów są potem fitowane raz na każdy wymiar MRL —
przez **refit od zera**: slice chunka do `N` kolumn,
L2-renormalizacja wierszowa, ponowne obliczenie μ i Σ, świeże SVD.
(Nie liczymy pełnego `W` raz i nie slice'ujemy go — to dałoby złe
statystyki.) Cała siatka MRL dla jednej granularności jednego modelu
zajmuje poniżej dwóch minut na CPU po zakończeniu embed.

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
dokumentów. W drugą stronę: frazy `kw` są przy natywnym wymiarze
jeszcze bardziej anizotropowe niż dokumenty (np. 149.9 dla 8B kw
mrl4096 vs 157.6 dla 8B doc — ale 81.5 dla 4B kw mrl2560 vs 91.7
doc; pełna diagnostyka wszystkich 103 teł w
[`REGISTRY.md`](REGISTRY.md)).

## Zbudować od zera (lub dopasować dla własnego modelu)

Katalog `scripts/` zawiera kompletny pipeline który możesz odpalić z
dowolnym kluczem OpenRouter (dowolny model embeddujący wspierany
przez OpenRouter) albo kluczem OpenAI
(`--base-url https://api.openai.com/v1/embeddings --api-key-env
OPENAI_API_KEY`). Wall-time: ~0.5-3 h na model na granularność.
Koszt API dla 50k polskich dokumentów / 129k chunków: ~$0.4-1 na
model Qwen przez OpenRouter, ~$0.9-6.4 na model OpenAI ($0.02/M dla
3-small, $0.13/M dla 3-large), a każda rodzina `kw` to grosze
(~0.4 M tokenów).

```bash
git clone https://github.com/romek-rozen/polish-whitening-backgrounds.git
cd polish-whitening-backgrounds

# 1. Zainstaluj minimalne zależności (numpy + pyarrow + datasets + requests + tokenizers + tiktoken + trafilatura).
pip install -r requirements.txt

# 2. Podaj swój klucz / klucze.
cp .env.example .env
$EDITOR .env             # wklej OPENROUTER_API_KEY=sk-or-... i/lub OPENAI_API_KEY=sk-...

# 3. End-to-end dla rodzin Qwen: korpus(_chunks) → embed (4B + 8B) → fit → index.
bash scripts/run_full.sh

# 4. Rodziny keywordowe (wszystkie cztery modele):
python scripts/build_corpus_keywords.py
python scripts/embed_via_openrouter.py --model qwen/qwen3-embedding-4b \
    --corpus data/corpus_keywords.parquet --out data/kw_corpus/
python scripts/embed_via_openrouter.py --model text-embedding-3-small \
    --corpus data/corpus_keywords.parquet --out data/kw_corpus/ \
    --base-url https://api.openai.com/v1/embeddings --api-key-env OPENAI_API_KEY
# … analogicznie qwen3-embedding-8b / text-embedding-3-large, potem:
bash scripts/run_kw_fits.sh

# 5. Rodziny OpenAI doc+chunks:
bash scripts/run_oai_fits.sh   # po zembeddowaniu corpus.parquet i
                               # corpus_chunks_512_64.parquet oboma modelami

# 6. Granularność segments (rodziny Qwen):
python scripts/build_corpus_segments.py
CORPUS=data/corpus_segments_1024.parquet OUT_ROOT=data/segments_corpus \
  NAME_PREFIX=pl_mixed50k_segments bash scripts/run_full.sh

# 7. Granularność paragraphs (wszystkie cztery modele, jeden launch):
python scripts/build_corpus_paragraphs.py
bash scripts/run_paragraphs.sh
```

Co robi każdy skrypt:

| Skrypt | Zastosowanie |
|---|---|
| `scripts/build_corpus.py` | Próbkuje mix polski (wiki + FineWeb-2 PL + oasst) z seed=42 i progiem 500 znaków na akapit. Zapisuje `data/corpus.parquet`. Default: brak górnego capa. |
| `scripts/build_corpus_chunks.py` | Tnie `data/corpus.parquet` przez `lib.chunker` (512 tok / 64 tok overlap, merge sub-100-char, strip overlap fragments). Zapisuje `data/corpus_chunks.parquet` (129 181 chunków). |
| `scripts/build_corpus_segments.py` | Ten sam korpus → `lib.segmenter` (separatory z nagłówkiem na pierwszym miejscu, cap 1024 tokeny, bez overlapu, próg merge_tiny 300 znaków) → zapisuje `data/corpus_segments_1024.parquet` (73 692 wiersze). |
| `scripts/build_corpus_paragraphs.py` | Ten sam korpus → `lib.paragrapher` (cięcie ściśle po pustych liniach, jeden akapit = jeden wiersz, akapity zbyt duże >512 tokenów dzielone na granicach zdań, próg merge_tiny 120 znaków) → zapisuje `data/corpus_paragraphs.parquet` (196 759 wierszy). |
| `scripts/build_corpus_keywords.py` | Wydobywa z tego samego korpusu 50 000 fraz keyword-podobnych (n-gramy 1–5 słów, filtr stopwords-pl na brzegach, df ≥ 3, stratyfikowany mix długości). Zapisuje `data/corpus_keywords.parquet`. |
| `scripts/embed_via_openrouter.py` | Embedduje dowolny parquet korpusu przez endpoint zgodny z OpenAI `/v1/embeddings` — domyślnie OpenRouter, `api.openai.com` przez `--base-url` + `--api-key-env`. Wstępne, precyzyjne obcinanie po tokenach pod okno kontekstu modelu (tokenizer Qwen3 z HF albo tiktoken dla modeli OpenAI — zmiana przez `--max-tokens-per-doc` / `--tokenizer-repo`). Adaptacyjny batch (połowa przy 429/5xx, rośnie po seriach sukcesów). Idempotentny: resume z najwyższego istniejącego chunka. Pisze `chunks_<slug>/*.npy` plus per-call `cost_report_<slug>.json`. |
| `scripts/fit_zca.py` | Dwa streamingowe pass-y (μ, Σ) po chunkach + SVD. Opcjonalne `--truncate-to N` obcina każdy chunk do `N` kolumn i ponownie renormalizuje przed fitem, do refitów MRL. Pisze `backgrounds/<name>/{W_A.npy, mu_A.npy, eigvals_A.npy, *.meta.json}`. |
| `scripts/index_backgrounds.py` | Regeneruje `REGISTRY.md` + `registry.json`. Wywoływane przez skrypty-runnery. |
| `scripts/run_full.sh` | Orchestrator rodzin Qwen: korpus → embed na każdy model → fit przy każdym wymiarze z `DIMS_<MODEL>` → index. Idempotentny — bezpieczny do ponownego uruchomienia. Dla korpusów pochodnych ustaw `CORPUS=` + dedykowany `OUT_ROOT=` (np. run segments niżej); twardy guard odmawia korpusu pochodnego z domyślnym `OUT_ROOT`. |
| `scripts/run_paragraphs.sh` | Orchestrator granularności `paragraphs`: build/embed/fit wszystkich czterech modeli w jednym launchu z `data/corpus_paragraphs.parquet` → index. Idempotentny. |
| `scripts/run_kw_fits.sh` | Fituje każde tło `kw`, którego embeddingi są kompletne pod `data/kw_corpus/` (wszystkie cztery modele) → index. Pomija częściowe embeddingi. |
| `scripts/run_oai_fits.sh` | Fituje tła OpenAI `doc` + `chunks` z `data/chunks_<model>/` i `data/chunks_corpus/chunks_<model>/` → index. Pomija częściowe embeddingi. |

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

Jeżeli korzystasz z tych teł w publikacji, prosimy zacytować model
embeddingowy, którego używasz, oraz odesłać do tego repo, żeby inni
mogli też je znaleźć:

```
@misc{polish-whitening-backgrounds,
  author = {Rozenberger, Roman},
  title  = {Polish ZCA whitening backgrounds for Qwen3-Embedding and OpenAI text-embedding-3},
  year   = {2026},
  url    = {https://github.com/romek-rozen/polish-whitening-backgrounds}
}
```
