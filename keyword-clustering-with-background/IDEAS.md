# IDEAS — kierunki nieprzetestowane

Pomysły, które wyszły z dyskusji po eksperymencie, ale **nie zostały
zmierzone**. Trzymane osobno od [RESUME.md](./RESUME.md), gdzie są wnioski
poparte danymi. Nic tutaj nie jest rekomendacją — to lista rzeczy do
sprawdzenia, z zaznaczonym kosztem i ryzykiem.

---

## Problem, który to ma rozwiązać

Klastrowanie po surowych embeddingach fraz skleja po **cechach powierzchniowych**,
nie po intencji. Zaobserwowane w tym eksperymencie:

- **geo** — `<usługa A> <miasto>` i `<usługa B> <miasto>` lądują razem, bo dzielą
  miasto
- **rama składniowa** — wszystkie frazy `„<termin> co to"` w jednej grupie,
  niezależnie od tego, o czym są
- **nazwiska** — wszystkie trzy modele robią z nich jeden worek „osoby"

Wspólny mianownik: to są **poprawne klastry semantyczne**, które nie odpowiadają
podziałowi tematycznemu. Filtrowanie ich przed klastrowaniem (obecna rada
w RESUME) leczy objaw.

---

## Pomysł 1 — encje + Jaccard + cosinus  ⭐ główny kierunek

```
query  →  encje (1-2 na frazę)  →  klastrowanie (Jaccard ∩ cosinus)
```

Podział pracy między dwoma sygnałami:

| sygnał | rola | charakter |
|---|---|---|
| **Jaccard na encjach** | „czy to jest o tym samym" | dyskretny, strukturalny |
| **cosinus** | „jak bardzo blisko" | ciągły, gradacyjny |

Sam Jaccard jest za zgrubny, sam cosinus gubi strukturę i skleja po geo.
Razem każdy robi to, w czym jest dobry.

### Dlaczego DWIE encje mają znaczenie

Przy **jednej** encji na frazę Jaccard degeneruje się do wartości binarnej:
ta sama encja → 1.0, inna → 0.0. To już nie klastrowanie, tylko `GROUP BY`.

Przy **dwóch** pojawia się gradacja (1.0 / ~0.33-0.5 / 0), i dopiero wtedy
Jaccard niesie informację. To była realna wada wcześniejszej wersji pomysłu.

### ⚠️ Pułapka, która to wywróci

**Jeśli geo trafi do zbioru encji, problem wraca tym samym kanałem:**

```
agencja seo warszawa        → {agencja seo, warszawa}
catering dietetyczny warszawa → {catering dietetyczny, warszawa}
Jaccard = 1/3   ← niezerowe podobieństwo WYŁĄCZNIE z powodu miasta
```

Rozwiązanie istnieje już w głównym projekcie: `entities[]` mają `type`,
`category` i **`is_central`**. Klastruj **tylko po encjach centralnych**, geo
trzymaj jako osobny atrybut — przyda się później do targetowania kampanii,
czyli tam, gdzie faktycznie należy.

### Jak łączyć oba sygnały

**A. Ważona suma** — `sim = α·jaccard + (1−α)·cosinus`.
Proste, ale `α` to kolejne pokrętło do dobrania, a widzieliśmy już, jak źle
przenoszą się stałe między zbiorami.

**B. Encja jako WARUNEK, nie jako punkty** — klastruj cosinusem, ale **zabroń
scalenia grup, które nie dzielą żadnej encji centralnej**.
Cosinus decyduje *jak bardzo*, encja decyduje *czy wolno*.

**Skłaniam się do B.** Problemy zdiagnozowane w eksperymencie są **jakościowe**
(„te dwie grupy nie powinny się połączyć"), a nie ilościowe („połączyły się
o 15% za mocno"). Twardy warunek pasuje do natury problemu lepiej niż waga,
i nie wymaga strojenia.

---

## Pomysł 2 — odjęcie kierunku w przestrzeni (bez żadnej ekstrakcji)

Najtańszy wariant: **korzysta wyłącznie z wektorów, które już są policzone.**

1. Weź kilkaset par `<fraza> <miasto>` / `<fraza>` (w danych ich pełno).
2. Policz **średnią różnicę wektorów** → to jest „kierunek geo" w przestrzeni.
3. Zrzutuj ten kierunek z wszystkich wektorów.

Efekt: frazy różniące się tylko miastem **zbliżają się**, a frazy dzielące
wyłącznie miasto **oddalają** — bo to, co je łączyło, znika.

**To jest ta sama idea, co całe to repo.** Whitening usuwa kierunek dominujący;
geo i „bycie osobą" to po prostu *kolejne* takie kierunki, tylko węższe.
Narzędzie już jest, trzeba je wycelować.

Koszt: **jedno mnożenie macierzy.** Zero ekstrakcji, zero modelu, zero tagowania.

Ryzyko: kierunek geo może nie być spójnym wektorem — wtedy rzutowanie nic nie da
albo zepsuje coś innego. **Sprawdzalne w kilkanaście minut** na 19 801 frazach,
które już są zaembedowane.

---

## Pomysł 3 — ekstrakcja encji: co jest ile warte

Obawa „to drogie i czasochłonne" dotyczy LLM-a. Reszta jest tania:

| metoda | koszt na 1 mln fraz | uwagi |
|---|---|---|
| **reguły + słownik** | sekundy | odejmij modyfikatory zamiast rozpoznawać encje: geo (~2 tys. polskich miast), intencja (`cena`, `opinie`, `ranking`, `jak`, `co to`), własna lista brandów. Co zostanie, to rdzeń |
| **spaCy POS** (`pl_core_news_sm`) | ~5 min CPU | 15 MB modelu, wyłącz parser i NER, `nlp.pipe`. Fraza ma ~3 słowa, więc to 3 mln tokenów, nie milion dokumentów |
| **LLM** | godziny + koszt | tu obawa jest uzasadniona |

**POS-tagging to inna liga kosztowa niż LLM** — te dwie rzeczy łatwo zlepić
w jedno „ekstrakcja encji = drogo", a to nieprawda.

Ironia: spaCy odrzuciliśmy przy budowie korpusu (tytuły Wikipedii dały ten sam
efekt za darmo). Tutaj może być właściwym narzędziem.

---

## Kolejność testowania

**3 → 1 → 2** według stosunku wartości do kosztu, ale zacznij od najtańszego:

1. **Pomysł 2** (odjęcie kierunku) — kilkanaście minut, dane już są.
   Jeśli zadziała, problem znika bez budowania pipeline'u ekstrakcji.
2. **Pomysł 3, wariant regułowy** — pół dnia, daje encje do przetestowania
   Pomysłu 1 bez inwestycji w tagger.
3. **Pomysł 1 z wariantem B** (encja jako warunek) — dopiero gdy masz encje.

Wszystko testowalne na 19 801 frazach z tej sesji, które są już zaembedowane
i mają policzone klastry do porównania (`work/klastry_FINAL.xlsx`).

---

## Jak to zmierzyć, żeby nie powtórzyć błędu tej sesji

W tym eksperymencie oceniałem klastry pytaniem „czy da się do tego napisać jedną
reklamę" — czyli kryterium **jednego zastosowania**, nie jakości semantycznej.
Skutek: część poprawnych klastrów została zaraportowana jako błędy.

Przy testowaniu tych pomysłów użyj **`entity_coherence_lift`** z projektu
`clustering_Louvain_Leiden_UmapHdbscan`: średni Jaccard zbiorów encji
centralnych wewnątrz klastra, jako lift nad losowym podziałem tej samej
wielkości. Jest niezależny od kategorii i — co ważne — **nie nagradza metody za
optymalizowanie tego samego kosinusa, na którym klastruje**.

Uwaga na cykliczność: jeśli klastrujesz po encjach i mierzysz spójnością encji,
metryka staje się autoreferencyjna. Wtedy potrzebny jest niezależny sygnał —
np. wolumeny, CTR, albo ręczne etykiety na próbce.
