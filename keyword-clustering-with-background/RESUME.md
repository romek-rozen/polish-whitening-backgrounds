# RESUME — którą metodę klastrowania wybrać i do czego

Wnioski z eksperymentu na 19 801 polskich słowach kluczowych, 3 modelach
serwowalnych na CPU i 5 metodach klastrowania. Napisane pod kolejne użycie:
**docelowo ktoś wrzuci milion fraz i będzie chciał klastry semantyczne.**

To jest dokument decyzyjny. Uzasadnienia i liczby są w [README.md](./README.md).

---

## Najważniejsze: metoda zależy od SKALI, nie od gustu

Rekomendacja dla 20 tysięcy fraz **nie przenosi się** na milion. Aglomeracja
potrzebuje pełnej macierzy odległości — to rośnie kwadratowo:

| liczba fraz | macierz odległości | aglomeracja |
|---:|---:|---|
| 20 000 | 1.6 GB | ✅ ~50 s |
| 100 000 | 40 GB | ⚠️ na granicy |
| 1 000 000 | **4 TB** | ❌ niewykonalne |

Graf kNN nie ma tego problemu — pamięć rośnie liniowo (`n × k` krawędzi).

## Tabela decyzyjna

| skala | metoda | dlaczego |
|---|---|---|
| **do ~30 tys.** | **aglomeracja, average linkage** | najlepsza jakość: brak mega-worków, grupy wielkości grupy reklam (mediana 3, max 121) |
| **30 tys. – 200 tys.** | aglomeracja jeśli RAM pozwala, inaczej kNN + Leiden | granica przebiega tam, gdzie `n²/2 × 8 B` przestaje się mieścić |
| **powyżej 200 tys., w tym 1 mln** | **kNN + Leiden z odcięciem** | jedyna metoda, która skaluje. Zmierzone: 100 tys. fraz = 29 s, 1.5 GB na 4 wątkach |
| **kilkaset fraz** | próg + union-find (jak `keyword_cluster`) | przy małych listach łańcuchy nie mają się gdzie rozrosnąć, a próg ręczny jest sensowny |

**Nigdy:** UMAP + HDBSCAN jako metoda produkcyjna. Redukcja 1024 → 30 wymiarów
wyrzuca informację, za którą zapłaciłeś, a UMAP jest losowy. Zostaw go jako
**referencję** do oszacowania szumu — to jedyna metoda, której szum nie jest
Twoim pokrętłem.

## Przepis na milion fraz

```
1. embedding    bge-m3 + tło bgem3_pl_kwmix900k_mrl1024
                GPU: ~12 min (1400 fraz/s).  CPU: ~7 h (40 fraz/s) — cache'uj.
2. tło          (x - mu) @ W  →  ~4 s na milion. Zawsze. Bez tego progi nie działają.
3. kNN          pynndescent, k=15, cosine. NIE brute force (10^12 par).
4. odcięcie     usuń najsłabsze krawędzie — PERCENTYLEM, nie stałą (patrz niżej)
5. Leiden       RBConfiguration, resolution 4.0 na start
6. min_size     grupy < 3 fraz → szum
```

Szacunek pamięci przy 1 mln: wektory fp32 1024-dim = 4 GB, graf kNN ≈ 1 GB,
Leiden ≈ 2-3 GB. **Mieści się w 16 GB RAM.**

⚠️ Leiden przy `resolution 4.0` na 20 tys. fraz dawał grupy po 400-600 fraz —
za duże na grupę reklam. Przy milionie trzeba **podnieść resolution** albo
klastrować dwuetapowo: najpierw grube tematy, potem każdy temat osobno
(wtedy aglomeracja wraca do gry, bo pojedynczy temat ma znowu tysiące, nie
miliony fraz). **Klastrowanie dwuetapowe to prawdopodobnie właściwa odpowiedź
na milion** — daje jakość aglomeracji przy skalowalności grafu.

## Co jest prawdą niezależnie od skali

**1. Próg podobieństwa to percentyl, nie stała.**
Zmierzone: mediana najbliższego sąsiada w przestrzeni z tłem ≈ 0.75. Stały próg
0.8 leży *powyżej* mediany, więc odrzuca 60% fraz — nie dlatego, że są złe, tylko
dlatego, że nie mają tak bliskiego bliźniaka. Przewidywanie z rozkładu (60.6%)
i zmierzony szum (60.63%) zgodziły się co do drugiego miejsca po przecinku.
**Zawsze policz rozkład top-1 dla konkretnej listy** (`analyze_similarity_distribution.py`).

**2. Próg nie przenosi się między przestrzeniami ani modelami.**
Najostrzejszy dowód: qwen3-0.6b, przestrzeń **surowa**, próg 0.5 → **jeden
klaster na 19 263 z 19 801 fraz**. Ten sam 0.5 z tłem → 2 727 sensownych grup.

**3. Szumu nie kasuj.**
55-65% „szumu" to poprawne frazy bez bliźniaka. Realny odpad to 25-35%.
Przy milionie i 23% szumu to ~150 tys. wartościowych fraz do wyrzucenia,
gdyby potraktować szum jako śmieci.

**4. Tło jest kalibracją, nie ulepszaczem.**
Surowo trzy modele siedzą na różnych skalach podobieństwa (0.75-0.84), więc próg
dobrany na jednym nie działa na drugim. Po wybieleniu zbiegają do 0.51-0.56.
Zysk jakości jest wtórny (qwen +0.075 AMI, bge-m3 ~0) — **główna wartość to
porównywalność progów.**

**5. Metoda progowa + union-find = single linkage.**
Stąd łańcuchy: jedna fraza-most skleja dwie niepowiązane grupy. Aglomeracja
z **average linkage** to ta sama idea bez tej wady.

**6. 0% szumu to wada, nie zaleta.**
Leiden bez odcięcia etykietuje wszystko, więc raportuje 0% szumu na liście, która
zawiera ceny, pojedyncze znaki i przypadkowe ciągi. Jeśli metoda nie ma szumu,
to znaczy, że nie mówi prawdy o ogonie.

## Wybór modelu

**bge-m3** — cztery niezależne przesłanki: najlepszy w benchmarku z etykietami
(AMI 0.989 vs 0.924 i 0.897), najmniej przerośniętych grup (2 klastry ≥100 fraz
vs 9 i 16), najmniejszy największy klaster przy aglomeracji (136 vs 462 i 523),
brak sklejania po ramie składniowej.

`embeddinggemma` jest ~1.6× szybsza i ma mniejszy wektor (768) — jeśli przy
milionie fraz throughput będzie wąskim gardłem, to jest realna alternatywa,
ale skleja frazy o wspólnej ramie („kurs/opis/typy X") niezależnie od tematu.

## Odsiej to PRZED klastrowaniem

Trzykrotnie potwierdzone niezależnie. **Żaden model tego nie rozwiąże** — to są
frazy należące do osobnych kampanii:

- **Nazwiska** — wszystkie trzy modele robią z nich jeden worek „osoby". Są do
  siebie podobne jako *typ bytu*, nie jako intencja.
- **Modyfikatory geograficzne** — nazwa miasta przeciąga frazę do klastra
  lokalnego niezależnie od branży. W Google Ads lokalizacja to i tak
  targetowanie kampanii, nie semantyka grupy reklam.

Drugorzędnie warto rozdzielić **intencję informacyjną od transakcyjnej**
(„co to jest X" vs „X szkolenie cena") — inne stawki, inne strategie.

## Ile pracy zostaje

Na 19 801 frazach, po klastrowaniu wg rekomendacji: **2-4 dni do pierwszej
wersji kampanii.** Z 30 największych grup 6 było gotowych od ręki, 14 wymagało
lekkiego czyszczenia, 10 podziału. Z małych grup 70-75% miało realną wartość.

To nie jest „wrzuć i gotowe" — to jest materiał roboczy dobrej jakości.
Planuj ręczny przegląd, zwłaszcza największych grup.

## ⚠️ Kryterium oceny było stronnicze — przeczytaj przed użyciem wniosków

Klastry oceniano pytaniem **„czy da się do tego napisać jedną reklamę"**. To
kryterium konkretnego zastosowania (Google Ads), a nie miara jakości
klastrowania semantycznego. Cel docelowy — „ktoś wrzuca milion fraz i chce
klastry semantyczne" — jest szerszy.

**Skutek: część rzeczy zaraportowanych jako błędy nimi nie jest.**
`agencja seo warszawa` naprawdę jest semantycznie bliska innym frazom
z Warszawą. Nazwiska naprawdę są do siebie podobne jako byty. Model miał rację;
to ramka „grupa reklam" mówi, że się pomylił. Sekcja „Odsiej to PRZED
klastrowaniem" jest więc poradą **dla zastosowania reklamowego**, a nie
uniwersalną poprawką jakości.

**Lepsze kryterium prawdopodobnie istnieje i jest już w projekcie
`clustering_Louvain_Leiden_UmapHdbscan`: `entity_coherence_lift`** — średni
Jaccard zbiorów encji centralnych wewnątrz klastra, jako lift nad losowym
podziałem tej samej wielkości. Jest niezależny od kategorii i nie nagradza
metody za optymalizowanie tego samego kosinusa, na którym klastruje.

Otwarty kierunek (nieprzetestowany): **sprowadzić frazy do encji** i klastrować
po nich, zamiast po surowych embeddingach fraz. Wtedy `agencja seo warszawa`
i `catering dietetyczny warszawa` mają różne encje centralne (SEO vs catering)
mimo wspólnej lokalizacji — czyli problem, który dziś „naprawialiśmy"
filtrowaniem geo, rozwiązuje się sam. Koszt: trzeba ekstrakcji encji na milionie
fraz, a krótka fraza daje mało kontekstu.

## Czego ten eksperyment NIE sprawdził

- **Miliona fraz.** Wszystko powyżej 100 tys. to ekstrapolacja z pomiaru na
  100 tys. Przed produkcją zrób test skali na realnej wielkości.
- **Innej branży i języka.** Jedna polska witryna, dwie tematyki.
- **Klastrowania dwuetapowego**, które proponuję wyżej na milion. Pomysł wynika
  z ograniczeń pamięci, nie z pomiaru.
- **Jakości względem prawdziwych grup reklam.** Nie mieliśmy aktualnego ground
  truth; ocena jest ekspercka, nie ilościowa.
- **Jakości semantycznej jako takiej.** Oceniano przydatność reklamową — patrz
  ostrzeżenie wyżej. Metryka encyjna nie została policzona.
- **Klastrowania po encjach** zamiast po frazach.
