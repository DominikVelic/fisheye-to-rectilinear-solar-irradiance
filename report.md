# Odhad slnečného žiarenia z fotografií oblohy — Správa o návrhu riešenia

## Formulácia problému

Úloha je **regresný problém**: na základe fotografie oblohy predpovedať okamžitú hodnotu slnečného žiarenia (W/m²). Hlavná výskumná otázka je, či transformácia fisheye snímok do ekvirektangulárneho formátu zlepší presnosť odhadu pri rôznych architektúrach CNN a rôznych rozlíšeniach vstupu.

---

## Predspracovanie obrazu

### Transformácia fisheye → ekvirektangulárna projekcia

Kamera používa **ekvidistantnú fisheye** projekciu, kde vzdialenosť pixela od stredu obrazu je úmerná zenitovému uhlu:

```
r = R · θ / (π/2)
```

Pre konvolučné siete to spôsobuje dva praktické problémy:

1. **Zbytočné pixely** — približne 21 % obrazu (štyri rohy) je čiernych a nenesie žiadnu informáciu o oblohe.
2. **Nerovnomerné uhlové rozlíšenie** — oblasti pri horizonte sú silne komprimované oproti zenitu. CNN aplikuje konvolučné jadrá rovnomerne na všetky priestorové polohy a nedokáže túto geometrickú deformáciu kompenzovať.

Ekvirektangulárna transformácia premapuje pixely tak, že:
- os x pokrýva azimut φ ∈ [0°, 360°) rovnomerne
- os y pokrýva zenitový uhol θ ∈ [0°, 90°] rovnomerne

Každý výstupný pixel zodpovedá rovnakému priestorovému uhlu oblohy. CNN tak dostáva priestorovo konzistentné príznaky a eliminujú sa zbytočné rohové pixely. Výstupná veľkosť je (H/2) × W, teda 534×1068 pre vstup 1068×1068.

Testujú sa oba varianty (originálny aj transformovaný), aby sa kvantifikovalo, či geometrická korekcia skutočne zlepšuje presnosť.

### Rozlíšenie vstupu (224, 256, 320)

Predtrénované ImageNet modely boli pôvodne trénované na **224×224**, čo je prirodzený základ. Väčšie rozlíšenia (256, 320) zachovávajú viac priestorových detailov po zmenšení z originálnych 1068×1068, čo môže zlepšiť presnosť na úkor väčšej spotreby VRAM a dlhšieho tréningu. Testovanie troch rozlíšení odhaľuje kompromis medzi presnosťou a výpočtovými nárokmi pre túto konkrétnu úlohu.

### Normalizácia

**Normalizácia pixelov** používa ImageNet priemery a smerodajné odchýlky kanálov (RGB: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]). Predtrénované siete boli trénované práve s týmito hodnotami — použitie inej normalizácie by posunulo distribúciu vstupu mimo rozsah, s ktorým sieť počíta, a zhoršilo by transfer learning.

**Normalizácia cieľovej hodnoty** delí žiarenie hodnotou 1500 W/m² (mierne nad pozorovaným maximom ~1440 W/m²), čím sa všetky cieľové hodnoty mapujú do intervalu [0, 1]. Toto zlepšuje numerickú stabilitu pri výpočte straty a udržiava gradienty v rozumnom rozsahu bez ohľadu na fyzikálnu jednotku.

---

## Architektúry modelov

Všetky štyri modely sú použité ako **extraktory príznakov s transfer learningom**: pôvodná klasifikačná hlava je nahradená jedným lineárnym neurónom pre regresiu a pre všetky ostatné vrstvy sú načítané váhy predtrénované na ImageNet. Tým sa výrazne znižujú požiadavky na trénovacie dáta aj čas tréningu oproti trénovaniu od nuly.

Štyri architektúry boli vybrané tak, aby pokrývali rôzne body v priestore kompromisu medzi presnosťou a efektivitou:

### ResNet18
Plytká reziduálna sieť s 11M parametrami. Reziduálne (preskakujúce) spojenia riešia problém miznúcich gradientov a umožňujú stabilné trénovanie hlbších sietí. ResNet18 je najľahší z testovaných modelov — slúži ako rýchly základ a dolná hranica toho, čo jednoduchší model dokáže dosiahnuť.

### ResNet50
Hlbší variant tej istej rodiny s 25M parametrami a blokovými reziduálnymi blokmi (bottleneck). Väčšia hĺbka umožňuje učiť sa abstraktnejšie reprezentácie príznakov. Porovnanie ResNet18 vs ResNet50 izoluje vplyv kapacity modelu v rámci tej istej architektonickej rodiny.

### MobileNet V3 Small
Navrhnutý pre nasadenie na mobilných a embedded zariadeniach pomocou **depthwise separable konvolúcií**, ktoré rozkladajú štandardnú konvolúciu na priestorovú (per-kanálovú) a bodovú (1×1 medziikanálovú) operáciu. Tým sa zníži počet parametrov na ~2.5M s minimálnou stratou presnosti. MobileNetV3 je relevantný pre prípad nasadenia modelu priamo pri senzore.

### EfficientNet-B0
Používa **compound scaling**: šírka, hĺbka a rozlíšenie siete sú škálované súčasne pomocou fixného pomeru odvodeného z neural architecture search. B0 je najmenší variant (~5.3M parametrov), no dosahuje lepšiu presnosť na parameter ako ResNety. Predstavuje modernú paradigmu efektívnych architektúr a poskytuje priame porovnanie so staršími návrhmi.

---

## Stratová funkcia

Ako trénovací kritérium sa používa **Mean Squared Error (MSE)**:

```
L = (1/N) Σ (ŷᵢ − yᵢ)²
```

MSE je štandardná voľba pre regresiu z niekoľkých dôvodov:
- Je všade diferencovateľná, čo umožňuje optimalizáciu pomocou gradientu.
- Penalizuje veľké chyby kvadraticky, čo je vhodné pre odhad žiarenia, kde veľké odchýlky (napr. predpoveď nízkeho žiarenia počas jasnej oblohy) sú výrazne nežiaduce.
- Strata sa počíta na normalizovaných hodnotách (delené 1500), takže škála nespôsobuje numerické problémy.

MAE (stredná absolútna chyba) by bola odolnejšia voči odľahlým hodnotám, ale má nespojitý gradient v nule. MSE je preto vhodnejšia pre stabilný tréning. Pri vyhodnocovaní sa používajú obe metriky (MAE aj RMSE) v pôvodných jednotkách W/m² pre interpretovateľnosť.

---

## Optimalizátor — AdamW

**AdamW** udržiava adaptívne rýchlosti učenia pre každý parameter (ako Adam) a pridáva **oddelený weight decay**. Štandardný Adam aplikuje weight decay ako súčasť aktualizácie gradientu, kde interaguje s adaptívnym škálovaním a vedie k nedostatočnej regularizácii. AdamW aplikuje weight decay priamo na váhy, čím zabezpečuje správnu L2 regularizáciu. To je zvlášť dôležité pri fine-tuningu, kde by predtrénované váhy mali zostať blízko svojej inicializácie, pokiaľ dáta silne nenaznačujú zmenu.

Parametre: `lr` (škálované lineárne s veľkosťou dávky od základu 3×10⁻⁴ pri batch=64), `weight_decay=1e-4`.

### Lineárne škálovanie LR s veľkosťou dávky

Keď sa veľkosť dávky zväčší k-násobne, každá aktualizácia gradientu vidí k-krát viac vzoriek. Aby sa zachovala rovnaká dynamika učenia, rýchlosť učenia sa škáluje rovnakým faktorom:

```
lr = základné_lr × (veľkosť_dávky / referenčná_veľkosť_dávky)
```

Toto je lineárne škálovacie pravidlo (Goyal et al., 2017). Je to aproximácia — veľké dávky môžu vyžadovať warmup — ale je to robustný východiskový bod pre AdamW fine-tuning.

---

## Plánovač LR — Cosine Annealing

Rýchlosť učenia klesá podľa kosínusovej funkcie od počiatočnej hodnoty takmer na nulu počas celého tréningu:

```
lr_t = lr_min + 0.5 · (lr_max − lr_min) · (1 + cos(π · t / T_max))
```

Cosine annealing sa vyhýba náhlym poklesom LR, ktoré sú typické pre stupňové plánovače a môžu spôsobovať nestabilitu. Plynulý pokles umožňuje modelu robiť veľké aktualizácie na začiatku (explorácia) a stabílne konvergovať na konci (exploatácia). `T_max` je nastavené na počet epoch, takže kosínus dokončí jeden polcyklus počas tréningu.

---

## Predčasné ukončenie (Early Stopping)

Tréning sa zastaví, ak sa validačné RMSE nezlepší po dobu `patience` po sebe nasledujúcich epoch (predvolene 5). Tým sa predchádza pretrénovaniu po konvergencii a šetrí sa výpočtový čas. Najlepší checkpoint (najnižšie val RMSE) sa obnoví pre záverečné testovanie, takže reportované metriky nepochádzajú z poslednej epochy, ale z epochy s najlepšou generalizáciou.

---

## Automatická zmiešaná presnosť (AMP)

Na CUDA sa tréning vykonáva s `torch.autocast` pomocou `float16` pre väčšinu operácií a `float32` pre numericky citlivé operácie (škálovanie straty, normalizácia). `GradScaler` násobí stratu pred spätným šírením, aby zabránil podtečeniu float16 v gradientoch. AMP približne znižuje spotrebu VRAM na polovicu a urýchľuje výpočty na GPU s tensor core, bez merateľnej straty presnosti.

---

## Akumulácia gradientov

Akumulácia gradientov simuluje väčšiu efektívnu veľkosť dávky bez potreby zmestiť všetky vzorky do VRAM naraz. Gradienty sa akumulujú počas `GRAD_ACCUM_STEPS` mini-dávok pred krokom optimalizátora. Toto bolo použité na lokálnom GPU s 6 GB VRAM (batch=16, steps=2 → efektívny batch=32). Na GPU s 24 GB a batch=64 to už nie je potrebné (`GRAD_ACCUM_STEPS=1`).

---

## Hodnotiace metriky

Všetky metriky sú vypočítané na testovacej množine v pôvodných W/m² (po spätnom vynásobení predpovedí hodnotou 1500):

| Metrika | Vzorec | Interpretácia |
|---------|--------|---------------|
| **MAE** | priemer \|ŷ − y\| | Priemerná absolútna chyba v W/m²; priamo interpretovateľná |
| **RMSE** | √priemer(ŷ − y)² | Penalizuje veľké chyby viac; používa sa pre výber modelu |
| **R²** | 1 − SS_res/SS_tot | Podiel vysvetleného rozptylu; 1.0 = perfektné, 0 = predpovie priemer |

RMSE sa používa ako primárna metrika pre výber checkpointu a porovnanie, pretože je citlivejšia na veľké chyby, ktoré sú v predikcii žiarenia najdôležitejšie.

---

## Experimentálna mriežka

Každá kombinácia architektúra × typ obrazu × rozlíšenie vstupu je trénovaná nezávisle:

| Dimenzia | Hodnoty | Počet |
|----------|---------|-------|
| Architektúra | ResNet18, ResNet50, MobileNetV3-Small, EfficientNet-B0 | 4 |
| Typ obrazu | originálny (fisheye), obdĺžnikový (ekvirektangulárny) | 2 |
| Rozlíšenie vstupu | 224, 256, 320 | 3 |
| **Spolu** | | **24** |

Výsledky sa porovnávajú pomocou tabuľky ΔRMSE (RMSE_obdĺžnikový − RMSE_originálny pre každú kombináciu modelu/rozlíšenia). Záporné hodnoty znamenajú, že ekvirektangulárna transformácia zlepšuje presnosť.
