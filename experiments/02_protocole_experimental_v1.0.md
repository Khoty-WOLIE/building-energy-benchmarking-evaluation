# Protocole expérimental — Article 01 (Seattle) — v1.0

**Statut :** **GELÉ le 2026-10-29** (validation de Khoty, décisions A01-D17 à D20). Toute déviation est consignée au §12 avec sa date et sa justification, avant d'en examiner l'effet.
**Date :** 2026-10-29
**Configuration machine :** `experiments/config/protocol_v1.yaml`
**Code déjà implémenté et testé :** `src/a01/data.py`, `src/a01/splits.py`, `tests/test_splits_and_tiers.py` (8 tests), `src/prepare_phase03.py`
**Chiffres de référence :** `results/phase03_*.csv|json` (générés depuis l'instantané figé)

**Titre de travail :** *Which Data Vintage? Data Revisions, Information Availability and Evaluation Design in Building Energy Prediction from Public Benchmarking Data — Evidence from Seattle, 2015–2025*

---

## 1. Questions de recherche et hypothèses opérationnelles

**Questions de recherche**

| Code | Question | Contribution |
|---|---|---|
| **RQ1** (principale) | La version (le « millésime ») des données publiques change-t-elle les performances mesurées et les conclusions comparatives ? | C3 |
| **RQ2** | Que valent réellement les informations disponibles selon le scénario d'usage (T0 / T1), et quelle part du « gain » ENERGY STAR relève de la fuite ? | C4 + C1 |
| **RQ3** | Dans quelle mesure le protocole d'évaluation (aléatoire, groupé, temporel) modifie-t-il la performance estimée ? | C2 (cadre) |

**Hypothèses**

| Code | Hypothèse (formulée pour être réfutable) | Expérience | Contraste principal |
|---|---|---|---|
| **HV1** | Le même protocole appliqué aux publications 2016 d'origine et consolidée donne des métriques différentes. L'écart est plus grand pour la cible GES (révisée pour 90 % des bâtiments) que pour l'énergie (1,9 %). | E6a | P6, P7 |
| **HV2** | Remplacer les attributs instantanés de la version consolidée par les attributs « à date » (publication 2016 d'origine) modifie l'erreur, cible fixée. | E6b | P8 |
| H1 | L'erreur en découpage groupé par bâtiment (S2) est plus élevée qu'en découpage aléatoire par ligne (S1). | E1 | P1, P2 |
| H2 | L'erreur croît avec l'horizon temporel h (S3). | E2 | P3 |
| H3 | Sur des bâtiments jamais vus (S4), le modèle complexe ne surpasse pas nettement Ridge. | E3 | P4 |
| H4 | L'erreur est hétérogène selon le type, la taille, l'âge et le quartier. | E4 | descriptif |
| H5 | Ajouter ENERGY STAR à T0 réduit fortement l'erreur, même en S2, et ce gain est porté par l'information sur la consommation mesurée. | E5 | P5 |

---

## 2. Données

### 2.1 Sources (figées)

| Source | Identifiant | Empreinte SHA-256 | Rôle |
|---|---|---|---|
| Seattle « Building Energy Benchmarking Data, 2015–Present » | `teqw-tu6e`, téléchargé le 2026-10-29 | `d1a8227b…250f` (vérifiée par le code à chaque exécution) | Panel principal |
| Publication 2016 d'origine (ex-`2bpz-gwpy`, copie OpenClassrooms authentifiée) | — | `8bcb3dc9…3a9d` | Millésime de comparaison (E6) |

### 2.2 Population et filtres a priori

Les filtres ne reposent que sur des métadonnées, **jamais** sur la distribution de la cible. Attrition mesurée (`results/phase03_attrition.csv`) :

| Étape | Lignes | Bâtiments |
|---|---|---|
| Toutes les lignes 2015–2025 | 38 309 | 3 871 |
| Non résidentiel (`BuildingType` ≠ Multifamily*) | 17 392 | 1 744 |
| `ComplianceStatus == Compliant` | 16 126 | 1 736 |
| Non démoli | 15 865 | 1 668 |
| Cible présente et > 0 | **15 621** | **1 667** |

Lignes par année : 1 336 à 1 521. Les années **2020–2021** (COVID) sont marquées `is_covid_year` et analysées séparément en sensibilité (§6).

Différences assumées avec le projet d'origine :
- aucun retrait de valeurs extrêmes de la cible ;
- aucune imputation fondée sur des moyennes calculées hors du pli d'entraînement ;
- les bâtiments multi-usages et les campus sont conservés.

### 2.3 Cibles

| Rôle | Variable | Commentaire |
|---|---|---|
| **Principale** | `y = log(SiteEnergyUse kBtu)` | |
| Secondaire | `log(TotalGHGEmissions)` | Lignes avec GES > 0 ; utilisée dans E6, cible fortement révisée |
| Sensibilité | `log(SiteEnergyUseWN)` | Normalisée météo |

---

## 3. Niveaux d'information (variables)

| Niveau | Scénario d'usage | Variables |
|---|---|---|
| **T0** — ex ante | Bâtiment n'ayant jamais déclaré | `log_gfa_total`, `log_gfa_buildings`, `parking_share`, `NumberofFloors`, `NumberofBuildings`, `YearBuilt`, `Latitude`, `Longitude`, `largest_use_share`, `BuildingType`, `property_type` (EPAPropertyType, 15 types + « Other (grouped) » pour les types de moins de 30 bâtiments), `Neighborhood` (14, harmonisé). Aucune valeur manquante. |
| **T1** — premier relevé | Bâtiment disposant d'une année de mesure de référence (scénario du cahier des charges d'origine) | T0 + parts électricité / gaz / vapeur et `log` de l'énergie de la **première année observée strictement antérieure**, plus l'écart en années. En protocole temporel, l'année de référence doit être ≤ la date de coupure. Disponible pour 89,3 % des lignes. Gaz ou vapeur manquant = 0 (hypothèse documentée). |
| **T2** — dérivé de la cible | Aucun scénario de déploiement : **analyse de fuite uniquement** | `ENERGYSTARScore`, `SiteEUI`, `SourceEUI`, `SiteEUIWN`, `SourceEUIWN`, `GHGEmissionsIntensity`, consommations électricité / gaz / vapeur de la même année (leur somme égale la cible : ratio médian 1,000) |

**Comparabilité T0 / T1** : les comparaisons entre niveaux se font sur les **mêmes lignes de test**, celles qui disposent d'une référence T1.

**Prétraitement** : il est toujours ajusté sur le pli d'entraînement seul, au sein d'un pipeline scikit-learn :
- one-hot `handle_unknown='ignore'` ;
- standardisation pour Ridge ;
- aucun prétraitement pour les arbres hormis l'encodage.

---

## 4. Protocoles d'évaluation (découpages)

| Code | Définition | Paramètres |
|---|---|---|
| S1 `row_random` | K plis sur les bâtiments-années (un même bâtiment apparaît des deux côtés) | k = 5, 5 répétitions |
| S2 `grouped` | K plis sur les bâtiments | k = 5, 5 répétitions |
| S3 `forward_temporal` | Entraînement sur les années ≤ t, test sur l'année t + h | t ∈ {2017…2024}, h ∈ {1, 2, 3} |
| S4 `grouped_temporal` | S3 avec des bâtiments de test jamais vus à l'entraînement | idem S3, k = 5 groupes |

- Graine de base : `20261029`.
- Invariants vérifiés par tests unitaires : pas de bâtiment commun en S2 et S4 ; pas d'année future à l'entraînement en S3 et S4 ; référence T1 antérieure et plafonnée.
- **Réglage des hyperparamètres** : imbriqué, sur le pli d'entraînement seulement.
  - En S1, la validation croisée interne est aléatoire : on reproduit volontairement la pratique naïve.
  - En S2, S3 et S4, elle est groupée par bâtiment (3 plis).

---

## 5. Modèles et grilles (fixés a priori)

| Code | Modèle | Grille |
|---|---|---|
| M0 | Médiane de `y` par `property_type` (référence naïve) | — |
| M1 | Persistance : `log` énergie de la dernière année antérieure (≤ coupure) — **niveau T1 uniquement** | — |
| M2 | Ridge | α ∈ {0,01 ; 0,1 ; 1 ; 10 ; 100} |
| M3 | RandomForestRegressor | `n_estimators = 500` ; `max_features ∈ {0,33 ; 0,66}` ; `min_samples_leaf ∈ {1, 5}` |
| M4 | HistGradientBoostingRegressor | `learning_rate ∈ {0,05 ; 0,1}` ; `max_leaf_nodes ∈ {15, 31}` ; `max_iter = 500` avec arrêt précoce interne |

Aucun autre modèle n'est ajouté sans une entrée au journal des déviations. Le modèle « complexe » de référence pour les contrastes est **M4**.

---

## 6. Expériences

| Exp. | Contenu | Niveaux | Découpages | Sortie principale |
|---|---|---|---|---|
| **E0** | Audit des millésimes : statistiques de révision (GES, surfaces, types, population), attributs figés du panel. Déjà réalisé en phases 01–02, à reformater pour l'article. | — | — | Tableau 1 |
| **E1** | Optimisme du découpage aléatoire | T0, T1 | S1 vs S2 | ΔMAE (P1, P2) |
| **E2** | Dégradation avec l'horizon ; sensibilité COVID (tests 2020–2021 exclus ou isolés) | T0, T1 (+ M1) | S3 | courbe MAE(h) (P3) |
| **E3** | Bâtiments jamais vus et années futures ; complexe vs simple | T0, T1 | S4 | M4 − M2 (P4) |
| **E4** | Erreurs par sous-groupe (type, quintile de surface, décennie de construction, quartier) ; décalage de covariables par validation adversariale (AUC entraînement vs test, par coupure) | T0 | S4 | Figure d'hétérogénéité |
| **E5** | Fuite ENERGY STAR : T0 vs T0 + ES (+ indicateur de manquant) vs T1 vs T1 + ES vs T0 + EUI (plafond). Sur les lignes avec ES, puis sur toutes les lignes. Diagnostic complémentaire : corrélation de Spearman entre ES et `log(SourceEUI)` par type de propriété. | T0, T1, T2 | S2, S4 | Δ(T0 + ES − T0) (P5) |
| **E6a** | Même protocole sur deux millésimes de 2016 : publication d'origine vs lignes 2016 consolidées, restreint aux **1 343 bâtiments appariés** de la population d'analyse (puis chaque population complète en sensibilité). Cibles énergie et GES. Variables T0 communes : surfaces, étages, nombre de bâtiments, année de construction, coordonnées, quartier, `LargestPropertyUseType` regroupé. | T0 commun | S2 | écart de métriques, stabilité du classement (P6, P7) |
| **E6b** | Sensibilité aux attributs instantanés : cible consolidée 2016 fixée ; attributs T0 communs pris dans la publication d'origine (« à date ») vs consolidée (« instantané ») | T0 commun | S2 | ΔMAE (P8) |
| **E6c** | Mise en perspective du projet d'origine : son résultat reproduit (LR R² log = 0,5345, découpage aléatoire unique, troncature des valeurs extrêmes) vs le protocole v1.0 sur la même publication | — | S1 / S2 | tableau d'écart |

---

## 7. Métriques

Les métriques sont calculées par instance de découpage, puis agrégées.

| Rôle | Métrique |
|---|---|
| **Principale** | **MAE sur `y` (log)**, interprétable comme une erreur relative typique |
| Secondaires | RMSE log ; R² log ; **MdAPE** sur l'échelle d'origine ; MAE en kBtu après correction de retransformation de Duan (*smearing*), dont le facteur est estimé sur le pli d'entraînement |

Pour E6a / E6b : écart de MAE log ; corrélation de rang de Spearman entre les classements de modèles des deux millésimes.

---

## 8. Analyse statistique

- **Incertitude** : bootstrap par grappes de bâtiments, B = 2 000, IC percentile à 95 %. Les répétitions de validation croisée (S1 / S2) sont moyennées avant le bootstrap.
- **Contrastes appariés** : différence calculée sur les **mêmes lignes de test**, rééchantillonnées ensemble.
- **Contrastes principaux** (8, avec correction de Holm) :

| Code | Contraste |
|---|---|
| P1 | MAE(S2) − MAE(S1), M4, T0 |
| P2 | MAE(S2) − MAE(S1), M4, T1 |
| P3 | MAE(S3, h = 3) − MAE(S3, h = 1), M4, T0 |
| P4 | MAE(M4) − MAE(M2), S4, T0 |
| P5 | MAE(T0 + ES) − MAE(T0), S2, M4, lignes avec ES |
| P6 | MAE(origine) − MAE(consolidé), cible GES, S2, M4, T0 commun, 1 343 bâtiments |
| P7 | idem P6, cible énergie |
| P8 | MAE(attributs à date) − MAE(attributs instantanés), S2, M4, T0 commun |

- **Seuil pratique** : un effet n'est qualifié de « substantiel » que si l'IC à 95 % (après Holm) exclut 0 **et** si |Δ MAE log| ≥ **0,02** (≈ 2 % d'erreur relative).
- **Analyses secondaires** : rapportées avec leurs IC, sans correction, et étiquetées comme exploratoires.

---

## 9. Règles de décision et résultats négatifs

| Hypothèse | Soutenue si | Réfutée si |
|---|---|---|
| HV1 | P6 ou P7 substantiel | les deux IC incluent 0 → « les révisions n'affectent pas les métriques » — résultat publiable qui recentre C3 sur la documentation des révisions |
| HV2 | P8 substantiel | idem → l'instantané d'attributs est sans effet mesurable sur une coupe transversale |
| H1 | P1 > 0 substantiel | — |
| H2 | P3 > 0 substantiel | — |
| H3 | P4 n'est pas significativement < −0,02 | M4 bat M2 de plus de 0,02 |
| H5 | P5 < 0 substantiel. **Interprétation fixée a priori** : un gain dû à ES est classé comme fuite, car ES = f(EUI source réel) (EPA, 2021) ; il ne sera **pas** présenté comme une valeur prédictive déployable. | — |

Tous les résultats sont rapportés, y compris les hypothèses réfutées.

---

## 10. Reproductibilité et traçabilité

- **Environnement** : Python 3.11.9, `requirements-phase01.lock.txt` (à compléter en phase 04 par un lock final incluant pytest et pyyaml).
- **Données** : contrôle SHA-256 bloquant à chaque exécution (`prepare_phase03.py`).
- **Graines** : base `20261029`, dérivées par répétition (+r).
- **Sorties par exécution** : `results/experiments/<EXP>/<run_id>/` contenant `config.yaml` (copie), `metrics.csv` (une ligne par instance × modèle × niveau), `predictions.parquet`, `env.txt` (`pip freeze`, matériel), `data_sha256.txt`, `commit.txt` (dès que le dépôt local est initialisé).
- **Figures** : régénérées par un script unique depuis `metrics.csv` et `predictions`.

---

## 11. Menaces à la validité identifiées a priori

1. **Les deux millésimes 2016 diffèrent aussi par la population** (182 / 59 bâtiments non appariés) : E6a est restreint aux bâtiments appariés.
2. **Révisions non attribuées** : la cause des révisions n'est pas documentée (aucun contact avec la Ville, décision A01-D14). L'article décrit et mesure, sans attribuer.
3. **Attributs figés** : dans le panel consolidé, les découpages S3 / S4 utilisent des attributs instantanés. E6b quantifie l'effet sur 2016 seulement ; l'extrapolation aux autres années reste une limite.
4. **Changements de type** : `EPAPropertyType` varie pour 19,7 % des bâtiments multi-années. Il est traité comme un attribut annuel (pas de rétro-propagation).
5. **COVID 2020–2021** : rupture d'usage, traitée en sensibilité.
6. **Une seule ville** : la généralisation à d'autres villes n'est pas revendiquée (GHGbench couvre l'axe multi-villes).
7. **Biais de retransformation** : traité par la correction de Duan, avec la MdAPE comme métrique robuste.

---

## 12. Journal des déviations

| Date | Déviation | Justification | Effet connu au moment de la décision ? |
|---|---|---|---|
| 2026-10-29 | **Ajout (analyse secondaire, exploratoire)** : E6 étendu en plan factoriel 2 × 2 (attributs {origine, consolidé} × cible {origine, consolidée}), pour décomposer l'écart E6a en part « attributs » et part « cible ». Les contrastes principaux P6–P8 sont inchangés. | Décomposition naturelle de C3 | **Non** — décidé avant toute exécution de modèle |
| 2026-10-29 | **Précision** : E5 est exécutée avec M0, M2 et M4 (le §6 ne fixait pas les modèles de E5 ; P5 porte sur M4). Les modèles reçoivent ES avec une imputation médiane ajustée sur le pli d'entraînement et un indicateur `es_missing`. | Coût de calcul ; P5 inchangé | **Non** |
| 2026-10-29 | **Précision** : une imputation médiane, ajustée sur le pli d'entraînement, est ajoutée devant toutes les variables numériques (nécessaire pour la publication 2016 d'origine et pour ES ; sans effet sur T0 consolidé, qui n'a aucune valeur manquante) | Robustesse | **Non** |
| 2026-10-29 (après les résultats de la phase 04) | **Ajout exploratoire X1 — test d'« empreinte »** : S1 et S2 au niveau T0 avec attributs continus bruités indépendamment pour chaque ligne (surfaces ×(1 + N(0 ; 0,02)), coordonnées + N(0 ; 0,002°), année de construction ± 2 ans), afin de casser l'identification exacte du bâtiment sans détruire l'information. 2 répétitions × 5 plis ; modèles M2, M3, M4. **Prédiction** si le mécanisme est réel : l'erreur S1 des arbres remonte fortement, et S2 change peu. | Étayer l'interprétation de P1 (A01-D21) | **Oui** — décidé après avoir vu P1 ; résultat étiqueté exploratoire, hors Holm |
| 2026-10-29 (après les résultats de la phase 04) | **Ajout exploratoire X2 — T1 « dernière année »** : la référence T1 est la dernière année observée ≤ coupure (au lieu de la première), en S4, modèles M2 et M4, comparés à la persistance M1 | Comparaison à information égale (A01-D21) | **Oui** — idem |
| 2026-10-29 (relecture critique, phase 06) | **Ajout exploratoire X3 — compétence relative en E6** : MAE de M4 rapportée à la MAE du modèle naïf M0 de la même cellule, contraste apparié origine − consolidé ; écart-type de la cible par millésime. Objectif : distinguer un changement de difficulté d'un simple changement de dispersion de la cible. Aucun nouvel ajustement de modèle (prédictions E6 existantes). | Question de « reviewer » sur P6 | **Oui** — décidé après P6 ; exploratoire |
| 2026-10-29 | **Précision** : le facteur de Duan est la moyenne de exp(résidus) **sur le pli d'entraînement** (prédictions en échantillon). Pour les modèles à arbres, ces résidus sont optimistes : la MAE en kBtu reste une métrique secondaire, avec cette limite signalée. | Le protocole ne précisait pas quels résidus utiliser | **Non** |

---

## 13. Budget de calcul (estimation, à confirmer par un essai de chronométrage en phase 04)

Environ 15 600 lignes. Ordre de grandeur des ajustements :
- S1 / S2 : 5 répétitions × 5 plis × 3 modèles réglés × niveaux ;
- S3 : 21 instances ;
- S4 : 105 instances ;
- réglage interne à 3 plis sur 4 à 8 combinaisons.

Estimation : quelques heures sur processeur, sans GPU. L'essai de chronométrage précédera tout lancement complet.
