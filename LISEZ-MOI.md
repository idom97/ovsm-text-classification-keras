# Classification de passages des bulletins OVSM-IPGP par perceptron multicouche

Ce projet a été réalisé dans le cadre du cours **Intelligence artificielle et apprentissage** de la troisième année de Licence Informatique à **l'Université Paris 8**, durant l'année universitaire 2025–2026.

**Auteur :** Dominique ERIN

La version anglaise est disponible dans [`README.md`](README.md).

## Contexte du projet

La Montagne Pelée est un volcan actif situé au nord de la Martinique. Son activité est surveillée par l'**Observatoire volcanologique et sismologique de Martinique (OVSM-IPGP)**, qui publie régulièrement des bulletins de surveillance.

Le projet utilise comme corpus les **12 bulletins mensuels de l'OVSM-IPGP correspondant à l'année 2025**.

Sources OVSM-IPGP utilisées par le projet :

- https://www.ipgp.fr/observation/ovs/ovsm/
- https://www.ipgp.fr/actus-et-agenda/actualites/lovsm-ipgp-preconise-le-passage-en-vigilance-volcanique-jaune-de-la-montagne-pelee/

## Objectif

L'objectif est de construire un classifieur supervisé capable d'associer automatiquement un passage extrait d'un bulletin mensuel de l'OVSM-IPGP à l'une des trois classes suivantes :

- `sismicite_volcanique` ;
- `deformation_sol` ;
- `geochimie`.

Le projet couvre la chaîne allant de l'acquisition et du nettoyage des ressources jusqu'à l'entraînement, l'enregistrement et la réutilisation du modèle pour la classification automatique de passages.

## Principe méthodologique

Les titres des rubriques scientifiques des bulletins sont utilisés uniquement pour **construire automatiquement les étiquettes de référence**. Ces titres sont ensuite retirés des passages fournis au réseau.

Le modèle ne peut donc pas simplement lire un titre tel que *Sismicité volcanique*, *Déformation* ou *Géochimie* pour retrouver la classe attendue. Il doit apprendre les relations entre les caractéristiques textuelles des passages et leurs étiquettes.

## Chaîne générale de traitement

```text
Bulletins mensuels OVSM-IPGP au format PDF
        ↓
extraction du texte
        ↓
nettoyage
        ↓
stockage SQLite
        ↓
segmentation et étiquetage automatiques des passages
        ↓
séparation chronologique entraînement / validation / test
        ↓
vectorisation TF-IDF
        ↓
perceptron multicouche avec Keras
        ↓
entraînement et analyse de la convergence
        ↓
enregistrement et contrôle du rechargement du modèle
        ↓
classification automatique de passages / d'un bulletin entier
```

## Constitution du jeu de données

Le notebook produit **331 passages étiquetés** à partir des 12 bulletins :

| Classe | Nombre de passages |
| --- | ---: |
| Sismicité volcanique | 113 |
| Déformation du sol | 68 |
| Géochimie | 150 |

La séparation est effectuée par bulletin complet, et non aléatoirement passage par passage, afin d'éviter que des formulations très proches issues du même document se retrouvent dans plusieurs partitions.

| Partition | Bulletins | Passages |
| --- | --- | ---: |
| Entraînement | janvier à août 2025 | 206 |
| Validation | septembre à octobre 2025 | 66 |
| Test | novembre à décembre 2025 | 59 |

Le vocabulaire TF-IDF est adapté **uniquement sur les passages d'entraînement**.

## Modèle

Le texte est transformé avec la couche Keras `TextVectorization` en représentation TF-IDF. Le vocabulaire est limité à 6 000 unités et `ngrams=2` permet de prendre en compte les mots isolés ainsi que les groupes de deux mots.

L'architecture du réseau est la suivante :

```text
Texte
  ↓
TF-IDF — 6 000 caractéristiques
  ↓
Dense 64 + ReLU
  ↓
Dropout 30 %
  ↓
Dense 32 + ReLU
  ↓
Dense 3 + Softmax
  ↓
3 probabilités de classe
```

Le réseau comporte **386 243 paramètres entraînables**.

L'entraînement utilise :

- l'optimiseur Adam ;
- un taux d'apprentissage de `1e-3` ;
- la fonction de perte `SparseCategoricalCrossentropy` ;
- l'exactitude catégorielle sparse ;
- une pondération des classes afin de limiter l'effet de leur déséquilibre ;
- un maximum de 60 époques ;
- des lots de 32 passages ;
- un `EarlyStopping` surveillant `val_loss`, avec une patience de 6 époques et restauration des meilleurs poids.

Lors de l'exécution de référence enregistrée dans le notebook, l'entraînement s'arrête à l'époque 9 et restaure les poids de l'époque 3, correspondant à la plus faible perte de validation.

## Périmètre actuel du notebook

La version actuellement publiée couvre :

- l'acquisition, la conversion et le nettoyage du corpus ;
- la constitution automatique des passages étiquetés ;
- la séparation entraînement / validation / test et l'analyse de la convergence ;
- l'enregistrement et le contrôle du rechargement du modèle ;
- la classification de passages fournis à la fonction ;
- l'application du modèle à un bulletin complet stocké dans SQLite, après découpage automatique en passages.

La section d'évaluation finale sur le jeu de test avec `model.evaluate()`, rapport de classification, scores F1 et matrice de confusion **n'est pas présente dans cette version du notebook**. Le dépôt ne présente donc pas de valeur d'exactitude finale sur le jeu de test.

## Classification automatique

La fonction réutilisable recharge le modèle entraîné et retourne, pour chaque passage :

- la classe la plus probable ;
- la probabilité maximale ;
- les probabilités associées aux trois classes.

Un seuil de confiance de `0.60` est appliqué. Lorsque la probabilité maximale est inférieure à ce seuil, la sortie affichée devient `a_verifier`.

Ce seuil permet de signaler une incertitude, mais il ne crée pas une quatrième classe « hors sujet ». Le modèle doit donc être utilisé sur des textes relevant du même domaine que le corpus d'apprentissage.

Le notebook contient également une fonction qui récupère un bulletin complet dans la base SQLite, retire les titres manifestes de rubriques, découpe le texte en passages puis classe chacun de ces passages séparément.

## Organisation du dépôt

```text
projet/
├── .gitignore
├── README.md
├── LISEZ-MOI.md
├── LICENSE.md
├── requirements.txt
├── notebook_IA_apprentissage.ipynb
├── corpus/
├── data/
├── figures/
└── modeles/
```

Les contenus de `corpus/`, `data/`, `figures/` et `modeles/` sont produits lors de l'exécution et sont exclus de Git par `.gitignore` lorsqu'ils peuvent être recréés à partir du notebook.

## Reproductibilité

Les métadonnées du notebook indiquent **Python 3.11.6** et l'exécution de référence affiche **TensorFlow 2.21.0**. Une graine aléatoire de `42` est fixée pour NumPy et TensorFlow.

Pour une reproductibilité maximale, `requirements.txt` doit contenir les versions exactes des bibliothèques installées dans l'environnement ayant servi à produire l'exécution de référence. De légères différences numériques peuvent néanmoins apparaître selon le matériel ou le moteur d'exécution TensorFlow en raison des opérations en virgule flottante.

Avant la première exécution, vérifier que les répertoires attendus existent :

```bash
mkdir -p corpus/pdf corpus/txt_brut corpus/txt_propre data figures modeles
```

## Installation

Créer et activer un environnement virtuel :

```bash
python -m venv .venv
source .venv/bin/activate
```

Installer les dépendances :

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Lancer JupyterLab :

```bash
jupyter lab
```

Ouvrir ensuite `notebook_IA_apprentissage.ipynb` et exécuter les cellules dans l'ordre.

La première partie du notebook télécharge les bulletins mensuels 2025 de l'OVSM-IPGP lorsqu'ils ne sont pas déjà présents localement.

## Fichiers volontairement exclus de Git

Le fichier `.gitignore` exclut les éléments reproductibles générés par le notebook, notamment :

- les PDF téléchargés ;
- les textes bruts et nettoyés ;
- la base SQLite et les métadonnées générées ;
- les fichiers CSV produits pendant le traitement ;
- l'historique d'entraînement et les exports de prédictions ;
- les figures générées ;
- les modèles `.keras` et leurs métadonnées ;
- les caches Python/Jupyter et les environnements virtuels locaux.

Le dépôt reste ainsi centré sur le code et la documentation nécessaires pour reproduire le traitement, plutôt que sur les résultats générés.

## Limites

Le projet présente plusieurs limites :

- le corpus est limité à 12 bulletins mensuels ;
- les trois classes sont déséquilibrées ;
- seuls deux bulletins sont réservés à la validation et deux au test ;
- la construction automatique des étiquettes dépend de la reconnaissance de titres de rubriques connus et suppose donc une structure de bulletin relativement stable ;
- si l'OVSM-IPGP modifie les intitulés ou la structure des bulletins, les règles d'extraction et de segmentation devront probablement être adaptées ;
- une probabilité élevée produite par le réseau ne garantit pas qu'une prédiction soit correcte ;
- la version actuelle du notebook ne contient pas de section d'évaluation finale indépendante sur le jeu de test.

Une évolution possible consisterait à dépasser la classification de passages pour proposer une synthèse automatique destinée à rendre le contenu scientifique des bulletins plus accessible à un public non spécialiste.

## Licence

Ce dépôt est distribué selon les conditions décrites dans [`LICENSE.md`](LICENSE.md).

Il **n'est pas publié sous une licence open source**. Le téléchargement, le clonage, l'exécution et les tests sont autorisés pour l'évaluation personnelle, la revue pédagogique et le recrutement, mais la redistribution, la création d'œuvres dérivées et l'utilisation commerciale nécessitent l'autorisation écrite préalable de l'auteur.
