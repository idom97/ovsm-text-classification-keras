# OVSM-IPGP Passage Classification with a Multilayer Perceptron

This project was carried out as part of the **Artificial Intelligence and Learning** course in the third year of the Bachelor's Degree in Computer Science at **Université Paris 8** during the 2025–2026 academic year.

**Author:** Dominique ERIN

A French version is available in [`LISEZ-MOI.md`](LISEZ-MOI.md).

## Project Context

Mount Pelée is an active volcano located in northern Martinique. Its activity is monitored by the **Volcanological and Seismological Observatory of Martinique (OVSM-IPGP)**, which publishes periodic monitoring bulletins.

This project uses the **12 monthly OVSM-IPGP bulletins published for 2025** as its textual corpus.

OVSM-IPGP sources used by the project:

- https://www.ipgp.fr/observation/ovs/ovsm/
- https://www.ipgp.fr/actus-et-agenda/actualites/lovsm-ipgp-preconise-le-passage-en-vigilance-volcanique-jaune-de-la-montagne-pelee/

## Objective

The objective is to build a supervised text classifier capable of assigning a passage extracted from an OVSM-IPGP monthly bulletin to one of three classes:

- `sismicite_volcanique` — volcanic seismicity;
- `deformation_sol` — ground deformation;
- `geochimie` — geochemistry.

The project follows the complete workflow from resource acquisition and cleaning to model training, persistence, and reuse for automatic passage classification.

## Methodological Principle

The scientific section headings found in the bulletins are used only to **construct the reference labels automatically**. These headings are removed from the passages before they are supplied to the neural network.

The model therefore cannot simply read a heading such as *Sismicité volcanique*, *Déformation* or *Géochimie* to infer the expected class. It must learn relationships between the textual characteristics of the passages and their labels.

## Processing Workflow

```text
OVSM-IPGP monthly PDFs
        ↓
PDF text extraction
        ↓
text cleaning
        ↓
SQLite storage
        ↓
automatic passage segmentation and labelling
        ↓
chronological train / validation / test split
        ↓
TF-IDF vectorization
        ↓
Multilayer Perceptron (Keras)
        ↓
training and convergence analysis
        ↓
model saving and reload verification
        ↓
automatic classification of passages / a complete bulletin
```

## Dataset Construction

The notebook generates **331 labelled passages** from the 12 monthly bulletins:

| Class | Passages |
| --- | ---: |
| Volcanic seismicity | 113 |
| Ground deformation | 68 |
| Geochemistry | 150 |

The split is performed by complete bulletin rather than randomly by passage in order to avoid placing very similar passages from the same document in different subsets.

| Subset | Bulletins | Passages |
| --- | --- | ---: |
| Training | January–August 2025 | 206 |
| Validation | September–October 2025 | 66 |
| Test | November–December 2025 | 59 |

The TF-IDF vocabulary is adapted **only on the training passages**.

## Model

The text is vectorized with Keras `TextVectorization` using TF-IDF, a maximum vocabulary size of 6,000 units, and unigrams/bigrams (`ngrams=2`).

The neural network architecture is:

```text
Text
  ↓
TF-IDF — 6,000 features
  ↓
Dense 64 + ReLU
  ↓
Dropout 30%
  ↓
Dense 32 + ReLU
  ↓
Dense 3 + Softmax
  ↓
3 class probabilities
```

The network contains **386,243 trainable parameters**.

Training uses:

- Adam optimizer;
- learning rate `1e-3`;
- `SparseCategoricalCrossentropy` loss;
- sparse categorical accuracy;
- class weighting to reduce the effect of class imbalance;
- a maximum of 60 epochs;
- batch size 32;
- `EarlyStopping` monitoring `val_loss` with a patience of 6 epochs and restoration of the best weights.

In the reference notebook run, training stopped after epoch 9 and restored the weights from epoch 3, which had the lowest validation loss.

## Current Scope of the Notebook

The current repository version focuses on:

- corpus acquisition, conversion and cleaning;
- automatic construction of labelled passages;
- training/validation splitting and convergence analysis;
- model saving and reload verification;
- classification of supplied passages;
- application of the saved model to a complete bulletin stored in SQLite, after automatic segmentation into passages.

A final test-set evaluation block using `model.evaluate()`, a classification report, F1 scores and a confusion matrix is **not included in this version of the notebook**. The repository therefore does not claim a final test accuracy.

## Automatic Classification

The reusable classification function reloads the trained model and returns, for each passage:

- the most probable class;
- the maximum probability;
- the probabilities associated with all three classes.

A confidence threshold of `0.60` is used. If the highest probability is below this threshold, the displayed prediction becomes `a_verifier`.

This threshold indicates uncertainty; it does not create a fourth *out-of-domain* class. The classifier should therefore be applied to passages belonging to the same domain as the training corpus.

The notebook also includes a function that retrieves a complete bulletin from the SQLite database, removes obvious section headings, segments the text into passages, and classifies each passage individually.

## Repository Structure

```text
project/
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

The contents of `corpus/`, `data/`, `figures/`, and `modeles/` are generated during execution and are excluded from Git by `.gitignore` when they can be recreated from the notebook.

## Reproducibility

The notebook metadata records **Python 3.11.6**, and the reference run reports **TensorFlow 2.21.0**. A random seed of `42` is set for NumPy and TensorFlow.

For the most reproducible environment, `requirements.txt` should contain the exact package versions installed in the environment used for the reference run. Minor numerical differences may still occur across hardware or TensorFlow execution backends because of floating-point operations.

Before the first execution, make sure the expected directories exist:

```bash
mkdir -p corpus/pdf corpus/txt_brut corpus/txt_propre data figures modeles
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Start JupyterLab:

```bash
jupyter lab
```

Then open `notebook_IA_apprentissage.ipynb` and execute the cells in order.

The first part of the notebook downloads the 2025 OVSM-IPGP monthly bulletins when they are not already available locally.

## Files Intentionally Excluded from Git

The `.gitignore` excludes reproducible generated artefacts such as:

- downloaded PDFs;
- raw and cleaned text files;
- SQLite databases and generated metadata;
- generated CSV files;
- training-history files and prediction exports;
- generated figures;
- trained `.keras` models and model metadata;
- Python/Jupyter caches and local virtual environments.

This keeps the repository focused on the code and documentation required to reproduce the workflow rather than on generated outputs.

## Limitations

The project has several limitations:

- the corpus contains only 12 monthly bulletins;
- the three classes are imbalanced;
- only two complete bulletins are reserved for validation and two for testing;
- automatic label construction depends on detecting known section headings and therefore assumes a relatively stable bulletin structure;
- if OVSM-IPGP changes section names or document structure, the extraction/segmentation rules may need to be updated;
- a high model probability does not guarantee that a prediction is correct;
- the current notebook does not include a final independent test-set evaluation section.

A possible extension would be to move beyond passage classification toward automatic summarization designed to make technically dense scientific bulletins more accessible to non-specialist readers.

## License

This repository is distributed under the terms described in [`LICENSE.md`](LICENSE.md).

It is **not released under an open-source license**. Downloading, cloning, running and testing are permitted for personal evaluation, educational review and recruitment purposes, but redistribution, derivative works and commercial use require prior written permission from the author.
