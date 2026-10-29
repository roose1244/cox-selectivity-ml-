# Predicting COX-1/COX-2 Selectivity Using Machine Learning

## Overview

This project investigates whether molecular chemical and structural features can be used to predict the selectivity of non-steroidal anti-inflammatory drugs (NSAIDs) for cyclooxygenase-1 (COX-1) versus cyclooxygenase-2 (COX-2).

The project combines experimental bioactivity data, cheminformatics, machine learning and explainable AI to investigate both **whether selectivity can be predicted from molecular structure** and **which molecular characteristics contribute to selective inhibition**.

### Research Question

> Can machine learning predict the COX-1/COX-2 selectivity of NSAIDs from their chemical and structural features, and can explainable AI identify the molecular characteristics associated with selective inhibition?

---

## Scientific Background

Cyclooxygenase enzymes are important targets of NSAIDs, which reduce prostaglandin synthesis through inhibition of cyclooxygenase activity.

The two major isoforms are:

- **COX-1 (PTGS1)** — constitutively expressed in many tissues and involved in physiological processes including gastric protection and platelet function.
- **COX-2 (PTGS2)** — commonly associated with inducible inflammatory responses, although it also has physiological roles.

NSAIDs differ in their relative inhibition of COX-1 and COX-2. Understanding the molecular features associated with this selectivity is relevant to drug design and the investigation of differences in pharmacological activity.

---

## Project Objectives

1. Obtain experimentally measured COX-1 and COX-2 activity data for small molecules.
2. Curate the dataset to retain comparable experimental measurements.
3. Identify compounds with measurements against both COX-1 and COX-2.
4. Calculate a COX-1/COX-2 selectivity measure.
5. Generate molecular descriptors and structural fingerprints using RDKit.
6. Train machine-learning models to predict selectivity.
7. Evaluate model performance using appropriate validation methods.
8. Use explainable AI to investigate which molecular features contribute to model predictions.
9. Explore molecular structure and binding interactions as a potential future validation step.

---

## Methodology

The planned workflow is:

```text
Experimental Bioactivity Data
            │
            ▼
       Data Curation
            │
            ▼
   Human COX-1 / COX-2
            │
            ▼
      Paired Activity
       Measurements
            │
            ▼
    Selectivity Calculation
            │
            ▼
        RDKit
      ┌─────┴─────┐
      ▼           ▼
 Descriptors   Morgan
              Fingerprints
      │           │
      └─────┬─────┘
            ▼
     Machine Learning
            │
            ▼
   Selectivity Prediction
            │
            ▼
   Explainable AI (SHAP)
            │
            ▼
 Molecular Feature Analysis
```

### 1. Experimental Data

The data source is the **ChEMBL REST API**, using experimentally measured
protein–small-molecule activity data. ChEMBL was chosen over BindingDB because it
exposes a documented, paginated REST API that can be queried reproducibly from
code, and because it carries the assay, publication and data-validity metadata
this project needs in order to audit assay heterogeneity.

The analysis focuses on:

- Human COX-1 / PTGS1 — ChEMBL target `CHEMBL221` (UniProt P23219)
- Human COX-2 / PTGS2 — ChEMBL target `CHEMBL230` (UniProt P35354)
- Human single-protein targets only (`target_type == SINGLE PROTEIN`, `organism == Homo sapiens`)
- Small-molecule compounds with a canonical SMILES string
- Paired measurements against both targets

Raw, unmodified activity tables are written to `data/raw/` alongside
`data/raw/provenance.json`, which records the ChEMBL release and release date,
the exact query parameters, the retrieval timestamp and the saved row counts.
Record totals are read from ChEMBL's own `page_meta` at download time and are
never hard-coded.

### 1a. Endpoint decision

**Decision: exact (`=`) IC50 values reported in nM are the primary activity
endpoint for this project.** An earlier draft of this README proposed Ki. That
has been revised, and the two are not mixed.

Rationale:

- IC50 and Ki are different physical quantities. Ki is an equilibrium
  dissociation constant; IC50 is a functional potency that depends on substrate
  concentration, enzyme concentration and assay format. Pooling them would
  create an artificial spread unrelated to molecular structure.
- IC50 is by far the better-populated endpoint for both COX isoforms in this
  ChEMBL release. The exact per-step counts are logged in
  `data/processed/filter_log.csv`.
- It matches what the existing exploratory notebook already examined, so the
  earlier exploration remains valid rather than being discarded.

Ki, AC50 and percent-inhibition records are retained in `data/raw/` and may be
analysed separately later; they are excluded from the curated IC50 dataset.

Censored measurements (`>`, `<`, `>=`, `<=`, `~`, or a missing relation) are
**excluded** from this analysis, because a value such as "IC50 > 100000 nM" does
not give a usable potency for a regression target. The number of censored
records dropped at that step is recorded in `data/processed/filter_log.csv` so
the exclusion is visible rather than silent. Censored data could be handled
properly later with survival-style (tobit) regression; that is out of scope for
the first pass.

### 2. Potency and selectivity definitions

Potency is expressed on a logarithmic scale, where larger values mean more
potent inhibition:

```text
pIC50 = 9 - log10(IC50_nM)
```

Only positive IC50 values in nM are transformed. The selectivity target is:

```text
selectivity = log10(IC50_COX-1 / IC50_COX-2) = pIC50_COX-2 - pIC50_COX-1
```

**Orientation: positive values mean the molecule inhibits COX-2 more strongly
than COX-1 (COX-2 selective); negative values mean COX-1 selective.** This is
carried in the column named
`selectivity_log10_IC50_cox1_over_cox2` so the orientation is readable from the
data file itself and cannot be confused with the reverse ratio.

#### Handling replicate measurements

A molecule frequently has several IC50 records against the same target, from
different publications and assay formats. One defensible value per
molecule–target pair is taken as the **median pIC50** of its replicates. The
median is preferred to the mean because replicate potencies drawn across
publications are skewed and occasionally contain order-of-magnitude
transcription errors, which the median resists.

Nothing is averaged blindly. For every molecule–target pair the curated output
retains the number of records, the number of distinct assays, the number of
distinct source publications, and the minimum, maximum, range and standard
deviation of the replicate pIC50 values. Pairs whose replicate range exceeds one
log unit are flagged (`high_spread_cox1`, `high_spread_cox2`,
`either_high_spread`) so that poorly reproducible molecules can be inspected or
excluded in sensitivity analyses rather than quietly entering the model.

#### Comparability limits

ChEMBL assay IDs are target-specific, so a COX-1 and a COX-2 measurement can
never share an assay ID. The meaningful unit of comparability is the source
publication: a pair measured in the same paper is much more likely to share
enzyme source, substrate concentration, incubation time and readout. The paired
table therefore records `same_publication_pair`, `n_shared_documents` and the
shared document IDs.

This matters for interpretation. A selectivity value assembled from two
different papers mixes assay effects into the ratio, and that mixing is not
something the molecular structure can explain. Any model trained on the full
paired set should be checked against the same-publication subset, and the
difference reported.

### 3. Molecular Features

Chemical structures will be processed using **RDKit**.

Features will include molecular descriptors describing properties such as:

- Molecular weight
- Hydrogen-bond donors and acceptors
- Rotatable bonds
- Aromaticity
- Molecular composition
- Other calculated physicochemical descriptors

Morgan fingerprints will also be investigated to represent local structural patterns within each molecule.

### 4. Machine Learning

Machine-learning models will be evaluated for their ability to predict COX selectivity from molecular features.

Potential approaches include:

- Linear regression
- Random forest
- Gradient boosting
- Other appropriate models depending on dataset size and characteristics

Model performance will be evaluated using held-out data and appropriate regression metrics.

### 5. Explainable AI

**SHAP (SHapley Additive exPlanations)** will be used to investigate which molecular features contribute to model predictions.

This will allow the project to move beyond:

> "Can the model predict selectivity?"

towards:

> "Which molecular characteristics are associated with the model's predictions of COX selectivity?"

---

## Dataset Curation

A major component of this project is experimental data curation.

ChEMBL aggregates measurements from different experimental systems and publications. Measurements are therefore not treated as automatically equivalent.

The curation process will consider:

- Target identity
- Species
- Measurement type
- Units
- Assay conditions
- Experimental system
- Duplicate measurements
- Compound structure
- Publication/source

Raw data will be preserved separately from the processed analysis dataset to maintain reproducibility.

---

## Project Structure

```text
COX_Selectivity_ML/
│
├── data/
│   ├── raw/              # Original downloaded datasets
│   └── processed/        # Curated datasets used for analysis
│
├── notebooks/            # Exploratory analysis and experiments
│
├── src/                  # Reusable Python code
│
├── models/               # Trained model files
│
├── figures/              # Generated figures
│
├── results/              # Analysis outputs
│
├── reports/              # Research reports and documentation
│
├── README.md
├── requirements.txt
└── .gitignore
```

Raw datasets, trained models and generated outputs are excluded from version control where appropriate.

---

## Environment Setup

This project uses a Python virtual environment to keep its dependencies isolated from other Python projects.

### Create the environment

```bash
python3 -m venv .venv
```

### Activate the environment

```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Main packages

- Python
- pandas
- NumPy
- scikit-learn
- RDKit
- matplotlib
- seaborn
- SHAP
- Jupyter
- openpyxl

---

## Current Status

Last updated 2026-10-29 against ChEMBL release **ChEMBL_37** (release date
2026-05-01).

### Completed

- [x] Project structure created
- [x] Git repository initialised
- [x] GitHub repository connected
- [x] Python virtual environment created
- [x] Core Python dependencies installed
- [x] RDKit installation verified
- [x] Research question defined
- [x] Initial data curation strategy defined
- [x] Endpoint decision settled and documented (exact IC50 in nM, not Ki)
- [x] Both ChEMBL target records verified against the live API
- [x] Raw COX-1 and COX-2 activity data downloaded and saved with provenance
- [x] IC50 curation implemented with a logged, reproducible filter chain
- [x] Paired COX-1/COX-2 selectivity dataset built

### Dataset as currently built

| Stage | COX-1 (CHEMBL221) | COX-2 (CHEMBL230) |
|---|---|---|
| Raw activity records downloaded | 9,458 | 15,314 |
| IC50 records | 4,488 | 8,504 |
| Exact (`=`) relation | 2,364 | 6,414 |
| Units of nM | 2,346 | 6,393 |
| Positive value present | 2,346 | 6,386 |
| No ChEMBL data-validity flag | 2,142 | 6,116 |
| Assay type B or F | 2,029 | 6,100 |
| Canonical SMILES present | 2,029 | 6,099 |
| Distinct molecules after median collapse | 1,562 | 4,379 |

**Molecules measured against both targets: 1,280.** Of these, 1,275 share at
least one source publication between their COX-1 and COX-2 measurements, 117
involve a molecule whose replicate pIC50 values span more than one log unit, and
14 share a parent compound with another paired row (salt or child forms, which
are flagged rather than merged).

The authoritative, machine-readable version of this table is
`data/processed/filter_log.csv`, regenerated on every run. The numbers above are
a snapshot and will change with future ChEMBL releases.

Observed selectivity (`log10(IC50_COX-1 / IC50_COX-2)`, positive means COX-2
selective) ranges from −3.79 to +4.44, median +1.06. As a check on the
orientation and on the curation, the coxibs all fall on the positive side
(valdecoxib +3.33, celecoxib +2.27, etoricoxib +2.21, rofecoxib +1.78) while
established COX-1-selective tool compounds fall on the negative side (SC-560
−3.79, FR-122047 −3.11, mofezolac −2.50, indomethacin −0.88). This is a
consistency check on the sign convention and the pipeline, not a validation of
the eventual model.

### Known limitations in the current dataset

- Censored measurements are excluded, so very weak inhibitors reported only as
  `> 100000 nM` are absent. This truncates the selectivity distribution and
  probably biases it, because a compound inactive at COX-1 and potent at COX-2
  is exactly the kind of pair that gets a censored COX-1 value.
- Five of the 1,280 pairs combine measurements from entirely different
  publications, and even same-publication pairs may use different assay formats
  for the two isoforms.
- Only 92 of the 1,280 paired molecules carry a ChEMBL preferred name, so the
  dataset is dominated by medicinal-chemistry series rather than marketed drugs.
  Those series are highly scaffold-correlated, which is why a scaffold-aware
  split matters more here than a random one.
- Salt and parent forms of the same compound can both appear; they are flagged
  via `parent_has_multiple_paired_forms` but not deduplicated.

### Not started

- [ ] Exploratory analysis and plots of the paired dataset
- [ ] RDKit descriptors and Morgan fingerprints
- [ ] Baseline models, then random forest / gradient boosting
- [ ] Scaffold-aware split alongside a random split
- [ ] Model evaluation (MAE, RMSE, R² with sample sizes)
- [ ] SHAP analysis

### Future Work

- [ ] Investigate alternative activity endpoints
- [ ] Compare different molecular representations
- [ ] Investigate experimental assay heterogeneity
- [ ] Explore structural interpretation of model findings
- [ ] Potentially investigate molecular docking as a complementary mechanistic analysis

---

## Reproducibility

The project is being developed with reproducibility in mind.

Key principles include:

- Preserving raw data separately from processed data
- Recording data-cleaning decisions
- Keeping the computational environment documented
- Separating exploratory analysis from reusable code
- Recording model parameters and evaluation methods
- Maintaining the project under version control using Git

### Running the pipeline

```bash
cd ~/Documents/COX_Selectivity_ML
source .venv/bin/activate
pip install -r requirements.txt
python src/chembl_download.py   # -> data/raw/      (several minutes; ChEMBL is paginated)
python src/curate.py            # -> data/processed/ (seconds)
```

Both scripts are idempotent: re-running overwrites their own outputs and nothing
else. `src/chembl_download.py` refuses to save a table whose row count disagrees
with the total ChEMBL reported, so a truncated download cannot be mistaken for a
complete one.

### Code

| File | Purpose |
|---|---|
| `src/chembl_download.py` | The only code that contacts the ChEMBL API. Verifies target records, paginates with timeout, `raise_for_status` and exponential-backoff retry, writes raw tables and `provenance.json`. |
| `src/curate.py` | Applies the documented filter chain, collapses replicates to one median value per molecule–target pair, builds the paired selectivity table and `filter_log.csv`. |
| `notebooks/01_data_exploration.ipynb` | Original interactive exploration, plus the endpoint decision and cells that read the saved outputs back from disk. |

### Outputs

| Path | Contents |
|---|---|
| `data/raw/chembl_activities_cox{1,2}_*.csv.gz` | Unmodified ChEMBL activity records, all endpoints |
| `data/raw/chembl_target_cox{1,2}_*.json` | Full target records as returned by the API |
| `data/raw/provenance.json` | ChEMBL release and date, retrieval timestamp, exact query parameters, row counts |
| `data/processed/cox{1,2}_ic50_records.csv` | Curated record-level IC50 data with assay and source metadata |
| `data/processed/cox{1,2}_molecule_summary.csv` | One row per molecule, with replicate spread retained |
| `data/processed/cox_paired_selectivity.csv` | The paired selectivity dataset |
| `data/processed/filter_log.csv` | Row and molecule count after every filter step |
| `data/processed/curation_summary.json` | Endpoint, transform, selectivity definition, duplicate policy, final counts |

`data/raw/` is git-ignored because the tables are large and regenerable; the
provenance file records everything needed to reproduce them. `data/processed/`
is tracked, so the curated dataset is versioned alongside the code that built
it.

### Environment

Recorded at the time of writing:

- Python 3.12.10
- pandas 3.0.6, NumPy 2.5.3, requests 2.34.2
- RDKit 2026.03.6, scikit-learn 1.9.1

No random seed is involved in data acquisition or curation; the pipeline is
deterministic given a ChEMBL release. Seeds will be recorded explicitly once
model training begins.

---

## Technologies

**Programming:** Python

**Cheminformatics:** RDKit

**Data analysis:** pandas, NumPy

**Machine learning:** scikit-learn

**Visualisation:** matplotlib, seaborn

**Explainable AI:** SHAP

**Development:** Jupyter, Git, GitHub

---

## Author

**Robyn Ellis-Abbott**

Biomedical Sciences graduate | Computational Biology & Machine Learning

This project was developed as an independent computational drug-discovery / cheminformatics project.