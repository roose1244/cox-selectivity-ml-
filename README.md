]# Predicting COX-1/COX-2 Selectivity Using Machine Learning

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

The primary data source will be **BindingDB**, using experimentally measured protein–small-molecule activity data.

The initial analysis will focus on:

- Human COX-1 / PTGS1
- Human COX-2 / PTGS2
- Small-molecule compounds
- Paired measurements against both targets
- Ki measurements as the primary activity endpoint

Other activity measurements, such as IC50 and Kd, will initially be treated separately because these measurements represent different experimental quantities and may not be directly interchangeable.

### 2. Selectivity

For compounds with paired Ki measurements, selectivity will initially be calculated as:

\[
SI = \frac{K_i(\mathrm{COX-1})}{K_i(\mathrm{COX-2})}
\]

The machine-learning target will use the logarithm of this ratio:

\[
\log_{10}(SI)
\]

This produces a continuous measure of relative COX-1/COX-2 inhibition.

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

BindingDB contains measurements from different experimental systems and publications. Therefore, measurements will not be treated as automatically equivalent.

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

### Completed

- [x] Project structure created
- [x] Git repository initialised
- [x] GitHub repository connected
- [x] Python virtual environment created
- [x] Core Python dependencies installed
- [x] RDKit installation verified
- [x] Research question defined
- [x] Initial data curation strategy defined

### In Progress

- [ ] Obtain COX-1/COX-2 experimental data
- [ ] Curate activity measurements
- [ ] Identify paired COX-1/COX-2 compounds
- [ ] Calculate selectivity values
- [ ] Perform exploratory data analysis
- [ ] Generate molecular descriptors
- [ ] Generate Morgan fingerprints
- [ ] Train machine-learning models
- [ ] Evaluate model performance
- [ ] Perform SHAP analysis

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