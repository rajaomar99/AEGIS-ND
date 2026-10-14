# AEGIS-ND: Adversarial Evaluation & Genomic Information Security for Neurodegenerative Diseases
**Benchmarking Black-Box Membership Inference Leakage & Differential Privacy Defenses on Alzheimer's Disease Whole-Genome Microarrays**

[![Status](https://img.shields.io/badge/Status-Active%20Research-blue.svg)](https://github.com)
[![Domain](https://img.shields.io/badge/Domain-Trustworthy%20AI%20%7C%20Privacy%20%7C%20Bioinformatics-success.svg)](https://github.com)
[![Dataset](https://img.shields.io/badge/Dataset-ADNI%20(Illumina%20OmniExpress)-orange.svg)](https://adni.loni.usc.edu/)
[![Defense](https://img.shields.io/badge/Defense-Differential%20Privacy%20(%CE%B5)-purple.svg)](https://github.com/IBM/differential-privacy-library)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

---

## 📌 Executive Summary

Machine learning models trained on biomedical and genomic features offer significant potential for disease risk prediction and clinical decision support. However, genetic data represents **uniquely identifying, permanent biometric information**: unlike credentials or tokens, an individual's genome cannot be revoked or re-issued.

In medical genomics, models routinely operate in the high-dimensional, sample-constrained regime ($P \gg N$, with $P \sim 10^5 \text{--} 10^6$ genetic variants and $N \sim 10^2 \text{--} 10^3$ patients). In this regime, empirical risk minimization easily over-memorizes sample-specific features, creating a severe vulnerability to **Membership Inference Attacks (MIAs)**. If a target model predicts risk for a sensitive condition like Alzheimer's Disease (AD), inferring that an individual was part of the training set immediately leaks their clinical status.

**AEGIS-ND** is a rigorous evaluation-and-intervention framework designed to:
1. **Benchmark Baseline Leakage:** Train diagnostic classifiers on real Alzheimer's Disease Neuroimaging Initiative (ADNI) cohorts and quantify empirical membership leakage using shadow-model architectures.
2. **Implement Formal Privacy Interventions:** Embed $(\varepsilon, \delta)$-Differential Privacy (DP) mechanisms directly into the model training pipeline.
3. **Map the Privacy-Utility Pareto Frontier:** Empirically quantify the trade-off between privacy preservation (attacker ROC-AUC / recall) and clinical utility (diagnostic accuracy / AUC).

---

## 🎯 Threat Model & Problem Formulation

```
+---------------------------------------------------------------------------------+
|                                THREAT MODEL                                     |
+---------------------------------------------------------------------------------+
|                                                                                 |
|   Target Sample: x* = [Genetic Markers (SNPs), Demographics, APOE status]       |
|                                                                                 |
|          +---------------------------------------------------------+            |
|          |                  Black-Box Target Model                 |            |
|          |                 f_target(x) -> y_pred, y_conf           |            |
|          +---------------------------------------------------------+            |
|                                         |                                       |
|                                 Posterior Vector                                |
|                                 P(y = AD | x*)                                  |
|                                         v                                       |
|          +---------------------------------------------------------+            |
|          |                   Shadow Model Ensemble                 |            |
|          |      Simulates target behavior on D_in and D_out        |            |
|          +---------------------------------------------------------+            |
|                                         |                                       |
|                               Attack Training Set                               |
|                         (Confidence Vectors, In/Out)                            |
|                                         v                                       |
|          +---------------------------------------------------------+            |
|          |                   MIA Meta-Classifier                   |            |
|          |    h_attack(P(y|x*)) -> {1: Member, 0: Non-Member}      |            |
|          +---------------------------------------------------------+            |
+---------------------------------------------------------------------------------+
```

* **Adversary Goal:** Determine whether a target individual $x^*$ was part of the private training set $D_{\text{train}}$ of the target classifier $f_{\text{target}}$.
* **Adversary Capabilities:** Strict **black-box access**. The adversary queries $f_{\text{target}}(x^*)$ to observe predicted class probabilities, without access to model weights, loss gradients, or internal representations.
* **Attack Mechanism:** Shadow model training (Shokri et al., 2017) calibrated on resampled covariate distributions, capturing the differential confidence signatures of member vs. non-member samples.

---

## 🔬 Research Pipeline

| Stage | Objective | Key Methods |
| :--- | :--- | :--- |
| **01. Ingestion & Alignment** | Harmonize genomic arrays with clinical databases | Extract baseline diagnoses (`DXSUM`), demographics (`PTDEMOG`), and *APOE* status (`APOERES`) across 793 genotyped participants. |
| **02. Quality Control & Selection** | Filter noise and address $P \gg N$ dimensionality | Strict training-split QC (missingness $< 5\%$, $MAF \ge 5\%$, HWE $p \ge 10^{-6}$), followed by univariate association ranking to select top informative markers. |
| **03. Target Modeling** | Establish predictive disease-risk classifiers | Train baseline models (Random Forest, XGBoost, Logistic Regression) on binary AD vs. CN classification; quantify generalization gap. |
| **04. Adversarial Auditing** | Measure baseline membership leakage | Construct shadow ensembles to generate attack training data and train a black-box MIA meta-classifier. |
| **05. Differential Privacy Defense** | Mitigate membership leakage | Inject calibrated Differential Privacy noise ($\varepsilon$-DP classifiers via `diffprivlib`) across privacy budgets $\varepsilon \in [0.1, 10.0]$. |
| **06. Trade-off Analysis** | Evaluate the Pareto frontier | Re-evaluate attacks on private models to measure privacy gained vs. accuracy/AUC lost. |

---

## 📊 Dataset & Cohort Overview

Experiments utilize whole-genome microarray data and clinical metadata from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)**.

* **Genomic Arrays:** Illumina HumanOmniExpress BeadChip (716,503 shared autosomal SNPs across 793 genotyped participants).
* **Target Cohort:** Binary diagnostic classification comparing **Cognitively Normal (CN, $N=256$)** vs. **Alzheimer's Disease (AD, $N=127$)**, for an evaluation cohort of **$N = 383$ subjects**.
* **Clinical Covariates:** Patient age, biological sex, educational attainment, and *APOE* $\varepsilon4$ allele carrier status (the primary genetic risk factor for late-onset AD).

---

## 🛡️ Differential Privacy Formulation

A randomized algorithm $\mathcal{M}$ satisfies $(\varepsilon, \delta)$-Differential Privacy if for all neighboring datasets $D \sim D'$ differing in a single patient record, and all output subsets $\mathcal{S} \subseteq \text{Range}(\mathcal{M})$:

$$\mathbb{P}[\mathcal{M}(D) \in \mathcal{S}] \le e^\varepsilon \mathbb{P}[\mathcal{M}(D') \in \mathcal{S}] + \delta$$

* **$\varepsilon$ (Privacy Budget):** Controls the theoretical maximum privacy leakage bound (smaller $\varepsilon \implies$ stronger mathematical guarantee).
* **$\delta$:** Bounds the probability of catastrophic privacy failure ($\delta < \frac{1}{N}$).
* **Defense Implementation:** We evaluate DP decision trees and logistic regression via objective/output perturbation, systematically sweeping $\varepsilon \in [0.1, 10.0]$ to evaluate both empirical attack suppression and clinical utility retention.

---

## 🏗️ Repository Architecture

```text
AEGIS-ND/
├── data/
│   ├── raw/                           # ADNI PLINK binaries (.bed/.bim/.fam) & .rda tables (Git-ignored)
│   └── processed/                     # QC-filtered dosage matrices & aligned labels (Git-ignored)
├── guides/
│   └── AEGIS-ND_Roadmap.md            # Comprehensive theoretical and implementation roadmap
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory data analysis, distributions, PCA
│   ├── 02_baseline_model.ipynb        # Baseline classifier training and evaluation
│   ├── 03_membership_inference.ipynb  # Shadow model generation and MIA evaluation
│   └── 04_mitigation.ipynb            # DP defenses, noise calibration, and Pareto curves
├── src/
│   ├── data_loader.py                 # Memory-mapped PLINK reader and clinical merger
│   ├── preprocessing.py               # Genomic QC pipeline and feature selection
│   ├── models/
│   │   ├── target_model.py            # Baseline classifiers and shadow training routines
│   │   └── shadow_model.py            # Shadow model orchestration
│   ├── attacks/
│   │   └── membership_inference.py    # Black-box shadow MIA meta-classifier
│   ├── defenses/
│   │   └── differential_privacy.py    # DP-ERM wrappers and noise mechanisms
│   └── evaluation/
│       ├── metrics.py                 # Utility (AUC, F1) and privacy (MIA AUC, leakage) metrics
│       └── visualize.py               # Trade-off plots and ROC curves
├── report/
│   └── final_report.md                # Research report and findings
├── LICENSE                            # MIT License
└── requirements.txt                   # Pinned reproducible dependencies
```

---

## 🛠️ Technology Stack

* **Data Engineering & Genomics:** [`bed-reader`](https://github.com/fastdatascience/bed-reader) (C/Rust memory-mapped PLINK reader), [`pyreadr`](https://github.com/ofajardo/pyreadr) (R data ingestion), `numpy`, `pandas`.
* **Predictive Modeling:** `scikit-learn` (v1.4.2), `xgboost`, `scipy`.
* **Adversarial Evaluation:** [`adversarial-robustness-toolbox`](https://github.com/Trusted-AI/adversarial-robustness-toolbox) (IBM ART).
* **Privacy Defenses:** [`diffprivlib`](https://github.com/IBM/differential-privacy-library) (IBM Differential Privacy Library).
* **Visualization:** `matplotlib`, `seaborn`.

---

## 🚀 Getting Started

### 1. Environment Setup
```bash
git clone https://github.com/rajaomar99/AEGIS-ND.git
cd AEGIS-ND

python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Data Access Notice
Raw genotype array files and clinical tables originate from the **Alzheimer's Disease Neuroimaging Initiative (ADNI)** and require an approved Data Use Agreement (DUA). Authorized researchers can acquire access via [ADNI LONI](https://adni.loni.usc.edu/). Placed files are maintained in `data/raw/` (ignored by version control).

### 3. Running the Pipeline
```bash
# Verify environment and library bindings
python -c "import bed_reader, pyreadr, sklearn, diffprivlib, art, xgboost; print('Environment Verified!')"

# Run data ingestion and clinical harmonization
python src/data_loader.py

# Run genomic quality control and feature selection
python src/preprocessing.py
```

---

## 📚 Key References

1. **Chen, J., Wang, W. H., & Shi, X.** (2021). *Differential Privacy Protection Against Membership Inference Attack on Machine Learning for Genomic Data.* **Pacific Symposium on Biocomputing (PSB)**, 26, 26–37.
2. **Shokri, R., Stronati, M., Song, C., & Shmatikov, V.** (2017). *Membership Inference Attacks Against Machine Learning Models.* **IEEE Symposium on Security and Privacy (S&P)**, 3–18.
3. **Carlini, N., et al.** (2022). *Membership Inference Attacks From First Principles.* **IEEE Symposium on Security and Privacy (S&P)**, 1897–1914.
4. **Homer, N., et al.** (2008). *Resolving Individuals Contributing Traceable Amounts of DNA to Complex Mixtures Using High-Density SNP Genotyping Microarrays.* **PLOS Genetics**, 4(8), e1000167.
5. **Dwork, C.** (2006). *Differential Privacy.* **Automata, Languages and Programming (ICALP)**, Springer, 1–12.
6. **Abadi, M., et al.** (2016). *Deep Learning with Differential Privacy.* **ACM CCS**, 308–318.
7. **Mueller, S. G., et al.** (2005). *The Alzheimer's Disease Neuroimaging Initiative.* **Neuroimaging Clinics of North America**, 15(4), 869–877.

---

## 🎓 Academic Context & Author Statement

This project was developed by **Raja Omar** as an independent research investigation exploring the intersection of **Trustworthy AI, Adversarial Machine Learning, and Healthcare Informatics**.

It complements prior work in **machine unlearning for medical imaging (MedForget)**, forming a cohesive research narrative focused on evaluating and enhancing the trustworthiness of AI across clinical modalities:
* **Medical Imaging:** Data erasure, selective forgetting, and influence mitigation in diagnostic computer vision.
* **Genomic Informatics (AEGIS-ND):** Quantifying memorization, empirical membership inference leakage, and formal Differential Privacy guarantees on high-dimensional clinical arrays.

---

## 📜 Acknowledgments & Data Notice

Data collection and sharing for this project was funded by the Alzheimer's Disease Neuroimaging Initiative (ADNI) (National Institutes of Health Grant U01 AG024904) and DOD ADNI (Department of Defense award number W81XWH-12-2-0012).
