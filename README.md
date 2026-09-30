# AEGIS-ND
**Adversarial Evaluation and Genomic Information Security for Neurodegenerative Diseases**

![Status](https://img.shields.io/badge/Status-In%20Progress-blue)
![Domain](https://img.shields.io/badge/Domain-Bioinformatics%20%7C%20Trustworthy%20AI-success)

## 📌 Executive Summary
AEGIS-ND is a research-oriented evaluation framework designed to quantify and mitigate privacy leakage in machine learning models trained on genomic and clinical data for neurodegenerative diseases (specifically Alzheimer's Disease). 

While ML holds immense promise for disease risk prediction, genomic data is uniquely sensitive. This project rigorously answers a critical question: **Do disease-risk models leak sensitive membership information, and can we defend against this leakage without destroying the model's clinical utility?**

This repository serves as an evaluation harness, deliberately framed as *evaluation-plus-intervention* research. It mirrors methodologies used in broader trustworthiness and machine unlearning research, applied specifically to the tabular genomic domain.

## 🎯 Research Objectives
1. **Baseline Modeling:** Train a baseline disease-risk classifier predicting Alzheimer's risk using genetic markers (SNPs + APOE genotype).
2. **Vulnerability Assessment:** Implement and evaluate Membership Inference Attacks (MIAs) against the baseline model to empirically measure data leakage.
3. **Mitigation Application:** Apply privacy-preserving interventions (e.g., Differential Privacy, Regularization) to the training pipeline.
4. **Trade-off Analysis:** Re-evaluate the attack on the mitigated model to quantify the exact trade-off between privacy protection gained and model utility (accuracy/AUC) lost.

## 🧬 Dataset
The study utilizes the **Alzheimer's disease with 8 SNPs and APOE** dataset. 
- **Features:** Real genetic markers (8 SNPs + APOE genotype, a highly established risk factor).
- **Scale:** Small-N regime (499 samples), representing realistic constraints in individually accessible, non-consortium genomic studies. Our shadow-modeling methodology is explicitly adapted for this low-data regime.

## 🏗️ Pipeline Architecture
The evaluation pipeline is broken down into modular, reproducible stages:
1. **Data Ingestion & Preprocessing:** Cleaning, feature encoding, and safe data splitting.
2. **Target Model Training:** Building the baseline predictive classifier.
3. **Shadow Model Training:** Training surrogate models on resampled data distributions to generate attack training data.
4. **Membership Inference:** Training an attack classifier (via IBM Adversarial Robustness Toolbox) to detect training set membership.
5. **Mitigation Application:** Injecting Differential Privacy (via diffprivlib).
6. **Re-Evaluation:** Generating the final utility-vs-privacy trade-off metrics.

## 🛠️ Technology Stack
- **Language:** Python 3.10+
- **Data & ML Baseline:** `pandas`, `numpy`, `scikit-learn`
- **Adversarial Evaluation:** `adversarial-robustness-toolbox` (IBM ART)
- **Privacy Defenses:** `diffprivlib`
- **Visualization:** `matplotlib`, `seaborn`

## 📁 Repository Structure
```text
aegis-nd/
├── data/               # Raw and preprocessed genomic data
├── notebooks/          # Exploratory Data Analysis & pipeline prototyping
├── src/                # Modular Python source code
│   ├── models/         # Target and shadow model architectures
│   ├── attacks/        # Membership inference implementations
│   ├── defenses/       # Differential privacy wrappers
│   └── evaluation/     # Metrics and visualization logic
├── results/            # Generated figures and trade-off tables
└── report/             # Final methodology and findings report
```

## 🎓 Academic Context
This project was developed by **Raja Omar** as an independent bioinformatics research endeavor. It sits alongside prior work in machine unlearning for medical imaging, collectively demonstrating a cohesive research interest in building trustworthy, privacy-preserving AI systems for healthcare.

---
*Note: This repository is actively under development.*
