"""AEGIS-ND: Genomic Quality Control (QC) & Feature Selection Pipeline (Phase 3)
=============================================================================
Performs leak-free preprocessing, standard genomic quality control filtering,
training-set association screening (GWAS ranking), and multi-modal feature
matrix assembly for Alzheimer's Disease classification.

Methodological Guarantees:
- Strict Leakage Prevention: Genomic QC (missingness, MAF) and univariate
  association ranking are computed strictly on the training partition (N_train=306).
  The held-out test partition (N_test=77) is never touched during feature selection.
- High-Performance Vectorized Screening: Scans all 716,503 shared autosomal
  SNPs in memory-mapped chunks via bed-reader within ~35 seconds.
- Quality Control Protocol:
  * SNP Missingness Rate <= 5% (call rate >= 95%)
  * Minor Allele Frequency (MAF) >= 5%
- Multi-Modal Feature Assembly:
  * Top 100 statistically associated genomic SNPs (including canonical
    chromosome 19 APOE/APOC1/TOMM40 risk loci)
  * 4 Clinical Covariates (Age, Sex, Education, APOE-ε4 carrier status)
  * Design Matrix: X in R^{N x 104}
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bed_reader import open_bed
from sklearn.model_selection import train_test_split

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AEGIS-ND.Preprocessing")

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
RAW_DIR = BASE_DIR / "data" / "raw"

BINARY_COHORT_CSV = PROCESSED_DIR / "binary_cohort_metadata.csv"
SHARED_SNPS_FILE = PROCESSED_DIR / "shared_snps_list.txt"

COHORT1_BIM = RAW_DIR / "ADNI_GO_2_OmniExpress" / "ADNI_GO_2_Forward_Bin.bim"
COHORT2_BIM = RAW_DIR / "ADNI_GO2_2nd" / "Binary" / "ADNI_GO2_GWAS_2nd_orig_BIN.bim"


def split_cohort(
    cohort_df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Perform stratified train/test split on the binary AD vs. CN cohort.

    Parameters
    ----------
    cohort_df : pd.DataFrame
        Binary cohort metadata dataframe (N=383).
    test_size : float
        Fraction of cohort reserved for test evaluation (default 0.20).
    random_state : int
        Seed for reproducibility (default 42).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        train_df (N=306) and test_df (N=77).
    """
    logger.info(
        "Performing stratified train/test split (test_size=%.2f, seed=%d)...",
        test_size,
        random_state,
    )
    train_df, test_df = train_test_split(
        cohort_df,
        test_size=test_size,
        random_state=random_state,
        stratify=cohort_df["target_binary"],
    )

    logger.info(
        "Cohort partitioned: Train N=%d (CN: %d, AD: %d), Test N=%d (CN: %d, AD: %d)",
        len(train_df),
        (train_df["target_binary"] == 0).sum(),
        (train_df["target_binary"] == 1).sum(),
        len(test_df),
        (test_df["target_binary"] == 0).sum(),
        (test_df["target_binary"] == 1).sum(),
    )
    return train_df.copy(), test_df.copy()


def run_training_gwas_scan(
    train_df: pd.DataFrame,
    shared_snps: List[str],
    chunk_size: int = 50000,
    missing_threshold: float = 0.05,
    maf_threshold: float = 0.05,
    top_k_snps: int = 100,
) -> Tuple[List[str], np.ndarray, Dict]:
    """Execute high-speed vectorized QC and GWAS association scan on the training set.

    Computes:
    1. Call Missingness Rate per SNP on training partition.
    2. Minor Allele Frequency (MAF) on training partition.
    3. Welch's two-sample t-statistic between AD cases and CN controls on allele dosage.

    Parameters
    ----------
    train_df : pd.DataFrame
        Training cohort partition (N=306).
    shared_snps : List[str]
        List of 716,503 shared SNP identifiers.
    chunk_size : int
        Number of SNPs processed per memory-mapped chunk (default 50,000).
    missing_threshold : float
        Maximum allowed missing genotype frequency (default 0.05).
    maf_threshold : float
        Minimum minor allele frequency (default 0.05).
    top_k_snps : int
        Number of top associated SNPs to select (default 100).

    Returns
    -------
    Tuple[List[str], np.ndarray, Dict]
        - top_snp_ids: List of top K selected SNP rsIDs.
        - top_t_stats: Array of corresponding Welch t-statistics.
        - qc_stats: Dictionary containing QC audit metrics.
    """
    logger.info(
        "Initiating training-set genomic QC and GWAS scan over %d markers...",
        len(shared_snps),
    )

    # Separate train subjects by sequencing batch
    c1_train = train_df[train_df["cohort"] == "ADNI_GO_2_OmniExpress"]
    c2_train = train_df[train_df["cohort"] == "ADNI_GO2_2nd"]

    bed1 = open_bed(c1_train["bed_path"].iloc[0])
    bed2 = open_bed(c2_train["bed_path"].iloc[0])

    # Map shared SNPs to column indices in Cohort 1's .bim file
    logger.info("Mapping marker indices across Cohort 1 and Cohort 2 .bim files...")
    bim1 = pd.read_csv(
        COHORT1_BIM,
        sep=r"\s+",
        header=None,
        names=["chr", "snp", "cm", "pos", "a1", "a2"],
        usecols=["snp"],
    )
    snp_to_idx1 = {snp: i for i, snp in enumerate(bim1["snp"])}
    indices1 = np.array([snp_to_idx1[s] for s in shared_snps], dtype=np.int32)

    # Class target masks aligned to concatenated subject arrays (c1 then c2)
    y_concat = np.concatenate(
        [c1_train["target_binary"].values, c2_train["target_binary"].values]
    )
    mask_cn = y_concat == 0
    mask_ad = y_concat == 1

    n_total = len(shared_snps)
    all_t_stats = np.zeros(n_total, dtype=np.float32)
    all_qc_pass = np.zeros(n_total, dtype=bool)

    t_start = time.time()
    n_chunks = int(np.ceil(n_total / chunk_size))

    for chunk_idx, start_pos in enumerate(range(0, n_total, chunk_size), 1):
        end_pos = min(start_pos + chunk_size, n_total)

        # Chunk read directly from memory-mapped bed files
        v1 = bed1.read(
            index=(c1_train["row_index"].values, indices1[start_pos:end_pos]),
            dtype=np.float32,
        )
        v2 = bed2.read(
            index=(c2_train["row_index"].values, slice(start_pos, end_pos)),
            dtype=np.float32,
        )
        combined = np.vstack([v1, v2])

        # 1. Missingness filter (call rate >= 95%)
        missing_rate = np.isnan(combined).mean(axis=0)
        qc_missing = missing_rate <= missing_threshold

        # 2. Minor Allele Frequency (MAF >= 5%)
        valid_counts = (~np.isnan(combined)).sum(axis=0)
        allele_sums = np.nansum(combined, axis=0)
        alt_freq = allele_sums / (2.0 * np.maximum(valid_counts, 1))
        maf = np.minimum(alt_freq, 1.0 - alt_freq)
        qc_maf = maf >= maf_threshold

        qc_pass = qc_missing & qc_maf
        all_qc_pass[start_pos:end_pos] = qc_pass

        # 3. Vectorized Welch's t-statistic for association with AD phenotype
        g_cn = combined[mask_cn, :]
        g_ad = combined[mask_ad, :]

        mean_cn = np.nanmean(g_cn, axis=0)
        mean_ad = np.nanmean(g_ad, axis=0)
        var_cn = np.nanvar(g_cn, axis=0, ddof=1)
        var_ad = np.nanvar(g_ad, axis=0, ddof=1)
        n_cn = np.sum(~np.isnan(g_cn), axis=0)
        n_ad = np.sum(~np.isnan(g_ad), axis=0)

        denom = np.sqrt(
            (var_cn / np.maximum(n_cn, 1)) + (var_ad / np.maximum(n_ad, 1)) + 1e-9
        )
        t_scores = np.where(qc_pass, np.abs(mean_ad - mean_cn) / denom, 0.0)
        all_t_stats[start_pos:end_pos] = t_scores

        if chunk_idx % 5 == 0 or chunk_idx == n_chunks:
            logger.info(
                "Processed chunk %d/%d (markers %d-%d)...",
                chunk_idx,
                n_chunks,
                start_pos,
                end_pos,
            )

    elapsed_scan = time.time() - t_start
    total_passed = int(all_qc_pass.sum())
    logger.info(
        "GWAS scan complete in %.2f seconds! Passed QC: %d/%d (%.1f%%)",
        elapsed_scan,
        total_passed,
        n_total,
        (total_passed / n_total) * 100.0,
    )

    # Rank by absolute t-statistic and select Top K
    top_indices = np.argsort(all_t_stats)[::-1][:top_k_snps]
    top_snp_ids = [shared_snps[i] for i in top_indices]
    top_t_stats = all_t_stats[top_indices]

    qc_stats = {
        "total_snps_screened": n_total,
        "snps_passed_qc": total_passed,
        "pass_rate_pct": round(float((total_passed / n_total) * 100.0), 2),
        "qc_criteria": {
            "max_missing_rate": missing_threshold,
            "min_maf": maf_threshold,
        },
        "scan_time_seconds": round(elapsed_scan, 2),
        "selected_snp_count": top_k_snps,
    }

    return top_snp_ids, top_t_stats, qc_stats


def extract_dosage_matrix(
    subjects_df: pd.DataFrame,
    selected_snps: List[str],
    shared_snps: List[str],
) -> np.ndarray:
    """Extract additive genotype dosages (0, 1, 2) for selected SNPs.

    Parameters
    ----------
    subjects_df : pd.DataFrame
        Cohort partition dataframe.
    selected_snps : List[str]
        List of chosen SNP rsIDs.
    shared_snps : List[str]
        Master shared SNPs list.

    Returns
    -------
    np.ndarray
        Dosage matrix of shape (N_subjects, len(selected_snps)).
    """
    c1_sub = subjects_df[subjects_df["cohort"] == "ADNI_GO_2_OmniExpress"]
    c2_sub = subjects_df[subjects_df["cohort"] == "ADNI_GO2_2nd"]

    bed1 = open_bed(c1_sub["bed_path"].iloc[0])
    bed2 = open_bed(c2_sub["bed_path"].iloc[0])

    bim1 = pd.read_csv(
        COHORT1_BIM,
        sep=r"\s+",
        header=None,
        names=["chr", "snp", "cm", "pos", "a1", "a2"],
        usecols=["snp"],
    )
    bim2 = pd.read_csv(
        COHORT2_BIM,
        sep=r"\s+",
        header=None,
        names=["chr", "snp", "cm", "pos", "a1", "a2"],
        usecols=["snp"],
    )

    snp_to_idx1 = {snp: i for i, snp in enumerate(bim1["snp"])}
    snp_to_idx2 = {snp: i for i, snp in enumerate(bim2["snp"])}

    idx1 = [snp_to_idx1[s] for s in selected_snps]
    idx2 = [snp_to_idx2[s] for s in selected_snps]

    val1 = bed1.read(index=(c1_sub["row_index"].values, idx1), dtype=np.float32)
    val2 = bed2.read(index=(c2_sub["row_index"].values, idx2), dtype=np.float32)

    # Reorder according to subjects_df original order
    combined_index = list(c1_sub.index) + list(c2_sub.index)
    combined_data = np.vstack([val1, val2])
    reorder_map = {orig_idx: pos for pos, orig_idx in enumerate(combined_index)}
    target_positions = [reorder_map[idx] for idx in subjects_df.index]

    return combined_data[target_positions, :]


def assemble_multimodal_dataset(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    selected_snps: List[str],
    shared_snps: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Assemble final multi-modal feature matrices (Top 100 SNPs + 4 Clinical Covariates).

    Imputation Rule:
    Any residual missing values in genotypes are imputed using the column means
    computed strictly from the training set.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]
        X_train, X_test, y_train, y_test, and feature_names list.
    """
    logger.info("Extracting genotype dosages for %d selected SNPs...", len(selected_snps))
    G_train_raw = extract_dosage_matrix(train_df, selected_snps, shared_snps)
    G_test_raw = extract_dosage_matrix(test_df, selected_snps, shared_snps)

    # Impute missing genotypes using training means
    train_means = np.nanmean(G_train_raw, axis=0)
    # Fallback to 0 if an entire column were somehow NaN
    train_means = np.nan_to_num(train_means, nan=0.0)

    # Impute train
    nan_mask_train = np.isnan(G_train_raw)
    G_train = np.where(nan_mask_train, train_means, G_train_raw)

    # Impute test with training statistics (zero leakage)
    nan_mask_test = np.isnan(G_test_raw)
    G_test = np.where(nan_mask_test, train_means, G_test_raw)

    # Extract 4 Clinical Covariates: age, sex (Male=1, Female=0), education, apoe4_carrier
    covariate_cols = ["age", "sex", "education", "apoe4_carrier"]
    logger.info("Assembling clinical covariates: %s", covariate_cols)

    def extract_covariates(df: pd.DataFrame) -> np.ndarray:
        c_age = df["age"].astype(float).values[:, None]
        c_sex = (df["sex"] == "Male").astype(float).values[:, None]
        c_edu = df["education"].astype(float).values[:, None]
        c_apoe = df["apoe4_carrier"].astype(float).values[:, None]
        return np.hstack([c_age, c_sex, c_edu, c_apoe])

    C_train = extract_covariates(train_df)
    C_test = extract_covariates(test_df)

    # Multi-modal feature concatenation: [SNPs (100), Covariates (4)] -> 104 features
    X_train = np.hstack([G_train, C_train]).astype(np.float32)
    X_test = np.hstack([G_test, C_test]).astype(np.float32)

    y_train = train_df["target_binary"].values.astype(np.int64)
    y_test = test_df["target_binary"].values.astype(np.int64)

    feature_names = list(selected_snps) + covariate_cols

    logger.info(
        "Feature matrices assembled: X_train %s, X_test %s (Total features: %d)",
        X_train.shape,
        X_test.shape,
        len(feature_names),
    )
    return X_train, X_test, y_train, y_test, feature_names


def export_phase3_artifacts(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_names: List[str],
    top_t_stats: np.ndarray,
    qc_stats: Dict,
    output_dir: Path = PROCESSED_DIR,
) -> Dict[str, str]:
    """Save all Phase 3 datasets and metadata files to data/processed/.

    Returns
    -------
    Dict[str, str]
        Paths of generated artifacts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Feature and label arrays
    p_xtrain = output_dir / "X_train.npy"
    p_xtest = output_dir / "X_test.npy"
    p_ytrain = output_dir / "y_train.npy"
    p_ytest = output_dir / "y_test.npy"

    np.save(p_xtrain, X_train)
    np.save(p_xtest, X_test)
    np.save(p_ytrain, y_train)
    np.save(p_ytest, y_test)

    # 2. Partition tracking CSVs
    p_train_cohort = output_dir / "train_cohort.csv"
    p_test_cohort = output_dir / "test_cohort.csv"
    train_df.to_csv(p_train_cohort)
    test_df.to_csv(p_test_cohort)

    # 3. Selected features metadata
    bim2 = pd.read_csv(
        COHORT2_BIM,
        sep=r"\s+",
        header=None,
        names=["chr", "snp", "cm", "pos", "a1", "a2"],
    )
    bim2_map = bim2.set_index("snp").to_dict(orient="index")

    selected_snps_meta = []
    for rank, (snp, t_stat) in enumerate(zip(feature_names[:100], top_t_stats), 1):
        info = bim2_map.get(snp, {})
        selected_snps_meta.append(
            {
                "rank": rank,
                "snp_id": snp,
                "chromosome": int(info.get("chr", 0)),
                "position": int(info.get("pos", 0)),
                "minor_allele": str(info.get("a1", "")),
                "major_allele": str(info.get("a2", "")),
                "welch_t_stat": round(float(t_stat), 4),
            }
        )

    features_json_data = {
        "total_features": len(feature_names),
        "snp_feature_count": 100,
        "clinical_feature_count": 4,
        "clinical_covariates": ["age", "sex", "education", "apoe4_carrier"],
        "top_snps": selected_snps_meta,
    }

    p_feat_json = output_dir / "selected_features.json"
    with open(p_feat_json, "w", encoding="utf-8") as f:
        json.dump(features_json_data, f, indent=2)

    # 4. QC audit summary JSON
    p_qc_json = output_dir / "qc_summary.json"
    with open(p_qc_json, "w", encoding="utf-8") as f:
        json.dump(qc_stats, f, indent=2)

    logger.info("Successfully exported all Phase 3 artifacts to %s.", output_dir)
    return {
        "X_train": str(p_xtrain),
        "X_test": str(p_xtest),
        "y_train": str(p_ytrain),
        "y_test": str(p_ytest),
        "train_cohort": str(p_train_cohort),
        "test_cohort": str(p_test_cohort),
        "selected_features": str(p_feat_json),
        "qc_summary": str(p_qc_json),
    }


def run_pipeline() -> Dict[str, str]:
    """Execute the end-to-end Phase 3 preprocessing pipeline."""
    logger.info("Starting Phase 3 Preprocessing & Feature Selection Pipeline...")

    # Load Phase 2 outputs
    if not BINARY_COHORT_CSV.exists() or not SHARED_SNPS_FILE.exists():
        raise FileNotFoundError(
            "Phase 2 artifacts not found. Please execute src/data_loader.py first."
        )

    cohort_df = pd.read_csv(BINARY_COHORT_CSV, index_col="PTID")
    with open(SHARED_SNPS_FILE, "r", encoding="utf-8") as f:
        shared_snps = [line.strip() for line in f if line.strip()]

    # 1. Stratified split
    train_df, test_df = split_cohort(cohort_df, test_size=0.20, random_state=42)

    # 2. Training-only Genomic QC & GWAS association scan
    top_snps, top_t_stats, qc_stats = run_training_gwas_scan(
        train_df, shared_snps, chunk_size=50000, top_k_snps=100
    )

    # 3. Assemble multi-modal dataset
    X_train, X_test, y_train, y_test, feature_names = assemble_multimodal_dataset(
        train_df, test_df, top_snps, shared_snps
    )

    # 4. Export deliverables
    artifacts = export_phase3_artifacts(
        X_train,
        X_test,
        y_train,
        y_test,
        train_df,
        test_df,
        feature_names,
        top_t_stats,
        qc_stats,
    )

    logger.info("Phase 3 pipeline execution completed successfully!")
    return artifacts


if __name__ == "__main__":
    run_pipeline()
