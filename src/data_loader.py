"""AEGIS-ND: Data Ingestion and Clinical Harmonization Module (Phase 2)
===================================================================
Harmonizes Alzheimer's Disease Neuroimaging Initiative (ADNI) whole-genome
Illumina HumanOmniExpress microarrays across two cohorts with clinical
diagnostic and demographic metadata extracted from ADNIMERGE2 (.rda tables).

Key Functionality:
- Extracts baseline clinical diagnosis (CN, MCI, AD) from DXSUM.rda.
- Extracts patient demographics (age, sex, education) from PTDEMOG.rda.
- Extracts Apolipoprotein E (APOE ε4) risk alleles from APOERES.rda.
- Intersects Cohort 1 (432 subjects) and Cohort 2 (361 subjects) PLINK arrays.
- Implements memory-safe on-demand SNP dosage extraction via bed-reader.
- Exports harmonized metadata and manifest artifacts to data/processed/.

Author: Raja Omar
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyreadr
from bed_reader import open_bed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("AEGIS-ND.DataLoader")

# Base project paths
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

COHORT1_DIR = RAW_DIR / "ADNI_GO_2_OmniExpress"
COHORT1_BED = COHORT1_DIR / "ADNI_GO_2_Forward_Bin.bed"
COHORT1_BIM = COHORT1_DIR / "ADNI_GO_2_Forward_Bin.bim"
COHORT1_FAM = COHORT1_DIR / "ADNI_GO_2_Forward_Bin.fam"

COHORT2_DIR = RAW_DIR / "ADNI_GO2_2nd" / "Binary"
COHORT2_BED = COHORT2_DIR / "ADNI_GO2_GWAS_2nd_orig_BIN.bed"
COHORT2_BIM = COHORT2_DIR / "ADNI_GO2_GWAS_2nd_orig_BIN.bim"
COHORT2_FAM = COHORT2_DIR / "ADNI_GO2_GWAS_2nd_orig_BIN.fam"

RDA_DIR = RAW_DIR / "download" / "ADNIMERGE2" / "ADNIMERGE2" / "data"


def load_clinical_records(rda_dir: Path = RDA_DIR) -> pd.DataFrame:
    """Extract and standardize baseline diagnosis, demographics, and APOE status.

    Parameters
    ----------
    rda_dir : Path
        Directory containing ADNIMERGE2 .rda tables (DXSUM, PTDEMOG, APOERES).

    Returns
    -------
    pd.DataFrame
        DataFrame indexed by PTID with columns:
        - diagnosis: 'CN', 'MCI', or 'AD'
        - target_binary: 0 for CN, 1 for AD, NaN for MCI
        - sex: 'Male' or 'Female'
        - education: Years of education (float)
        - age: Patient age at baseline exam (int)
        - apoe_genotype: e.g., '3/3', '3/4', '4/4'
        - apoe4_carrier: 1 if carrying >= 1 ε4 allele, else 0
    """
    logger.info("Extracting clinical metadata from %s...", rda_dir)

    # 1. Diagnostic Summary (DXSUM)
    dx_file = rda_dir / "DXSUM.rda"
    if not dx_file.exists():
        raise FileNotFoundError(f"Missing clinical table: {dx_file}")

    dx_data = pyreadr.read_r(str(dx_file))["DXSUM"]

    # Filter for baseline visit ('bl'), falling back to screening ('sc')
    bl_dx = dx_data[dx_data["VISCODE2"] == "bl"][
        ["PTID", "DIAGNOSIS", "EXAMDATE"]
    ].drop_duplicates(subset=["PTID"])
    sc_dx = dx_data[dx_data["VISCODE2"] == "sc"][
        ["PTID", "DIAGNOSIS", "EXAMDATE"]
    ].drop_duplicates(subset=["PTID"])

    dx_clean = pd.concat([bl_dx, sc_dx]).drop_duplicates(
        subset=["PTID"], keep="first"
    )

    # Standardize diagnosis nomenclature: 'Dementia' in ADNI maps to Alzheimer's Disease (AD)
    dx_clean["diagnosis"] = dx_clean["DIAGNOSIS"].replace(
        {"Dementia": "AD", "TEAM_NODX": np.nan}
    )
    dx_clean = dx_clean[dx_clean["diagnosis"].isin(["CN", "MCI", "AD"])].copy()

    # Binary classification target (CN = 0, AD = 1, MCI = NaN)
    dx_clean["target_binary"] = dx_clean["diagnosis"].map(
        {"CN": 0, "AD": 1, "MCI": np.nan}
    )

    # 2. Patient Demographics (PTDEMOG)
    demog_file = rda_dir / "PTDEMOG.rda"
    demog_data = pyreadr.read_r(str(demog_file))["PTDEMOG"]
    demog_clean = demog_data.sort_values("USERDATE").drop_duplicates(
        subset=["PTID"], keep="first"
    )

    # 3. APOE Genotyping Results (APOERES)
    apoe_file = rda_dir / "APOERES.rda"
    apoe_data = pyreadr.read_r(str(apoe_file))["APOERES"]
    apoe_clean = apoe_data.drop_duplicates(subset=["PTID"], keep="first")

    # Merge clinical tables
    clinical = dx_clean.merge(
        demog_clean[["PTID", "PTGENDER", "PTDOBYY", "PTEDUCAT", "USERDATE"]],
        on="PTID",
        how="left",
    )
    clinical = clinical.merge(
        apoe_clean[["PTID", "GENOTYPE"]], on="PTID", how="left"
    )

    # Calculate baseline age: exam year minus birth year
    exam_year = pd.to_datetime(clinical["EXAMDATE"], errors="coerce").dt.year
    birth_year = pd.to_datetime(clinical["PTDOBYY"], errors="coerce").dt.year
    clinical["age"] = (exam_year - birth_year).astype("Int64")

    # Format demographic columns
    clinical["sex"] = clinical["PTGENDER"].fillna("Unknown")
    clinical["education"] = clinical["PTEDUCAT"].astype(float)
    clinical["apoe_genotype"] = clinical["GENOTYPE"].fillna("Unknown")

    # Binary indicator for APOE-ε4 carrier (strongest known genetic risk factor)
    clinical["apoe4_carrier"] = clinical["GENOTYPE"].apply(
        lambda g: 1 if ("4" in str(g)) else (0 if pd.notna(g) else 0)
    )

    cols_to_keep = [
        "PTID",
        "diagnosis",
        "target_binary",
        "age",
        "sex",
        "education",
        "apoe_genotype",
        "apoe4_carrier",
    ]
    clinical = clinical[cols_to_keep].set_index("PTID")

    logger.info(
        "Successfully harmonized clinical records: %d unique subjects "
        "(CN: %d, MCI: %d, AD: %d).",
        len(clinical),
        (clinical["diagnosis"] == "CN").sum(),
        (clinical["diagnosis"] == "MCI").sum(),
        (clinical["diagnosis"] == "AD").sum(),
    )
    return clinical


def load_plink_subject_manifest() -> pd.DataFrame:
    """Load and reconcile patient identifiers across both PLINK .fam files.

    Returns
    -------
    pd.DataFrame
        DataFrame containing PTID, cohort origin, and PLINK family row index.
    """
    logger.info("Loading PLINK subject manifests from Cohort 1 and Cohort 2...")

    fam1 = pd.read_csv(
        COHORT1_FAM,
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "pat", "mat", "sex_plink", "pheno_plink"],
    )
    fam1["cohort"] = "ADNI_GO_2_OmniExpress"
    fam1["bed_path"] = str(COHORT1_BED)
    fam1["row_index"] = np.arange(len(fam1))

    fam2 = pd.read_csv(
        COHORT2_FAM,
        sep=r"\s+",
        header=None,
        names=["fid", "iid", "pat", "mat", "sex_plink", "pheno_plink"],
    )
    fam2["cohort"] = "ADNI_GO2_2nd"
    fam2["bed_path"] = str(COHORT2_BED)
    fam2["row_index"] = np.arange(len(fam2))

    manifest = pd.concat([fam1, fam2], ignore_index=True)
    manifest["PTID"] = manifest["iid"].astype(str).str.strip()

    # Verify uniqueness
    if manifest["PTID"].duplicated().any():
        dups = manifest[manifest["PTID"].duplicated()]["PTID"].tolist()
        raise ValueError(f"Found duplicate subject IDs across cohorts: {dups}")

    logger.info(
        "Loaded %d total genotyped subjects (Cohort 1: %d, Cohort 2: %d).",
        len(manifest),
        len(fam1),
        len(fam2),
    )
    return manifest


def get_shared_snp_list() -> List[str]:
    """Identify the exact set of shared autosomal SNPs across both Illumina arrays.

    Returns
    -------
    List[str]
        List of shared SNP marker identifiers (rsIDs).
    """
    logger.info("Finding intersection of SNPs across Cohort 1 and Cohort 2 .bim files...")

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

    snps1 = set(bim1["snp"])
    # Preserve order of Cohort 2 (the standard 716,503 Illumina OmniExpress panel)
    shared_snps = [snp for snp in bim2["snp"] if snp in snps1]

    logger.info(
        "Identified %d shared SNPs (Cohort 1 has %d, Cohort 2 has %d).",
        len(shared_snps),
        len(bim1),
        len(bim2),
    )
    return shared_snps


def align_clinical_and_genomics() -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Align genotyped participants with clinical records and extract target cohorts.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, List[str]]
        - full_cohort: All 793 genotyped participants with aligned clinical labels.
        - binary_cohort: 383 participants with CN vs. AD diagnosis.
        - shared_snps: 716,503 shared marker IDs.
    """
    clinical_df = load_clinical_records()
    genomic_manifest = load_plink_subject_manifest()
    shared_snps = get_shared_snp_list()

    # Inner join on PTID
    aligned = genomic_manifest.merge(
        clinical_df, on="PTID", how="inner"
    ).set_index("PTID")

    logger.info(
        "Aligned %d genotyped subjects with complete clinical labels.",
        len(aligned),
    )

    # Binary evaluation cohort (Cognitively Normal vs. Alzheimer's Disease)
    binary_cohort = aligned[aligned["diagnosis"].isin(["CN", "AD"])].copy()
    binary_cohort["target_binary"] = binary_cohort["target_binary"].astype(int)

    logger.info(
        "Binary cohort summary: %d total subjects (CN: %d, AD: %d).",
        len(binary_cohort),
        (binary_cohort["diagnosis"] == "CN").sum(),
        (binary_cohort["diagnosis"] == "AD").sum(),
    )

    return aligned, binary_cohort, shared_snps


class ADNIArrayReader:
    """Memory-safe, high-performance reader for dual-cohort ADNI PLINK arrays."""

    def __init__(self, cohort_manifest: pd.DataFrame, shared_snps: List[str]):
        self.manifest = cohort_manifest
        self.shared_snps = shared_snps
        self._snp_to_idx = {snp: i for i, snp in enumerate(shared_snps)}

        # Map SNPs to their column index inside each individual .bim file
        logger.info("Indexing SNP column positions for Cohort 1...")
        bim1 = pd.read_csv(
            COHORT1_BIM,
            sep=r"\s+",
            header=None,
            names=["chr", "snp", "cm", "pos", "a1", "a2"],
            usecols=["snp"],
        )
        self._bim1_idx = {snp: i for i, snp in enumerate(bim1["snp"])}

        logger.info("Indexing SNP column positions for Cohort 2...")
        bim2 = pd.read_csv(
            COHORT2_BIM,
            sep=r"\s+",
            header=None,
            names=["chr", "snp", "cm", "pos", "a1", "a2"],
            usecols=["snp"],
        )
        self._bim2_idx = {snp: i for i, snp in enumerate(bim2["snp"])}

        # Open memory-mapped bed-readers
        self._bed1 = open_bed(str(COHORT1_BED))
        self._bed2 = open_bed(str(COHORT2_BED))

    def read_snp_dosages(
        self,
        snp_ids: List[str],
        subjects_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Query additive allele dosages (0, 1, 2) for specific SNPs.

        Parameters
        ----------
        snp_ids : List[str]
            Subset of SNP rsIDs to retrieve.
        subjects_df : pd.DataFrame, optional
            Subset of subjects to extract (default is all subjects in manifest).

        Returns
        -------
        pd.DataFrame
            Matrix of shape (N_subjects, len(snp_ids)) with values in {0, 1, 2, NaN}.
        """
        if subjects_df is None:
            subjects_df = self.manifest

        c1_subjects = subjects_df[subjects_df["cohort"] == "ADNI_GO_2_OmniExpress"]
        c2_subjects = subjects_df[subjects_df["cohort"] == "ADNI_GO2_2nd"]

        # Indices in bim1 and bim2
        idx1 = [self._bim1_idx[snp] for snp in snp_ids]
        idx2 = [self._bim2_idx[snp] for snp in snp_ids]

        # Read cohort 1
        if len(c1_subjects) > 0:
            c1_rows = c1_subjects["row_index"].values
            val1 = self._bed1.read(index=(c1_rows, idx1), dtype=np.float32)
            df1 = pd.DataFrame(val1, index=c1_subjects.index, columns=snp_ids)
        else:
            df1 = pd.DataFrame(columns=snp_ids)

        # Read cohort 2
        if len(c2_subjects) > 0:
            c2_rows = c2_subjects["row_index"].values
            val2 = self._bed2.read(index=(c2_rows, idx2), dtype=np.float32)
            df2 = pd.DataFrame(val2, index=c2_subjects.index, columns=snp_ids)
        else:
            df2 = pd.DataFrame(columns=snp_ids)

        frames = [df for df in (df1, df2) if len(df) > 0]
        if not frames:
            return pd.DataFrame(index=subjects_df.index, columns=snp_ids, dtype=np.float32)
        combined = pd.concat(frames).loc[subjects_df.index]
        return combined


def export_phase2_artifacts(output_dir: Path = PROCESSED_DIR) -> Dict[str, str]:
    """Execute Phase 2 harmonization and save deliverables to disk.

    Parameters
    ----------
    output_dir : Path
        Target directory for processed artifacts.

    Returns
    -------
    Dict[str, str]
        Dictionary mapping artifact names to their file paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    full_cohort, binary_cohort, shared_snps = align_clinical_and_genomics()

    # 1. Export clinical metadata
    metadata_csv = output_dir / "clinical_metadata.csv"
    full_cohort.to_csv(metadata_csv)
    logger.info("Saved complete clinical metadata to %s.", metadata_csv)

    # 2. Export binary cohort metadata
    binary_csv = output_dir / "binary_cohort_metadata.csv"
    binary_cohort.to_csv(binary_csv)
    logger.info("Saved binary classification cohort to %s.", binary_csv)

    # 3. Export cohort manifest and summary JSON
    manifest_json = output_dir / "cohort_manifest.json"
    manifest_data = {
        "total_genotyped_subjects": len(full_cohort),
        "cohort1_count": int((full_cohort["cohort"] == "ADNI_GO_2_OmniExpress").sum()),
        "cohort2_count": int((full_cohort["cohort"] == "ADNI_GO2_2nd").sum()),
        "total_shared_snps": len(shared_snps),
        "diagnostic_distribution": {
            "CN": int((full_cohort["diagnosis"] == "CN").sum()),
            "MCI": int((full_cohort["diagnosis"] == "MCI").sum()),
            "AD": int((full_cohort["diagnosis"] == "AD").sum()),
        },
        "binary_classification_task": {
            "total_subjects": len(binary_cohort),
            "control_cn_count": int((binary_cohort["diagnosis"] == "CN").sum()),
            "case_ad_count": int((binary_cohort["diagnosis"] == "AD").sum()),
            "prevalence_rate_pct": round(
                float((binary_cohort["diagnosis"] == "AD").mean() * 100), 2
            ),
        },
        "demographics_binary_cohort": {
            "mean_age": round(float(binary_cohort["age"].mean()), 2),
            "std_age": round(float(binary_cohort["age"].std()), 2),
            "female_pct": round(
                float((binary_cohort["sex"] == "Female").mean() * 100), 2
            ),
            "apoe4_carrier_pct": round(
                float((binary_cohort["apoe4_carrier"] == 1).mean() * 100), 2
            ),
        },
    }

    with open(manifest_json, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    logger.info("Saved cohort summary manifest to %s.", manifest_json)

    # 4. Export list of shared SNPs
    shared_snps_file = output_dir / "shared_snps_list.txt"
    with open(shared_snps_file, "w", encoding="utf-8") as f:
        f.write("\n".join(shared_snps))
    logger.info("Saved shared SNP list (%d SNPs) to %s.", len(shared_snps), shared_snps_file)

    return {
        "clinical_metadata": str(metadata_csv),
        "binary_cohort": str(binary_csv),
        "cohort_manifest": str(manifest_json),
        "shared_snps": str(shared_snps_file),
    }


if __name__ == "__main__":
    logger.info("Starting Phase 2 execution from CLI...")
    artifacts = export_phase2_artifacts()
    logger.info("Phase 2 deliverables successfully exported:")
    for k, v in artifacts.items():
        logger.info("  - %s: %s", k, v)
