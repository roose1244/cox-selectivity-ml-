"""Curate raw ChEMBL activity records into a paired COX-1 / COX-2 table.

Endpoint decision for this project: exact ("=") IC50 values reported in nM.
IC50 and Ki are not interchangeable (IC50 depends on substrate concentration
and assay format, Ki is an equilibrium constant), so they are never pooled.
IC50 is used because it is by far the better-populated endpoint for both COX
isoforms in ChEMBL, and because it is what the existing exploratory notebook
already examined. Ki records remain in ``data/raw/`` for a separate analysis.

Potency is expressed as ``pIC50 = 9 - log10(IC50_nM)``, so larger values mean
more potent inhibition.

Selectivity is defined as

    selectivity = log10(IC50_COX1 / IC50_COX2) = pIC50_COX2 - pIC50_COX1

Positive values therefore mean the molecule inhibits COX-2 more strongly than
COX-1 (COX-2 selective). This orientation is stated on every output column.

Every filter step appends a row to ``data/processed/filter_log.csv`` so the
final row count is traceable back to the raw download.

Run as a script:

    python src/curate.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Endpoint and selectivity configuration, kept in one place.
ENDPOINT_TYPE = "IC50"
ENDPOINT_UNITS = "nM"
EXACT_RELATION = "="
# ChEMBL assay types kept: B = binding, F = functional. Both are legitimate
# enzyme-inhibition formats; A (ADME), T (toxicity) and U (unclassified) are not
# comparable potency measurements and are dropped.
KEPT_ASSAY_TYPES = ("B", "F")
# Molecules whose replicate pIC50 values span more than this many log units are
# flagged as poorly reproducible rather than silently averaged away.
SPREAD_FLAG_LOG_UNITS = 1.0

# Columns carried through curation. Identifiers and assay metadata are kept so
# that assay heterogeneity can be audited later; they must not become model
# features.
KEEP_COLUMNS = [
    "activity_id",
    "molecule_chembl_id",
    "parent_molecule_chembl_id",
    "molecule_pref_name",
    "canonical_smiles",
    "target_chembl_id",
    "target_pref_name",
    "target_organism",
    "target_tax_id",
    "assay_chembl_id",
    "assay_description",
    "assay_type",
    "bao_format",
    "bao_label",
    "src_id",
    "document_chembl_id",
    "document_journal",
    "document_year",
    "standard_type",
    "standard_relation",
    "standard_value",
    "standard_units",
    "pchembl_value",
    "data_validity_comment",
    "potential_duplicate",
]


class FilterLog:
    """Records the row and molecule count after each curation step."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, target: str, step: str, frame: pd.DataFrame, note: str = "") -> None:
        molecules = (
            frame["molecule_chembl_id"].nunique()
            if "molecule_chembl_id" in frame.columns
            else np.nan
        )
        self.rows.append(
            {
                "target": target,
                "step": step,
                "n_records": len(frame),
                "n_molecules": molecules,
                "note": note,
            }
        )
        print(f"  [{target}] {step}: {len(frame):,} records, {molecules:,} molecules  {note}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def load_raw(label: str, target_chembl_id: str) -> pd.DataFrame:
    path = RAW_DIR / f"chembl_activities_{label}_{target_chembl_id}.csv.gz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python src/chembl_download.py` first."
        )
    return pd.read_csv(path, low_memory=False)


def curate_target(
    raw: pd.DataFrame, label: str, target_chembl_id: str, log: FilterLog
) -> pd.DataFrame:
    """Apply the documented filter chain to one target's raw records."""
    frame = raw.copy()
    log.add(label, "01_raw_download", frame, "as downloaded from ChEMBL")

    frame = frame[frame["target_chembl_id"] == target_chembl_id]
    log.add(label, "02_target_id", frame, f"target_chembl_id == {target_chembl_id}")

    frame = frame[frame["target_organism"] == "Homo sapiens"]
    log.add(label, "03_human_target", frame, "target_organism == Homo sapiens")

    frame = frame[frame["standard_type"] == ENDPOINT_TYPE]
    log.add(label, "04_endpoint_type", frame, f"standard_type == {ENDPOINT_TYPE}")

    censored = frame["standard_relation"].value_counts(dropna=False).to_dict()
    frame = frame[frame["standard_relation"] == EXACT_RELATION]
    log.add(
        label,
        "05_exact_relation",
        frame,
        f"standard_relation == '{EXACT_RELATION}'; all relations seen: {censored}",
    )

    frame = frame[frame["standard_units"] == ENDPOINT_UNITS]
    log.add(label, "06_units", frame, f"standard_units == {ENDPOINT_UNITS}")

    frame["standard_value"] = pd.to_numeric(frame["standard_value"], errors="coerce")
    frame = frame[frame["standard_value"].notna() & (frame["standard_value"] > 0)]
    log.add(label, "07_positive_value", frame, "standard_value present and > 0")

    flagged = frame["data_validity_comment"].dropna().value_counts().to_dict()
    frame = frame[frame["data_validity_comment"].isna()]
    log.add(
        label,
        "08_no_validity_flag",
        frame,
        f"dropped ChEMBL-flagged records: {flagged}" if flagged else "no flagged records",
    )

    assay_types = frame["assay_type"].value_counts(dropna=False).to_dict()
    frame = frame[frame["assay_type"].isin(KEPT_ASSAY_TYPES)]
    log.add(
        label,
        "09_assay_type",
        frame,
        f"kept assay_type in {KEPT_ASSAY_TYPES}; all types seen: {assay_types}",
    )

    frame = frame[frame["canonical_smiles"].notna() & (frame["canonical_smiles"] != "")]
    log.add(label, "10_has_smiles", frame, "canonical_smiles present")

    frame = frame[[c for c in KEEP_COLUMNS if c in frame.columns]].copy()
    frame["pIC50"] = 9.0 - np.log10(frame["standard_value"])
    frame = frame.rename(columns={"standard_value": "IC50_nM"})
    return frame


def summarise_molecules(records: pd.DataFrame, label: str, log: FilterLog) -> pd.DataFrame:
    """Collapse replicate records to one defensible value per molecule.

    The representative value is the **median** pIC50 across replicates. The
    median is preferred to the mean because replicate potencies drawn from
    different publications and assay formats are skewed and occasionally
    contain transcription-scale errors; the median resists those without
    discarding data. Nothing is averaged blindly: the spread, the number of
    distinct assays and documents, and a high-spread flag are all retained so
    that unreliable molecules can be inspected or excluded downstream.
    """
    grouped = records.groupby("molecule_chembl_id")

    summary = pd.DataFrame(
        {
            "n_records": grouped.size(),
            "n_assays": grouped["assay_chembl_id"].nunique(),
            "n_documents": grouped["document_chembl_id"].nunique(),
            "pIC50_median": grouped["pIC50"].median(),
            "pIC50_min": grouped["pIC50"].min(),
            "pIC50_max": grouped["pIC50"].max(),
            "pIC50_std": grouped["pIC50"].std(),
            "parent_molecule_chembl_id": grouped["parent_molecule_chembl_id"].agg(
                lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA
            ),
            "molecule_pref_name": grouped["molecule_pref_name"].agg(
                lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA
            ),
            "canonical_smiles": grouped["canonical_smiles"].agg(
                lambda s: s.value_counts().index[0]
            ),
            "n_distinct_smiles": grouped["canonical_smiles"].nunique(),
            "document_chembl_ids": grouped["document_chembl_id"].agg(
                lambda s: "|".join(sorted(set(s.dropna().astype(str))))
            ),
            "assay_chembl_ids": grouped["assay_chembl_id"].agg(
                lambda s: "|".join(sorted(set(s.dropna().astype(str))))
            ),
        }
    ).reset_index()

    summary["pIC50_range"] = summary["pIC50_max"] - summary["pIC50_min"]
    summary["high_spread"] = summary["pIC50_range"] > SPREAD_FLAG_LOG_UNITS
    summary["IC50_nM_median"] = 10.0 ** (9.0 - summary["pIC50_median"])
    summary["target"] = label

    n_flagged = int(summary["high_spread"].sum())
    log.add(
        label,
        "11_one_value_per_molecule",
        records,
        (
            f"collapsed to {len(summary):,} molecules by median pIC50; "
            f"{n_flagged:,} flagged with pIC50 range > {SPREAD_FLAG_LOG_UNITS} log units"
        ),
    )
    return summary


def build_pairs(
    cox1: pd.DataFrame, cox2: pd.DataFrame, log: FilterLog
) -> pd.DataFrame:
    """Join the two molecule summaries into the paired selectivity table."""
    paired = cox1.merge(cox2, on="molecule_chembl_id", suffixes=("_cox1", "_cox2"))

    paired["selectivity_log10_IC50_cox1_over_cox2"] = (
        paired["pIC50_median_cox2"] - paired["pIC50_median_cox1"]
    )

    # Comparability check: ChEMBL assays are target-specific, so the useful
    # unit of comparison is the source publication. A pair measured in the same
    # paper is far more likely to share assay conditions.
    def shared_documents(row: pd.Series) -> str:
        docs1 = set(str(row["document_chembl_ids_cox1"]).split("|"))
        docs2 = set(str(row["document_chembl_ids_cox2"]).split("|"))
        return "|".join(sorted(docs1 & docs2 - {"", "nan"}))

    paired["shared_document_chembl_ids"] = paired.apply(shared_documents, axis=1)
    paired["n_shared_documents"] = (
        paired["shared_document_chembl_ids"]
        .apply(lambda s: 0 if not s else len(s.split("|")))
    )
    paired["same_publication_pair"] = paired["n_shared_documents"] > 0

    paired["smiles_consistent"] = (
        paired["canonical_smiles_cox1"] == paired["canonical_smiles_cox2"]
    )
    paired["canonical_smiles"] = paired["canonical_smiles_cox1"]
    paired["either_high_spread"] = paired["high_spread_cox1"] | paired["high_spread_cox2"]

    # Salts and other child forms get their own ChEMBL molecule ID, so the same
    # parent compound can appear as more than one paired row (for example
    # diclofenac as the free acid and as the sodium salt). These are not
    # independent observations. They are flagged rather than merged, because
    # choosing between a parent and a salt measurement needs a look at the
    # underlying assays.
    paired["parent_molecule_chembl_id"] = paired["parent_molecule_chembl_id_cox1"]
    parent_counts = paired["parent_molecule_chembl_id"].value_counts()
    paired["parent_has_multiple_paired_forms"] = (
        paired["parent_molecule_chembl_id"].map(parent_counts).fillna(1) > 1
    )
    paired["is_salt_or_child_form"] = (
        paired["molecule_chembl_id"] != paired["parent_molecule_chembl_id"]
    )

    log.add(
        "paired",
        "12_molecules_measured_against_both",
        paired.rename(columns={"molecule_chembl_id": "molecule_chembl_id"}),
        (
            f"{int(paired['same_publication_pair'].sum()):,} pairs share at least one "
            f"source publication; "
            f"{int(paired['either_high_spread'].sum()):,} pairs involve a high-spread molecule; "
            f"{int(paired['parent_has_multiple_paired_forms'].sum()):,} pairs share a parent "
            f"compound with another paired row (salt/child forms); "
            f"{int((~paired['smiles_consistent']).sum()):,} pairs with inconsistent SMILES"
        ),
    )

    ordered = [
        "molecule_chembl_id",
        "parent_molecule_chembl_id",
        "canonical_smiles",
        "molecule_pref_name_cox1",
        "molecule_pref_name_cox2",
        "pIC50_median_cox1",
        "pIC50_median_cox2",
        "IC50_nM_median_cox1",
        "IC50_nM_median_cox2",
        "selectivity_log10_IC50_cox1_over_cox2",
        "same_publication_pair",
        "n_shared_documents",
        "shared_document_chembl_ids",
        "either_high_spread",
        "is_salt_or_child_form",
        "parent_has_multiple_paired_forms",
        "smiles_consistent",
        "n_records_cox1",
        "n_records_cox2",
        "n_assays_cox1",
        "n_assays_cox2",
        "n_documents_cox1",
        "n_documents_cox2",
        "pIC50_range_cox1",
        "pIC50_range_cox2",
        "pIC50_std_cox1",
        "pIC50_std_cox2",
        "high_spread_cox1",
        "high_spread_cox2",
        "assay_chembl_ids_cox1",
        "assay_chembl_ids_cox2",
        "document_chembl_ids_cox1",
        "document_chembl_ids_cox2",
    ]
    remaining = [c for c in paired.columns if c not in ordered]
    return paired[[c for c in ordered if c in paired.columns] + remaining]


def main() -> dict:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    provenance_path = RAW_DIR / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}

    targets = {t["label"]: t["target_chembl_id"] for t in provenance.get("targets", [])}
    cox1_id = targets.get("cox1", "CHEMBL221")
    cox2_id = targets.get("cox2", "CHEMBL230")

    log = FilterLog()

    print("Curating COX-1 records")
    cox1_records = curate_target(load_raw("cox1", cox1_id), "cox1", cox1_id, log)
    cox1_summary = summarise_molecules(cox1_records, "cox1", log)

    print("Curating COX-2 records")
    cox2_records = curate_target(load_raw("cox2", cox2_id), "cox2", cox2_id, log)
    cox2_summary = summarise_molecules(cox2_records, "cox2", log)

    print("Building paired table")
    paired = build_pairs(cox1_summary, cox2_summary, log)

    cox1_records.to_csv(PROCESSED_DIR / "cox1_ic50_records.csv", index=False)
    cox2_records.to_csv(PROCESSED_DIR / "cox2_ic50_records.csv", index=False)
    cox1_summary.to_csv(PROCESSED_DIR / "cox1_molecule_summary.csv", index=False)
    cox2_summary.to_csv(PROCESSED_DIR / "cox2_molecule_summary.csv", index=False)
    paired.to_csv(PROCESSED_DIR / "cox_paired_selectivity.csv", index=False)
    log.to_frame().to_csv(PROCESSED_DIR / "filter_log.csv", index=False)

    summary = {
        "curated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "curator": "src/curate.py",
        "chembl_db_version": provenance.get("chembl_db_version"),
        "chembl_release_date": provenance.get("chembl_release_date"),
        "raw_retrieved_at_utc": provenance.get("retrieved_at_utc"),
        "endpoint": {
            "standard_type": ENDPOINT_TYPE,
            "standard_units": ENDPOINT_UNITS,
            "standard_relation": EXACT_RELATION,
            "censored_relations": "excluded from this analysis; counts recorded in filter_log.csv",
            "rationale": (
                "IC50 and Ki are different quantities and are not pooled. IC50 in nM "
                "with an exact '=' relation is the best-populated comparable endpoint "
                "for both COX isoforms in this ChEMBL release, and matches the existing "
                "exploratory notebook."
            ),
        },
        "potency_transform": "pIC50 = 9 - log10(IC50_nM)",
        "selectivity_definition": (
            "selectivity_log10_IC50_cox1_over_cox2 = log10(IC50_COX1 / IC50_COX2) "
            "= pIC50_COX2 - pIC50_COX1; positive means stronger COX-2 inhibition"
        ),
        "duplicate_policy": (
            f"median pIC50 per molecule-target pair; spread retained and flagged when "
            f"the replicate range exceeds {SPREAD_FLAG_LOG_UNITS} log units"
        ),
        "assay_types_kept": list(KEPT_ASSAY_TYPES),
        "counts": {
            "cox1_curated_records": int(len(cox1_records)),
            "cox1_molecules": int(len(cox1_summary)),
            "cox2_curated_records": int(len(cox2_records)),
            "cox2_molecules": int(len(cox2_summary)),
            "paired_molecules": int(len(paired)),
            "paired_same_publication": int(paired["same_publication_pair"].sum()),
            "paired_either_high_spread": int(paired["either_high_spread"].sum()),
            "paired_sharing_parent_with_another_row": int(
                paired["parent_has_multiple_paired_forms"].sum()
            ),
        },
        "outputs": sorted(p.name for p in PROCESSED_DIR.glob("*.csv")),
    }
    (PROCESSED_DIR / "curation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\nPaired molecules: {len(paired):,}")
    print(f"Outputs written to {PROCESSED_DIR}")
    return summary


if __name__ == "__main__":
    main()
