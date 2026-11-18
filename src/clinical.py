"""Clinical table of the TCIA Soft-tissue-Sarcoma collection (labels for the analysis)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CLINICAL_XLSX = ROOT / "data" / "clinical" / "INFOclinical_STS.xlsx"


def load_clinical(path: Path = CLINICAL_XLSX) -> pd.DataFrame:
    """One row per patient: LungMets label (0/1) + a few clinical variables.

    `LungMets` comes from the 'Outcome vector' sheet (the label used by Vallières et al., 2015).
    `AnyDistantMets` is derived from the free-text outcome column (lungs, bones, arms, spine, abdomen).
    """
    info = pd.read_excel(path, sheet_name="Clinical Information")
    info = info[info["Patient ID"].astype(str).str.startswith("STS_")]
    lab = pd.read_excel(path, sheet_name="Outcome vector")
    df = info.merge(lab, on="Patient ID").rename(columns={
        "Patient ID": "PatientID", "Histological type": "Histology", "Site of primary STS": "Site",
        "Outcome (recurrence, mets)": "Outcome", "Status (NED, AWD, D)": "Status"})
    df["AnyDistantMets"] = df["Outcome"].astype(str).str.startswith("Mets").astype(int)
    keep = ["PatientID", "Age", "Sex", "Histology", "MSKCC type", "Site", "Grade",
            "Treatment", "Outcome", "Status", "LungMets", "AnyDistantMets"]
    return df[keep].set_index("PatientID")
