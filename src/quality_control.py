"""Step 2 — Quality control: acquisition parameters read from the DICOM headers."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pydicom

from .io_dicom import DATA_DIR, list_patients

TAGS = ["Manufacturer", "ManufacturerModelName", "KVP", "XRayTubeCurrent",
        "ConvolutionKernel", "SliceThickness", "ContrastBolusAgent"]


def acquisition_table(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One row per patient with the scanner / protocol parameters (header of the first CT slice)
    plus the slice count and in-plane pixel size."""
    rows = []
    for pid in list_patients(data_dir):
        files = sorted((Path(data_dir) / pid / "CT").glob("*.dcm"))
        d = pydicom.dcmread(files[0], stop_before_pixels=True)
        row = {"PatientID": pid, "n_slices": len(files),
               "PixelSpacing_mm": float(d.PixelSpacing[0])}
        for t in TAGS:
            v = d.get(t)
            row[t] = None if v in (None, "") else (float(v) if t in ("KVP", "XRayTubeCurrent", "SliceThickness") else str(v))
        rows.append(row)
    return pd.DataFrame(rows).set_index("PatientID")
