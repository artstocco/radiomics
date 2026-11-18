"""Step 1 — Import: read a CT series and its RTSTRUCT contours as SimpleITK images.

Expected layout (see scripts/download_sts.py):
    data/raw/STS/<PatientID>/CT/*.dcm
    data/raw/STS/<PatientID>/RTSTRUCT/*.dcm
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import SimpleITK as sitk

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "raw" / "STS"

# Harmonise ROI names across patients. STS_046 uses different names for the same
# structures (checked: in every patient the edema ROI fully contains the tumour ROI,
# i.e. it is "tumour + surrounding edema").
ROI_ALIASES = {
    "GTV_Research": "GTV_Mass",
    "GTV_Res+edema": "GTV_Edema",
}


def list_patients(data_dir: Path = DATA_DIR) -> list[str]:
    return sorted(p.name for p in data_dir.glob("STS_*") if (p / "CT").is_dir())


def load_ct(patient_dir: Path) -> sitk.Image:
    """Read the CT series. GDCM sorts slices by ascending position along the slice normal."""
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(str(Path(patient_dir) / "CT"))
    if not files:
        raise FileNotFoundError(f"No CT series in {patient_dir}")
    reader.SetFileNames(files)
    return reader.Execute()


def load_masks(patient_dir: Path, ct: sitk.Image | None = None) -> dict[str, sitk.Image]:
    """Rasterise every ROI of the RTSTRUCT onto the CT grid.

    rt-utils returns arrays as (rows, cols, slices) with slices in ascending
    position, the same order as GDCM, so a transpose to (slices, rows, cols)
    puts them on the SimpleITK grid.
    """
    from rt_utils import RTStructBuilder

    patient_dir = Path(patient_dir)
    ct = ct if ct is not None else load_ct(patient_dir)
    rt_file = next((patient_dir / "RTSTRUCT").glob("*.dcm"))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rt = RTStructBuilder.create_from(
            dicom_series_path=str(patient_dir / "CT"), rt_struct_path=str(rt_file)
        )
        masks = {}
        for name in rt.get_roi_names():
            arr = np.transpose(rt.get_roi_mask_by_name(name), (2, 0, 1)).astype(np.uint8)
            if arr.shape != sitk.GetArrayViewFromImage(ct).shape:
                raise ValueError(f"{name}: mask {arr.shape} does not match CT grid")
            m = sitk.GetImageFromArray(arr)
            m.CopyInformation(ct)
            masks[ROI_ALIASES.get(name, name)] = m
    return masks


def load_patient(patient_id: str, data_dir: Path = DATA_DIR):
    """Return (ct, masks) for one patient."""
    pdir = Path(data_dir) / patient_id
    ct = load_ct(pdir)
    return ct, load_masks(pdir, ct)
