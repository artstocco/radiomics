"""Step 4 — Feature extraction with PyRadiomics."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PARAMS = ROOT / "config" / "pyradiomics_params.yaml"

logging.getLogger("radiomics").setLevel(logging.ERROR)


def get_extractor(params: Path = DEFAULT_PARAMS):
    return featureextractor.RadiomicsFeatureExtractor(str(params))


def extract(image: sitk.Image, mask: sitk.Image, extractor=None) -> dict:
    """Return the features as a flat dict, without the 'diagnostics_*' entries."""
    extractor = extractor or get_extractor()
    res = extractor.execute(image, mask)
    return {k: float(np.real(v)) for k, v in res.items() if not k.startswith("diagnostics_")}


def features_to_frame(feats: dict) -> pd.DataFrame:
    """Split names like 'original_glcm_Contrast' into class / feature columns."""
    rows = []
    for k, v in feats.items():
        _, cls, name = k.split("_", 2)
        rows.append({"class": cls, "feature": name, "value": v})
    return pd.DataFrame(rows)
