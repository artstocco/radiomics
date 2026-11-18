"""Step 6 — Robustness: controlled perturbations of the extraction and ICC per feature."""
from __future__ import annotations

import numpy as np
import pandas as pd
import SimpleITK as sitk


def perturb_mask(mask: sitk.Image, mode: str) -> sitk.Image:
    """Erode or dilate the ROI by one voxel in-plane (~1 mm here).

    Simulates a second observer drawing the contour slightly tighter or looser.
    The slice axis is left alone: one voxel there is 3+ mm, far more than
    typical inter-observer variability.
    """
    radius = (1, 1, 0)
    if mode == "erode":
        out = sitk.BinaryErode(mask, radius, sitk.sitkBall, 0, 1)
    elif mode == "dilate":
        out = sitk.BinaryDilate(mask, radius, sitk.sitkBall, 0, 1)
    else:
        raise ValueError(mode)
    out.CopyInformation(mask)
    return out


def icc_3_1(Y: np.ndarray) -> float:
    """ICC(3,1), two-way mixed, consistency, single measurement (Shrout & Fleiss).

    Y has shape (n_subjects, k_conditions). It answers: do the conditions keep the
    patients in the same order / at the same relative distance? 1 = perfectly stable,
    < 0.75 = moderate, < 0.5 = poor (Koo & Li, 2016).
    """
    Y = np.asarray(Y, float)
    Y = Y[~np.isnan(Y).any(axis=1)]
    n, k = Y.shape
    if n < 3:
        return np.nan
    grand = Y.mean()
    ss_rows = k * ((Y.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((Y.mean(axis=0) - grand) ** 2).sum()
    ss_err = ((Y - grand) ** 2).sum() - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_err = ss_err / ((n - 1) * (k - 1))
    denom = ms_rows + (k - 1) * ms_err
    return float((ms_rows - ms_err) / denom) if denom > 0 else np.nan


def icc_table(long: pd.DataFrame, conditions: list[str], features: list[str]) -> pd.Series:
    """ICC of every feature across `conditions`. `long` has one row per (PatientID, condition)."""
    out = {}
    for f in features:
        Y = long.pivot(index="PatientID", columns="condition", values=f)[conditions].values
        out[f] = icc_3_1(Y)
    return pd.Series(out, name="ICC")
