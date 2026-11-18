"""Step 2 — Standardisation: cropping, isotropic resampling, intensity windowing."""
from __future__ import annotations

import numpy as np
import SimpleITK as sitk

# Common CT display windows: (center, width) in HU
WINDOWS = {
    "soft tissue": (40, 400),
    "lung": (-600, 1500),
    "bone": (400, 1800),
    "full range": (500, 3000),
}


def window(arr: np.ndarray, center: float, width: float) -> np.ndarray:
    """Map HU values to [0, 1] for display. Display only — never feed this to radiomics."""
    lo, hi = center - width / 2, center + width / 2
    return np.clip((arr - lo) / (hi - lo), 0, 1)


def mask_bbox(mask: sitk.Image, pad_mm: float = 10.0):
    """Bounding box (index, size) of the mask plus a margin in mm, clipped to the image."""
    f = sitk.LabelShapeStatisticsImageFilter()
    f.Execute(mask)
    x, y, z, sx, sy, sz = f.GetBoundingBox(1)
    pad = [int(np.ceil(pad_mm / s)) for s in mask.GetSpacing()]
    lo = [max(0, v - p) for v, p in zip((x, y, z), pad)]
    hi = [min(n, v + s + p) for v, s, p, n in zip((x, y, z), (sx, sy, sz), pad, mask.GetSize())]
    return lo, [h - l for h, l in zip(hi, lo)]


def crop_to_mask(image: sitk.Image, mask: sitk.Image, pad_mm: float = 10.0):
    lo, size = mask_bbox(mask, pad_mm)
    return (sitk.RegionOfInterest(image, size, lo), sitk.RegionOfInterest(mask, size, lo))


def resample(image: sitk.Image, spacing=(1.0, 1.0, 1.0), is_mask: bool = False) -> sitk.Image:
    """Resample to a new voxel spacing, keeping the same physical extent.

    Images use B-spline interpolation; masks use nearest neighbour so they stay binary.
    """
    old_sp, old_sz = np.array(image.GetSpacing()), np.array(image.GetSize())
    new_sz = np.maximum(1, np.round(old_sz * old_sp / np.array(spacing))).astype(int)
    return sitk.Resample(
        image,
        [int(s) for s in new_sz],
        sitk.Transform(),
        sitk.sitkNearestNeighbor if is_mask else sitk.sitkBSpline,
        image.GetOrigin(),
        tuple(float(s) for s in spacing),
        image.GetDirection(),
        0 if is_mask else -1024,  # outside the field of view: air
        image.GetPixelID(),
    )
