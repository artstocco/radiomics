"""Extracts the radiomic features of every downloaded patient, under several conditions.

Output: outputs/features_cohort.csv, one row per (PatientID, ROI, condition).

Conditions (all on GTV_Mass unless stated):
    baseline      settings of config/pyradiomics_params.yaml
    erode         contour shrunk by 1 voxel in-plane   } segmentation
    dilate        contour grown by 1 voxel in-plane    } robustness
    bw15, bw40    bin width 15 / 40 HU instead of 25   } discretisation robustness
    edema         baseline settings on GTV_Edema (tumour + peritumoral edema),
                  only for the 32 patients that have this contour

Usage:
    .venv/bin/python scripts/extract_cohort.py            # 4 parallel workers
    .venv/bin/python scripts/extract_cohort.py --jobs 2   # less RAM (~1.5 GB per worker)
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.extraction import extract, get_extractor  # noqa: E402
from src.io_dicom import list_patients, load_patient  # noqa: E402
from src.robustness import perturb_mask  # noqa: E402

OUT = ROOT / "outputs" / "features_cohort.csv"


def run_patient(pid: str) -> list[dict]:
    t0 = time.time()
    ct, masks = load_patient(pid)
    base = get_extractor()
    gtv = masks["GTV_Mass"]
    jobs = {
        ("GTV_Mass", "baseline"): (gtv, base),
        ("GTV_Mass", "erode"): (perturb_mask(gtv, "erode"), base),
        ("GTV_Mass", "dilate"): (perturb_mask(gtv, "dilate"), base),
    }
    if "GTV_Edema" in masks:  # only 32 / 51 patients have an edema contour
        jobs[("GTV_Edema", "baseline")] = (masks["GTV_Edema"], base)
    for bw in (15, 40):
        ex = get_extractor()
        ex.settings["binWidth"] = bw
        jobs[("GTV_Mass", f"bw{bw}")] = (gtv, ex)

    rows = []
    for (roi, cond), (mask, ex) in jobs.items():
        rows.append({"PatientID": pid, "ROI": roi, "condition": cond, **extract(ct, mask, ex)})
    print(f"{pid} ok ({time.time() - t0:.0f} s)", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=4)
    args = ap.parse_args()

    patients = list_patients()
    print(f"{len(patients)} patients, {args.jobs} workers", flush=True)
    res = Parallel(n_jobs=args.jobs)(delayed(run_patient)(p) for p in patients)
    df = pd.DataFrame([r for rows in res for r in rows])
    OUT.parent.mkdir(exist_ok=True)
    df.to_csv(OUT, index=False)
    print(df.shape, "->", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
