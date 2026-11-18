"""Baixa CT + RTstruct_CT da coleção TCIA Soft-tissue-Sarcoma.

Uso:
    .venv/bin/python scripts/download_sts.py --n 8        # primeiros 8 pacientes
    .venv/bin/python scripts/download_sts.py --all        # todos os 51 (~7.5 GB)

Estrutura gerada:
    data/raw/STS/<PatientID>/CT/*.dcm
    data/raw/STS/<PatientID>/RTSTRUCT/*.dcm
    data/raw/STS/series_index.csv
    data/clinical/INFOclinical_STS.xlsx   (sheet "Outcome vector" = LungMets labels)
"""
import argparse
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from tcia_utils import nbia

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "STS"
COLLECTION = "Soft-tissue-Sarcoma"
CLINICAL_URL = "https://www.cancerimagingarchive.net/wp-content/uploads/INFOclinical_STS.xlsx"
CLINICAL_OUT = ROOT / "data" / "clinical" / "INFOclinical_STS.xlsx"


def download_clinical():
    if CLINICAL_OUT.exists():
        print("  [skip] clinical table")
        return
    import urllib.request
    CLINICAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(CLINICAL_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as r:
        CLINICAL_OUT.write_bytes(r.read())
    print(f"  [ok]   clinical table -> {CLINICAL_OUT.relative_to(ROOT)}")


def list_series() -> pd.DataFrame:
    df = pd.DataFrame(nbia.getSeries(collection=COLLECTION))
    keep = (df.Modality == "CT") | (
        (df.Modality == "RTSTRUCT") & (df.SeriesDescription.str.lower() == "rtstruct_ct")
    )
    return df[keep].sort_values(["PatientID", "Modality"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="número de pacientes")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    download_clinical()
    df = list_series()
    df.to_csv(OUT / "series_index.csv", index=False)

    patients = sorted(df.PatientID.unique())
    if not args.all:
        patients = patients[: args.n]
    sel = df[df.PatientID.isin(patients)]
    print(f"{len(patients)} pacientes, {len(sel)} séries, "
          f"{sel.FileSize.sum() / 1e9:.2f} GB")

    for _, row in sel.iterrows():
        dest = OUT / row.PatientID / row.Modality
        if dest.exists() and any(dest.iterdir()):
            print(f"  [skip] {row.PatientID} {row.Modality}")
            continue
        with tempfile.TemporaryDirectory(dir=OUT) as tmp:
            nbia.downloadSeries([row.SeriesInstanceUID], input_type="list",
                                path=tmp)
            src = Path(tmp) / row.SeriesInstanceUID
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
        print(f"  [ok]   {row.PatientID} {row.Modality} "
              f"({row.ImageCount} arquivos)")


if __name__ == "__main__":
    main()
