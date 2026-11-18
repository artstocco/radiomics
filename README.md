# STEREOSARC: CT radiomics pipeline

ENSICAEN industrial project (Centre François Baclesse). Context: [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

> **⚠️ All notebooks and results in this repository use a PUBLIC dataset, not the STEREOSARC patients.**
> The Centre François Baclesse data is not available yet (and will never be put in this repository).
> While waiting, the pipeline is developed and tested on the **TCIA Soft-tissue-Sarcoma** collection.

## Dataset used here: TCIA Soft-tissue-Sarcoma

| | |
|---|---|
| Collection page / download | https://www.cancerimagingarchive.net/collection/soft-tissue-sarcoma/ |
| DOI | [10.7937/K9/TCIA.2015.7GO2GSKS](https://doi.org/10.7937/K9/TCIA.2015.7GO2GSKS) |
| Patients | 51, soft-tissue sarcoma of the extremities (McGill University Health Centre) |
| Images | FDG-PET/CT and MRI before treatment; **we use the CT** of the PET/CT |
| Contours | RTSTRUCT `RTstruct_CT`: `GTV_Mass` (tumour) for all, `GTV_Edema` (tumour + edema) for 32 |
| Outcome | `LungMets`: lung metastases during follow-up (19 / 51), in [`INFOclinical_STS.xlsx`](https://www.cancerimagingarchive.net/wp-content/uploads/INFOclinical_STS.xlsx) |
| License | [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/) |

You do not need to download anything by hand: `scripts/download_sts.py` (below) fetches the CT, the contours and the
clinical table through the TCIA API. Alternatively, use the NBIA Data Retriever from the collection page.

**Differences with STEREOSARC:** one pre-treatment scan per patient (not follow-up CTs), the ROI is the primary
tumour (not non-irradiated tissue), and the label is lung metastasis. The goal here is to build and validate the
*method* (import, QC, extraction, robustness, evaluation), not to answer the STEREOSARC question.

**Citation (required by the license):**
- Vallières M, Freeman CR, Skamene SR, El Naqa I. *A radiomics model from joint FDG-PET and MRI texture features for
  the prediction of lung metastases in soft-tissue sarcomas of the extremities.* Phys Med Biol. 2015;60(14):5471-96.
  doi:10.1088/0031-9155/60/14/5471
- Vallières M, et al. *Data from Soft-tissue-Sarcoma.* The Cancer Imaging Archive, 2015.
  doi:10.7937/K9/TCIA.2015.7GO2GSKS
- Clark K, et al. *The Cancer Imaging Archive (TCIA): maintaining and operating a public information repository.*
  J Digit Imaging. 2013;26(6):1045-57. doi:10.1007/s10278-013-9622-7

## Setup
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt     # requirements.lock.txt = exact versions
```
> PyRadiomics 3.1.0 from PyPI does not build on Python ≥ 3.12, so `requirements.txt` installs it from GitHub.

## Data
```bash
.venv/bin/python scripts/download_sts.py --n 8   # 8 patients, ~1.2 GB
.venv/bin/python scripts/download_sts.py --all   # 51 patients, ~7.5 GB
```
Files go to `data/raw/STS/<PatientID>/{CT,RTSTRUCT}/`. `data/` and `outputs/` are git-ignored, so **never commit image data**.

## Run
| What | Command |
|---|---|
| 01: DICOM → features, walkthrough on one patient | `.venv/bin/jupyter lab notebooks/01_visualization_and_extraction.ipynb` |
| Extract the whole cohort (+ robustness conditions, ~30 min) | `.venv/bin/python scripts/extract_cohort.py --jobs 4` |
| 02: robustness (ICC), redundancy, lung-mets model | `.venv/bin/jupyter lab notebooks/02_robustness_and_model.ipynb` |
| Interactive viewer | `.venv/bin/streamlit run app/viewer.py` |
| Regenerate a notebook from source | `.venv/bin/python scripts/build_notebook_0X.py` |

The notebooks are committed **with their outputs**, so they can be read directly on GitHub without running anything.
Each code cell is followed by a "What the output shows" note.

> ⚠️ When the Baclesse data arrives, clear the outputs before committing (`jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`):
> images and per-patient tables in the outputs would otherwise end up in git.

## Layout
```
src/io_dicom.py          step 1: CT series + RTSTRUCT → SimpleITK image + masks
src/preprocessing.py     step 2: crop, isotropic resampling, display windows
src/quality_control.py   step 2: acquisition parameters from the DICOM headers
src/extraction.py        step 4: PyRadiomics wrapper
src/robustness.py        step 6: contour perturbations, ICC(3,1)
src/clinical.py          clinical table / LungMets labels
config/pyradiomics_params.yaml   extraction settings (bin width, resampling, HU range)
app/viewer.py            Streamlit viewer (slices, histogram, resampling, features, cohort)
notebooks/               step-by-step analysis
scripts/                 data download, notebook generation
```
