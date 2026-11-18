"""Interactive CT + RTSTRUCT viewer and radiomics explorer.

Run from the project root:
    .venv/bin/streamlit run app/viewer.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import SimpleITK as sitk
import streamlit as st
from PIL import Image

from src.extraction import extract, features_to_frame, get_extractor
from src.io_dicom import list_patients, load_patient
from src.preprocessing import WINDOWS, crop_to_mask, resample, window

st.set_page_config(page_title="STEREOSARC viewer", layout="wide")

ROI_COLORS = {"GTV_Mass": (255, 60, 60), "GTV_Edema": (255, 210, 0)}
FALLBACK_COLORS = [(0, 200, 255), (0, 255, 120), (255, 0, 255)]
DISPLAY_PX = 420  # width of each rendered view


# ---------------------------------------------------------------- data (cached)
@st.cache_resource(max_entries=2, show_spinner="Loading DICOM + RTSTRUCT…")
def get_patient(pid: str):
    ct, masks = load_patient(pid)
    return {
        "ct": ct,
        "masks": masks,
        "ct_arr": sitk.GetArrayFromImage(ct),  # (z, y, x)
        "m_arr": {k: sitk.GetArrayFromImage(v).astype(bool) for k, v in masks.items()},
        "spacing": ct.GetSpacing(),  # (x, y, z)
    }


@st.cache_resource
def get_cached_extractor():
    return get_extractor()


@st.cache_data(show_spinner="Running PyRadiomics…")
def get_features(pid: str, roi: str) -> pd.DataFrame:
    p = get_patient(pid)
    return features_to_frame(extract(p["ct"], p["masks"][roi], get_cached_extractor()))


@st.cache_data(show_spinner="Cropping + resampling…")
def get_resampled(pid: str, roi: str, iso_mm: float):
    p = get_patient(pid)
    ci, cm = crop_to_mask(p["ct"], p["masks"][roi], pad_mm=15)
    ri, rm = resample(ci, (iso_mm,) * 3), resample(cm, (iso_mm,) * 3, is_mask=True)
    a = lambda im: sitk.GetArrayFromImage(im)
    return a(ci), a(cm).astype(bool), ci.GetSpacing(), a(ri), a(rm).astype(bool), ri.GetSpacing()


# ---------------------------------------------------------------- rendering
def outline(m: np.ndarray) -> np.ndarray:
    """1-pixel boundary of a 2D binary mask."""
    inner = m.copy()
    inner[1:, :] &= m[:-1, :]
    inner[:-1, :] &= m[1:, :]
    inner[:, 1:] &= m[:, :-1]
    inner[:, :-1] &= m[:, 1:]
    return m & ~inner


def render(img2d, masks2d, center, width, alpha, mode, row_mm, col_mm, flip_ud=False):
    """Grey image + coloured ROIs → PIL image resized to its physical aspect ratio."""
    g = (window(img2d.astype(np.float32), center, width) * 255).astype(np.uint8)
    rgb = np.stack([g] * 3, -1).astype(np.float32)
    for name, (m, color) in masks2d.items():
        if mode == "Filled":
            rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(color)
        if mode in ("Outline", "Filled"):
            rgb[outline(m)] = color
    rgb = rgb.astype(np.uint8)
    if flip_ud:  # put the head at the top for coronal / sagittal views
        rgb = rgb[::-1]
    h_mm, w_mm = rgb.shape[0] * row_mm, rgb.shape[1] * col_mm
    out_w = DISPLAY_PX
    out_h = max(1, int(round(out_w * h_mm / w_mm)))
    return Image.fromarray(rgb).resize((out_w, out_h), Image.NEAREST)


def color_for(roi, i):
    return ROI_COLORS.get(roi, FALLBACK_COLORS[i % len(FALLBACK_COLORS)])


# ---------------------------------------------------------------- sidebar
patients = list_patients()
if not patients:
    st.error("No data found in data/raw/STS. Run `scripts/download_sts.py` first.")
    st.stop()

with st.sidebar:
    st.header("Patient")
    pid = st.selectbox("Patient ID", patients, index=len(patients) - 1)
    p = get_patient(pid)
    rois = list(p["masks"])
    shown = st.multiselect("ROIs to display", rois, default=rois)

    st.header("Display")
    preset = st.selectbox("Window", list(WINDOWS) + ["custom"])
    if preset == "custom":
        center = st.slider("Center (HU)", -1000, 1500, 40)
        width = st.slider("Width (HU)", 1, 4000, 400)
    else:
        center, width = WINDOWS[preset]
    st.caption(f"Showing HU in [{center - width // 2}, {center + width // 2}]")
    mode = st.radio("ROI overlay", ["Outline", "Filled", "Hidden"], horizontal=True)
    alpha = st.slider("Fill opacity", 0.0, 1.0, 0.35) if mode == "Filled" else 0.0

ct_arr, sx_sy_sz = p["ct_arr"], p["spacing"]
sx, sy, sz = sx_sy_sz
nz, ny, nx = ct_arr.shape
main_roi = "GTV_Mass" if "GTV_Mass" in rois else rois[0]
zc, yc, xc = [int(np.mean(i)) for i in np.nonzero(p["m_arr"][main_roi])]

st.title(f"{pid}")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Volume (voxels)", f"{nx}×{ny}×{nz}")
c2.metric("Spacing (mm)", f"{sx:.2f}×{sy:.2f}×{sz:.2f}")
c3.metric("ROIs", ", ".join(rois))
c4.metric(f"{main_roi} volume", f"{p['m_arr'][main_roi].sum() * sx * sy * sz / 1000:.1f} cm³")

tab_view, tab_hist, tab_res, tab_feat, tab_cohort = st.tabs(
    ["🩻 Slices", "📊 HU histogram", "🔲 Resampling", "🧮 Features", "👥 Cohort"]
)

# ---------------------------------------------------------------- tab: slices
with tab_view:
    centre = {"z": zc, "y": yc, "x": xc}
    reset = st.button("Center on tumour")
    for k, v in centre.items():
        if reset or f"s_{pid}_{k}" not in st.session_state:
            st.session_state[f"s_{pid}_{k}"] = v
    cols = st.columns(3)
    z = cols[0].slider("Axial slice (z)", 0, nz - 1, key=f"s_{pid}_z")
    y = cols[1].slider("Coronal slice (y)", 0, ny - 1, key=f"s_{pid}_y")
    x = cols[2].slider("Sagittal slice (x)", 0, nx - 1, key=f"s_{pid}_x")

    def m2d(sl):
        return {r: (p["m_arr"][r][sl], color_for(r, i)) for i, r in enumerate(rois) if r in shown}

    hidden = mode == "Hidden"
    views = [
        ("Axial", ct_arr[z], m2d(np.s_[z]), sy, sx, False),
        ("Coronal", ct_arr[:, y, :], m2d(np.s_[:, y, :]), sz, sx, True),
        ("Sagittal", ct_arr[:, :, x], m2d(np.s_[:, :, x]), sz, sy, True),
    ]
    for col, (title, img, ms, rmm, cmm, flip) in zip(cols, views):
        col.image(render(img, {} if hidden else ms, center, width, alpha, mode, rmm, cmm, flip),
                  caption=title, width="stretch")
    in_slice = {r: int(p["m_arr"][r][z].sum()) for r in shown}
    st.caption(f"Pixel under cross-hair (x={x}, y={y}, z={z}): **{ct_arr[z, y, x]} HU** · "
               f"ROI pixels on this axial slice: {in_slice}")

# ---------------------------------------------------------------- tab: histogram
with tab_hist:
    body = ct_arr > -500
    lo, hi = st.slider("HU range", -1000, 2000, (-300, 400))
    bins = np.linspace(lo, hi, 120)
    data = {"whole body (HU > -500)": np.histogram(ct_arr[body], bins, density=True)[0]}
    for r in shown:
        data[r] = np.histogram(ct_arr[p["m_arr"][r]], bins, density=True)[0]
    st.line_chart(pd.DataFrame(data, index=np.round((bins[:-1] + bins[1:]) / 2)))
    st.dataframe(pd.DataFrame({
        r: pd.Series(ct_arr[p["m_arr"][r]]).describe().round(1) for r in shown
    }))
    st.caption("Radiomics is computed on these raw HU values; the display window has no effect on it.")

# ---------------------------------------------------------------- tab: resampling
with tab_res:
    st.markdown("Crop around the ROI (+15 mm), then resample to **isotropic voxels**. "
                "B-spline for the image, nearest neighbour for the mask.")
    rc1, rc2 = st.columns([1, 3])
    roi_r = rc1.selectbox("ROI", rois, key="roi_r")
    iso = rc1.select_slider("Target spacing (mm)", [0.5, 1.0, 2.0, 3.0, 5.0], value=1.0)
    plane = rc1.radio("Plane", ["Coronal", "Sagittal", "Axial"])
    a0, m0, sp0, a1, m1, sp1 = get_resampled(pid, roi_r, iso)
    out = []
    for a, m, sp, label in ((a0, m0, sp0, "original"), (a1, m1, sp1, f"{iso} mm isotropic")):
        if plane == "Axial":
            k = a.shape[0] // 2
            im, mm, r, c, fl = a[k], m[k], sp[1], sp[0], False
        elif plane == "Coronal":
            k = a.shape[1] // 2
            im, mm, r, c, fl = a[:, k, :], m[:, k, :], sp[2], sp[0], True
        else:
            k = a.shape[2] // 2
            im, mm, r, c, fl = a[:, :, k], m[:, :, k], sp[2], sp[1], True
        ov = {} if mode == "Hidden" else {roi_r: (mm, color_for(roi_r, 0))}
        out.append((render(im, ov, center, width, alpha, mode if mode != "Hidden" else "Outline", r, c, fl),
                    f"{label}: grid {a.shape[::-1]} · spacing {tuple(round(s, 2) for s in sp)} mm"))
    with rc2:
        i1, i2 = st.columns(2)
        i1.image(*out[0], width="stretch")
        i2.image(*out[1], width="stretch")

# ---------------------------------------------------------------- tab: features
CLASS_INFO = {
    "shape": "3D geometry of the ROI, independent of intensities",
    "firstorder": "Histogram of HU values inside the ROI",
    "glcm": "Co-occurrence of grey levels between neighbouring voxel pairs",
    "glrlm": "Runs of consecutive voxels with the same grey level",
    "glszm": "Connected zones of the same grey level",
    "gldm": "How many neighbours are similar to each voxel",
    "ngtdm": "Difference between each voxel and its neighbourhood average",
}
with tab_feat:
    fc1, fc2 = st.columns([1, 3])
    roi_f = fc1.selectbox("ROI", rois, key="roi_f")
    fc1.caption("Settings: `config/pyradiomics_params.yaml`")
    if fc1.button("Extract features", type="primary") or st.session_state.get(f"done_{pid}_{roi_f}"):
        st.session_state[f"done_{pid}_{roi_f}"] = True
        df = get_features(pid, roi_f)
        with fc2:
            st.write(f"**{len(df)} features**")
            classes = df["class"].unique().tolist()
            cls = st.segmented_control("Class", classes, default=classes[0])
            if cls:
                st.caption(CLASS_INFO.get(cls, ""))
                st.dataframe(df[df["class"] == cls][["feature", "value"]].reset_index(drop=True),
                             width="stretch", height=420)
            st.download_button("Download CSV", df.to_csv(index=False), f"{pid}_{roi_f}_features.csv")
    else:
        fc2.info("Click **Extract features** (~15 s per ROI).")

# ---------------------------------------------------------------- tab: cohort
with tab_cohort:
    csv = ROOT / "outputs" / "features_demo.csv"
    if not csv.exists():
        st.info("Run notebook 01 (section 5) to build `outputs/features_demo.csv`.")
    else:
        feats = pd.read_csv(csv).set_index("PatientID")
        num = feats.drop(columns=["ROI"], errors="ignore")
        st.write(f"`{csv.relative_to(ROOT)}`: {num.shape[0]} patients × {num.shape[1]} features")
        f1 = st.selectbox("Feature", num.columns,
                          index=list(num.columns).index("original_shape_MeshVolume"))
        st.bar_chart(num[f1])
        f2 = st.selectbox("Compare with", num.columns,
                          index=list(num.columns).index("original_glcm_JointEntropy"))
        st.scatter_chart(num[[f1, f2]].reset_index(), x=f1, y=f2)
        rho = num[f1].corr(num[f2], method="spearman")
        st.caption(f"Spearman ρ = {rho:.2f} (n = {len(num)}, so read this with caution)")
