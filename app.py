# app.py (updated with dynamic axis labels)
from FitsFlow.csv_handler import ingest_csv_file
from FitsFlow.detectors import detect_anomalies, annotate_plotly
from FitsFlow.fields import map_columns
from FitsFlow.reporters import generate_pdf_report

import streamlit as st
import numpy as np
from astropy.io import fits
from scipy.signal import savgol_filter
import pandas as pd
import tempfile, os, io, time, re, hashlib
from typing import Tuple
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from astroquery.mast import Observations
from astroquery.simbad import Simbad

st.set_page_config(page_title="AstroFlow · FITSFlow", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Global limits  (tune as needed)
# ---------------------------
MAX_PRODUCTS   = 15         # Max MAST products downloaded in one click
MAX_FILE_MB    = 2048       # Warn (but still load) files larger than this (MB)
MAX_IMAGES     = 10         # Max 2D HDU images rendered in Images tab
MAX_HDU_ROWS   = 50_000     # Truncate very wide image HDUs before nanmean

# ---------------------------
import matplotlib as mpl
mpl.rcParams.update({
  "font.family": "serif",
  "font.size": 11,
  "axes.linewidth": 0.8,
  "xtick.direction": "in",
  "ytick.direction": "in",
  "xtick.minor.visible": True,
  "ytick.minor.visible": True,
  "figure.dpi": 150,
  "savefig.dpi": 300,
  "savefig.bbox": "tight",
  "axes.spines.top": False,
  "axes.spines.right": False,
})
# Helper: stable key generator
# ---------------------------
def make_key(*parts):
    raw = "_".join(str(p) for p in parts if p is not None)
    short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
    key = re.sub(r'\W+', '_', raw).strip('_')
    return f"{key}_{short_hash}"

# ---------------------------
# Helper / Processing Utils
# ---------------------------
WL_COLS = ['WAVELENGTH', 'WAVE', 'LAMBDA', 'WLEN', 'LAMBDA_MICRON', 'LAMBDA_UM', 'WAVELENGTH_MICRON']
FLUX_COLS = ['FLUX', 'FLUX_DENSITY', 'SPECTRUM', 'INTENSITY', 'FLUX_1', 'FLUX_0']

def safe_names(arr):
    try:
        return list(arr.names)
    except Exception:
        return []

def try_extract_spectrum(hdu):
    """
    Returns: (wl_array, fl_array, labels_dict)
    labels_dict contains keys: x_label, y_label
    """
    data = hdu.data
    default_labels = {"x_label": "Wavelength", "y_label": "Flux"}
    if data is None:
        return None, None, default_labels

    # Table-like HDU -> try pandas + map_columns
    try:
        if hasattr(data, 'names') or (hasattr(data, 'dtype') and data.dtype.names is not None):
            import pandas as _pd
            df = _pd.DataFrame(data)
            mapping = map_columns(df)
            wl_col = mapping.get("wavelength")
            fl_col = mapping.get("flux")
            if wl_col and fl_col and wl_col in df.columns and fl_col in df.columns:
                wl = _pd.to_numeric(df[wl_col], errors="coerce").to_numpy(dtype=float)
                fl = _pd.to_numeric(df[fl_col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(wl) & np.isfinite(fl)
                if np.any(mask):
                    labels = {"x_label": wl_col, "y_label": fl_col}
                    return wl[mask], fl[mask], labels

            # fallback: first two numeric columns
            names = safe_names(data)
            nums = [n for n in names if np.issubdtype(data[n].dtype, np.number)]
            if len(nums) >= 2:
                wl = np.array(data[nums[0]]).astype(float).flatten()
                fl = np.array(data[nums[1]]).astype(float).flatten()
                mask = np.isfinite(wl) & np.isfinite(fl)
                if np.any(mask):
                    labels = {"x_label": nums[0], "y_label": nums[1]}
                    return wl[mask], fl[mask], labels
    except Exception:
        pass

    # Image-like HDU: collapse to 1D or return pixel index
      # Image-like / multi-dimensional HDU support
    try:
        arr = np.asarray(data, dtype=float)
        if arr.ndim == 0:
            return None, None, default_labels

        if arr.ndim == 1:
            wl = np.arange(arr.size)
            fl = arr
        elif arr.ndim == 2:
            fl = np.nanmean(arr, axis=0)
            wl = np.arange(fl.size)
        elif arr.ndim == 3:
            collapsed = np.nanmean(arr, axis=0)
            fl = np.nanmean(collapsed, axis=0)
            wl = np.arange(fl.size)
        elif arr.ndim == 4:
            collapsed = np.nanmean(arr, axis=(0, 1))
            fl = np.nanmean(collapsed, axis=0)
            wl = np.arange(fl.size)
        else:
            return None, None, default_labels

        mask = np.isfinite(fl)
        if np.any(mask):
            return wl[mask], fl[mask], {"x_label": "Pixel", "y_label": "Mean Intensity"}
    except Exception:
        pass

    return None, None, default_labels

def interp_to_reference(wl, fl, ref_wl):
    try:
        return np.interp(ref_wl, wl, fl, left=np.nan, right=np.nan)
    except Exception:
        return np.full_like(ref_wl, np.nan)

#==============================
def build_stacked_spectrum(
    spectra,
    method="mean"
):
    min_wl = min(np.nanmin(r["wl"]) for r in spectra)
    max_wl = max(np.nanmax(r["wl"]) for r in spectra)

    ref_wl = np.linspace(min_wl, max_wl, 2000)

    interp_fluxes = [
        interp_to_reference(r["wl"], r["fl"], ref_wl)
        for r in spectra
    ]

    arr = np.array(interp_fluxes)

    # Guard: all-NaN input (e.g. collapsed 2D image HDUs) produces empty mean
    if arr.size == 0 or not np.any(np.isfinite(arr)):
        return ref_wl, np.full_like(ref_wl, np.nan)

    with np.errstate(invalid='ignore', divide='ignore'):
        stacked = np.nanmedian(arr, axis=0) if method == "median" else np.nanmean(arr, axis=0)

    # HARD CLEAN after stacking
    stacked = np.asarray(stacked, dtype=float)
    stacked[~np.isfinite(stacked)] = np.nan

    return ref_wl, stacked
#=============================

def smooth_flux(flux, window, polyorder):
    flux = np.asarray(flux, dtype=float)

    # remove inf
    flux[~np.isfinite(flux)] = np.nan

    finite = np.isfinite(flux)

    if finite.sum() < max(window, polyorder + 2):
        return flux

    # interpolate missing values
    if not np.all(finite):
        x = np.arange(len(flux))
        flux = np.interp(x, x[finite], flux[finite])

    # ensure valid window
    if window >= len(flux):
        window = len(flux) - 1 if len(flux) % 2 == 0 else len(flux)

    if window % 2 == 0:
        window += 1

    if window < polyorder + 2:
        return flux

    try:
        return savgol_filter(flux, window, polyorder)
    except Exception:
        return flux

def calc_snr_on_band(ref_wl, ref_flux, band_range: Tuple[float, float]):
    start, end = band_range
    mask = (ref_wl >= start) & (ref_wl <= end)
    if not np.any(mask):
        return 0.0
    signal = np.abs(1 - np.nanmean(ref_flux[mask]))
    left_mask = (ref_wl >= (start - 0.3)) & (ref_wl <= (start - 0.1))
    right_mask = (ref_wl >= (end + 0.1)) & (ref_wl <= (end + 0.3))
    noise_vals = []
    if np.any(left_mask):
        noise_vals.append(np.nanstd(ref_flux[left_mask]))
    if np.any(right_mask):
        noise_vals.append(np.nanstd(ref_flux[right_mask]))
    noise = np.nanmean(noise_vals) if noise_vals else np.nanstd(ref_flux)
    if noise == 0 or np.isnan(noise):
        return 0.0
    return float(signal / noise)

def fits_open_smart(path):
    """
    Open a FITS file with memmap=True when safe, falling back to
    memmap=False when BZERO/BSCALE/BLANK keywords are present (astropy
    requires in-memory scaling for those files).  Returns the open HDUList;
    caller must use it as a context manager or close it manually.
    """
    try:
        hdul = fits.open(path, memmap=True)
        # Trigger a header read to surface the BZERO conflict early
        _ = hdul[0].header
        return hdul
    except Exception as e:
        if "BZERO" in str(e) or "BSCALE" in str(e) or "BLANK" in str(e) or "memory-mapped" in str(e):
            return fits.open(path, memmap=False)
        raise


# ==========================================================
# SIMBAD INTEGRATION
# ==========================================================
@st.cache_data(ttl=3600)
def simbad_query(target_name: str):
    try:
        simbad = Simbad()
        simbad.add_votable_fields("sptype", "distance", "flux(V)", "otype")
        result = simbad.query_object(target_name)
        if result is None or len(result) == 0:
            return None
        row = result[0]
        def safe(col):
            try:
                v = row[col]
                return str(v) if v is not None and str(v) != "--" else None
            except Exception:
                return None
        return {
            "main_id": safe("MAIN_ID"),
            "otype": safe("OTYPE"),
            "sptype": safe("SP_TYPE"),
            "distance": safe("Distance_distance"),
            "dist_unit": safe("Distance_unit"),
            "flux_v": safe("FLUX_V"),
        }
    except Exception as e:
        return {"_error": str(e)}


@st.cache_data(ttl=3600)
def mast_search_target(target_name, mission=None, radius="0.05 deg"):
    """
    Search MAST archive for a target.
    """
    try:
        obs = Observations.query_object(
            target_name,
            radius=radius
        )

        if mission and mission != "All":
            obs = obs[
                obs["obs_collection"] == mission
            ]

        return obs

    except Exception as e:
        st.error(f"MAST search failed: {e}")
        return None

def mast_download_products(observation_row):
    """
    Download science FITS products for a selected observation.
    Limits downloads to MAX_PRODUCTS.
    """
    try:
        products = Observations.get_product_list(observation_row)

        try:
            products = Observations.filter_products(
                products, productType="SCIENCE", extension="fits"
            )
        except Exception:
            products = Observations.filter_products(
                products, productType="SCIENCE"
            )

        n_total = len(products)
        if n_total == 0:
            st.warning("No SCIENCE FITS products found for this observation.")
            return None

        if n_total > MAX_PRODUCTS:
            st.warning(
                f"Found **{n_total}** science products. Downloading the first "
                f"**{MAX_PRODUCTS}**. Use the [MAST Portal](https://mast.stsci.edu) "
                f"for bulk downloads."
            )
            products = products[:MAX_PRODUCTS]

        manifest = Observations.download_products(products, cache=True)
        return manifest

    except Exception as e:
        st.error(f"MAST download failed: {e}")
        return None

@st.cache_data(ttl=3600)
def mast_import_fits(file_path):
    """
    Import a downloaded FITS into AstroFlow format.
    Large files produce a warning but are still processed.
    BZERO/BSCALE/BLANK files are opened memmap=False automatically.
    """
    imported_results = []

    try:
        file_mb = os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        file_mb = 0

    if file_mb > MAX_FILE_MB:
        st.warning(
            f"⚠️ **{os.path.basename(file_path)}** is large ({file_mb:.0f} MB). "
            f"Loading may take a while — a progress bar is shown above."
        )

    try:
        with fits_open_smart(file_path) as hdul:
            for idx, hdu in enumerate(hdul):
                wl, fl, labels = try_extract_spectrum(hdu)
                if wl is None:
                    continue
                imported_results.append({
                    "file":      os.path.basename(file_path),
                    "path":      file_path,
                    "hdu_index": idx,
                    "header":    dict(hdu.header),
                    "wl":        np.array(wl, dtype=float),
                    "fl":        np.array(fl, dtype=float),
                    "err":       None,
                    "x_label":   labels.get("x_label", "Wavelength"),
                    "y_label":   labels.get("y_label", "Flux"),
                })
    except Exception as e:
        st.error(f"Failed to import FITS {os.path.basename(file_path)}: {e}")

    return imported_results

# ==========================================================
# END MAST INTEGRATION
# ==========================================================

# ---------------------------
# Sidebar controls (UI)
# ---------------------------
st.sidebar.header("AstroFlow Controls")
st.sidebar.markdown("Upload FITS/CSV files and toggle analysis options.")

with st.sidebar.expander("Spectrum Processing", expanded=True):
    enable_downloads   = st.checkbox("Enable downloads", value=True, key="cb_enable_dl")
    smoothing_enabled  = st.checkbox("Enable smoothing", value=True, key="cb_smoothing")
    smoothing_window   = st.slider("Smoothing window (odd)", 5, 501, 51, step=2)
    polyorder          = st.slider("SavGol polyorder", 1, 5, 3)
    show_errorbars     = st.checkbox("Show error bars (if available)", value=False, key="cb_errorbars")

with st.sidebar.expander("MAST Archive", expanded=True):
    mast_target = st.text_input(
        "Target Name",
        value="K2-18"
    )

    mast_mission = st.selectbox(
        "Mission",
        [
            "All",
            "JWST",
            "HST",
            "TESS",
            "Kepler"
        ]
    )

    st.caption(f"Up to {MAX_PRODUCTS} products per download · files >{MAX_FILE_MB} MB get a warning")

    mast_search_btn = st.button("Search MAST")

with st.sidebar.expander("SIMBAD Lookup", expanded=False):
    simbad_target  = st.text_input("Object name", value="", key="simbad_input",
                                   placeholder="e.g. K2-18, TRAPPIST-1")
    simbad_btn     = st.button("Query SIMBAD", key="simbad_btn")
    if simbad_btn and simbad_target.strip():
        with st.spinner("Querying SIMBAD…"):
            st.session_state["simbad_result"] = simbad_query(simbad_target.strip())
    if st.session_state.get("simbad_result"):
        sr = st.session_state["simbad_result"]
        if "_error" in sr:
            st.warning(f"SIMBAD error: {sr['_error']}")
        else:
            for label, key in [
                ("Main ID",       "main_id"),
                ("Object type",   "otype"),
                ("Spectral type", "sptype"),
                ("Distance",      "distance"),
                ("V mag",         "flux_v"),
                ("Radial vel.",   "rv"),
                ("Redshift",      "redshift"),
            ]:
                val = sr.get(key)
                if val:
                    unit = sr.get("dist_unit", "") if key == "distance" else ""
                    st.caption(f"**{label}:** {val} {unit}".strip())

  
# ==========================================================
# RUN MAST SEARCH
# ==========================================================

if mast_search_btn:
    with st.spinner("Searching MAST..."):
        mast_results = mast_search_target(
            mast_target,
            mast_mission
        )

        if (
            mast_results is not None
            and len(mast_results) > 0
        ):
            st.session_state["mast_results"] = mast_results
            st.session_state["mast_imported_results"] = []
        else:
            st.warning(
                "No observations found."
            )


# ---------------------------
# Main UI area
# ---------------------------
st.title("🔭 AstroFlow · FITS Processor")
st.markdown("Upload FITS or CSV files (JWST/HST/TESS/generic)")

uploaded = st.file_uploader(
    "Upload one or more FITS/CSV files",
    type=["fits", "csv"],
    accept_multiple_files=True
)

mast_results = st.session_state.get("mast_results")
mast_imported_results = st.session_state.get("mast_imported_results", [])

if not uploaded and mast_results is None and len(mast_imported_results) == 0:
    st.info("Upload FITS or CSV spectral files to start or search the MAST archive")
    st.stop()

# --- Save uploaded files and process each (FITS or CSV) ---
tmpdir = tempfile.mkdtemp()
file_paths = []
uploaded_results = []
nfiles = len(uploaded) if uploaded else 0
progress = st.progress(0) if uploaded else None
status_text = st.empty() if uploaded else None

if uploaded:
    for i, up in enumerate(uploaded, start=1):
        pct = int((i - 1) / nfiles * 100)
        progress.progress(pct, text=f"Processing {up.name} ({i}/{nfiles})…")
        if status_text:
            file_mb = len(up.getvalue()) / (1024 * 1024)
            status_text.caption(f"📂 Loading **{up.name}** — {file_mb:.1f} MB")

        fname = up.name
        dst = os.path.join(tmpdir, fname)
        with open(dst, "wb") as f:
            f.write(up.getvalue())
        file_paths.append(dst)

        # File size guard for uploaded files too
        file_mb = os.path.getsize(dst) / (1024 * 1024)
        if file_mb > MAX_FILE_MB:
            st.warning(
                f"**{fname}** is {file_mb:.0f} MB (>{MAX_FILE_MB} MB limit) — skipped. "
                f"Consider pre-processing large files locally."
            )
            continue

        lower = fname.lower()
        # CSV handling
        if lower.endswith(".csv"):
            try:
                csv_outputs = ingest_csv_file(dst, filename=fname)
                for out in csv_outputs:
                    if out.get("wl") is not None:
                        out["wl"] = np.asarray(out["wl"], dtype=float)
                    if out.get("fl") is not None:
                        out["fl"] = np.asarray(out["fl"], dtype=float)
                    # derive labels from orig_df if possible
                    if out.get("orig_df") is not None:
                        mapping = map_columns(out["orig_df"])
                        x_label = mapping.get("wavelength") or mapping.get("time") or mapping.get("x") or mapping.get("lat") or "X"
                        y_label = mapping.get("flux") or mapping.get("value") or mapping.get("y") or mapping.get("temp") or "Y"
                        out.setdefault("x_label", x_label)
                        out.setdefault("y_label", y_label)
                    else:
                        out.setdefault("x_label", "Wavelength")
                        out.setdefault("y_label", "Flux")
                    uploaded_results.append(out)
            except Exception as e:
                st.error(f"Failed to parse CSV {fname}: {e}")
            continue

        # FITS handling — fits_open_smart handles BZERO/BSCALE/BLANK automatically
        try:
            file_mb_fits = os.path.getsize(dst) / (1024 * 1024)
            if file_mb_fits > MAX_FILE_MB:
                st.warning(
                    f"⚠️ **{fname}** is large ({file_mb_fits:.0f} MB). "
                    f"Loading — this may take a moment."
                )
            with fits_open_smart(dst) as hdul:
                found_any = False
                for idx, hdu in enumerate(hdul):
                    wl, fl, labels = try_extract_spectrum(hdu)
                    if wl is None:
                        continue
                    found_any = True
                    err = None
                    uploaded_results.append({
                        "file": fname,
                        "path": dst,
                        "hdu_index": idx,
                        "header": dict(hdu.header) if hasattr(hdu, "header") else {},
                        "wl": np.array(wl, dtype=float),
                        "fl": np.array(fl, dtype=float),
                        "err": err,
                        "x_label": labels.get("x_label", "Wavelength"),
                        "y_label": labels.get("y_label", "Flux"),
                    })
                if not found_any:
                    uploaded_results.append({
                        "file": fname,
                        "path": dst,
                        "hdu_index": None,
                        "header": {},
                        "wl": None,
                        "fl": None,
                        "err": None,
                        "x_label": "Wavelength",
                        "y_label": "Flux",
                    })
        except Exception as e:
            st.error(f"Failed to open {fname}: {e}")

    progress.progress(100, text="✅ All files processed.")
    if status_text:
        status_text.empty()

mast_imported_results = st.session_state.get("mast_imported_results", [])
results = uploaded_results + mast_imported_results

#=============================================
st.sidebar.markdown("---")
with st.sidebar.expander("Session Summary", expanded=False):
    num_spectra = sum(
        1
        for r in results
        if r.get("wl") is not None
    )

    num_files_with_path = sum(
        1
        for r in results
        if r.get("path")
    )

    st.metric("Spectra", num_spectra)
    st.metric("Files", len(results))
    st.metric("MAST Imports", len(mast_imported_results))
#====================================================

if len(results) == 0 and mast_results is None:
    st.error("No spectra could be extracted from uploaded files. You may upload pre-processed wavelength+flux CSVs.")
    st.stop()

tabs = st.tabs([
    "MAST Archive",
    "Spectrum",
    "Data Table",
    "Images",
    "Reports",
    "Anomalies"
])

def plot_spectrum_interactive(
    wl, fl, fl_smooth=None, err=None, title="Spectrum", bands=None,
    show_bands_flag=True, show_error=False, x_label="Wavelength", y_label="Flux"
):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wl, y=fl, mode='lines', name='raw', line=dict(color='rgba(0,150,200,0.7)')))
    if fl_smooth is not None:
        fig.add_trace(go.Scatter(x=wl, y=fl_smooth, mode='lines', name='smoothed', line=dict(color='black', width=2)))
    if show_error and err is not None:
        fig.add_trace(go.Scatter(x=wl, y=fl+err, mode='lines', name='err+', line=dict(width=0), showlegend=False, opacity=0.2))
        fig.add_trace(go.Scatter(x=wl, y=fl-err, mode='lines', name='err-', line=dict(width=0), showlegend=False, opacity=0.2))
    if show_bands_flag and bands:
        for mol, (a, b) in bands.items():
            fig.add_vrect(x0=a, x1=b, fillcolor="LightSkyBlue", opacity=0.25, layer="below", line_width=0, annotation_text=mol, annotation_position="top left")
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white", height=400)
    return fig

# ==========================================================
# MAST ARCHIVE TAB
# ==========================================================

with tabs[0]:
    st.header("MAST Archive")

    if "mast_results" not in st.session_state:
        st.info("Search for a target using the sidebar.")
    else:
        obs = st.session_state["mast_results"]

        display_cols = [
            c for c in [
                "target_name",
                "obs_collection",
                "instrument_name",
                "obs_id",
                "t_exptime"
            ]
            if c in obs.colnames
        ]

        if len(display_cols) > 0:
            st.dataframe(obs[display_cols], use_container_width=True)
        else:
            st.dataframe(obs, use_container_width=True)

        obs_ids = list(obs["obs_id"]) if "obs_id" in obs.colnames else []

        if len(obs_ids) > 0:
            selected_obs = st.selectbox(
                "Observation",
                obs_ids
            )

            if st.button("Download and Import FITS"):
                selected_row = obs[
                    obs["obs_id"] == selected_obs
                ]

                dl_status = st.empty()
                dl_bar = st.progress(0, text="Fetching product list…")

                with st.spinner("Downloading from MAST…"):
                    manifest = mast_download_products(
                        selected_row
                    )

                if manifest is not None:
                    imported = 0
                    new_imported = st.session_state.get("mast_imported_results", [])

                    manifest_colnames = list(manifest.colnames) if hasattr(manifest, "colnames") else []
                    local_path_col = None
                    if "Local Path" in manifest_colnames:
                        local_path_col = "Local Path"
                    elif "local_path" in manifest_colnames:
                        local_path_col = "local_path"

                    n_manifest = len(manifest)
                    for mi, row in enumerate(manifest):
                        dl_bar.progress(
                            int((mi + 1) / max(n_manifest, 1) * 100),
                            text=f"Importing file {mi + 1}/{n_manifest}…"
                        )
                        local_path = None
                        if local_path_col is not None:
                            try:
                                local_path = row[local_path_col]
                            except Exception:
                                local_path = None

                        if (
                            local_path
                            and str(local_path).lower().endswith((".fits", ".fits.gz"))
                            and os.path.exists(str(local_path))
                        ):
                            dl_status.caption(f"📥 Importing: `{os.path.basename(str(local_path))}`")
                            imported_data = mast_import_fits(str(local_path))
                            new_imported.extend(imported_data)
                            imported += len(imported_data)

                    dl_bar.progress(100, text="✅ Import complete.")
                    dl_status.empty()
                    st.session_state["mast_imported_results"] = new_imported
                    st.success(f"Imported {imported} spectra")
                    st.rerun()
        else:
            st.info("No observation IDs available in the current MAST search results.")


# ==================== COMBINED SPECTRUM TAB ====================
with tabs[1]:
    st.header("Spectrum")

    for ri, res in enumerate(results):
        if res.get("wl") is None or res.get("fl") is None:
            continue

        label = f"{res['file']} (HDU {res.get('hdu_index')})"
        with st.expander(label, expanded=False):
            wl = res['wl']
            fl = res['fl']
            x_label = res.get("x_label", "Wavelength")
            y_label = res.get("y_label", "Flux")

            fl_smooth = smooth_flux(fl.copy(), smoothing_window, polyorder) if smoothing_enabled else None

            # Loop counter ri guarantees uniqueness across multiple HDUs from the same file
            chart_key = make_key(res.get('file', ''), res.get('hdu_index', ''), ri, 'spectrum')

            fig = plot_spectrum_interactive(
                wl, fl,
                fl_smooth=fl_smooth,
                err=res.get('err'),
                title=label,
                bands=None,
                show_bands_flag=False,
                show_error=show_errorbars,
                x_label=x_label,
                y_label=y_label
            )
            st.plotly_chart(fig, width='stretch', key=chart_key)

            if enable_downloads:
                df_data = {x_label: wl, y_label: fl}
                if fl_smooth is not None:
                    df_data[f"{y_label}_smoothed"] = fl_smooth
                df = pd.DataFrame(df_data)
                st.download_button(
                    f"Download CSV - {res['file']}",
                    df.to_csv(index=False).encode('utf-8'),
                    file_name=f"{res['file']}_hdu{res.get('hdu_index')}.csv",
                    mime='text/csv',
                    key=make_key(res.get('file'), res.get('hdu_index'), ri, 'dl')
                )


# Data Table tab
with tabs[2]:
    st.header("Data Table")
    for r in results:
        label = f"{r['file']} (HDU {r.get('hdu_index')})"
        st.subheader(label)
        if r.get("wl") is not None and r.get("fl") is not None:
            df = pd.DataFrame({r.get("x_label", "Wavelength"): r['wl'], r.get("y_label", "Flux"): r['fl']})
        elif r.get("orig_df") is not None:
            df = r.get("orig_df")
        else:
            st.write("No 1D data for this file.")
            continue
        st.dataframe(df.head(500), use_container_width=True)
        if enable_downloads:
            dl_key = make_key(r.get('file'), r.get('hdu_index'), 'download', 'table_csv')
            st.download_button(f"Download CSV: {label}", df.to_csv(index=False).encode('utf-8'), file_name=f"{label}.csv", mime='text/csv', key=dl_key)


# Images tab
with tabs[3]:
    st.header("FITS Images")
    found_image = False
    seen_files = set()
    image_render_count = 0

    # Per-image display controls
    img_col1, img_col2 = st.columns(2)
    with img_col1:
        img_cmap = st.selectbox(
            "Colormap",
            ["gray", "viridis", "inferno", "plasma", "cividis", "hot"],
            index=0
        )
    with img_col2:
        img_lognorm = st.checkbox("Log scale (LogNorm)", value=False,
            help="Useful for wide dynamic range images (HST, PS1)")

    if image_render_count == 0 and not any(
        r.get("path") for r in results
    ):
        st.info("No FITS files with 2D image HDUs found.")

    for r in results:
        if image_render_count >= MAX_IMAGES:
            st.warning(
                f"Reached the **{MAX_IMAGES}-image** render limit. "
                f"Remaining 2D HDUs skipped to protect memory. "
                f"Adjust `MAX_IMAGES` in the source for more."
            )
            break

        file_path = r.get("path")

        if not file_path:
            continue

        if file_path in seen_files:
            continue

        seen_files.add(file_path)

        try:
            with fits_open_smart(file_path) as hdul:
                for idx, hdu in enumerate(hdul):
                    if image_render_count >= MAX_IMAGES:
                        break

                                          if hdu.data is None:
                        continue
                    plot_data = np.asarray(hdu.data, dtype=float)

                    if plot_data.ndim == 2:
                        display_data = plot_data
                    elif plot_data.ndim == 3:
                        display_data = plot_data[0] if len(plot_data) > 0 else plot_data
                    elif plot_data.ndim == 4:
                        display_data = plot_data[0, 0] if len(plot_data) > 0 else plot_data
                    else:
                        continue

                    found_image = True
                    image_render_count += 1
                    st.subheader(f"{r['file']} (HDU {idx}) — Image {display_data.shape}")

                    fig, ax = plt.subplots(figsize=(7, 5), dpi=120)

                    try:
                        import matplotlib.colors as mcolors
                        if img_lognorm and np.any(display_data > 0):
                            vmin = np.nanpercentile(display_data[display_data > 0], 1)
                            vmax = np.nanpercentile(display_data, 99)
                            norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=max(vmax, 1e-9))
                        else:
                            vmin = np.nanpercentile(display_data, 1)
                            vmax = np.nanpercentile(display_data, 99)
                            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                        im = ax.imshow(display_data, cmap=img_cmap, origin="lower", aspect="auto", norm=norm)
                    except Exception:
                        im = ax.imshow(display_data, cmap=img_cmap, origin="lower", aspect="auto")

                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    ax.set_title(f"{r['file']} · HDU {idx}", fontsize=10)
                    st.pyplot(fig)

                        if enable_downloads:
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
                            buf.seek(0)
                            dl_key = make_key(r['file'], idx, 'image_download')
                            st.download_button(
                                label=f"⬇ Download Image (PNG, 200 dpi) — {r['file']} HDU {idx}",
                                data=buf,
                                file_name=f"{r['file']}_hdu{idx}_{img_cmap}.png",
                                mime="image/png",
                                key=dl_key
                            )
                        plt.close(fig)
        except Exception as e:
            st.warning(f"Could not open {r.get('file')} for images: {e}")
    if not found_image:
        st.info("No 2D images found in uploaded FITS files.")


# Reports tab
with tabs[4]:
    st.header("Generate PDF Report")
    st.markdown("Compile spectra plots and FITS images into a single PDF.")

    # Report metadata inputs
    rpt_col1, rpt_col2 = st.columns(2)
    with rpt_col1:
        rpt_target = st.text_input("Target / Object name", value=mast_target if mast_target else "Unknown Target")
        rpt_author = st.text_input("Author(s)", value="AstroFlow User")
    with rpt_col2:
        rpt_instrument = st.text_input("Instrument / Mission", value="")
        rpt_notes = st.text_area("Report notes (optional)", value="", height=68)

    if st.button("Generate Report"):
        tmp_pdf = os.path.join(tempfile.gettempdir(), f"astroflow_report_{int(time.time())}.pdf")
        plots = []
        images = []
        # tables = []  # Removed - no CSV tables in PDF

        spec_results_for_report = [r for r in results if r.get("wl") is not None and r.get("fl") is not None]

        report_progress = st.progress(0, text="Building report…")
        n_report_steps = max(len(spec_results_for_report) + 1, 1)

        # Save 1D spectra as clean PNGs
        for ri, res in enumerate(spec_results_for_report):
            report_progress.progress(
                int(ri / n_report_steps * 80),
                text=f"Plotting spectrum {ri + 1}/{len(spec_results_for_report)}…"
            )
            wl = res["wl"]
            fl = res["fl"]
            x_label = res.get("x_label", "Wavelength")
            y_label = res.get("y_label", "Flux")

            fl_smooth_rpt = smooth_flux(fl.copy(), smoothing_window, polyorder) if smoothing_enabled else None

            buf = io.BytesIO()
            fig_rpt, ax_rpt = plt.subplots(figsize=(8, 4.5), dpi=200)

            # Raw spectrum
            ax_rpt.plot(wl, fl, color='steelblue', alpha=0.7, linewidth=1.2, label='Raw')

            # Smoothed spectrum (if enabled)
            if fl_smooth_rpt is not None:
                ax_rpt.plot(wl, fl_smooth_rpt, color='red', linewidth=2.0, label='Smoothed')

            # Ensure full data range is shown clearly
            ax_rpt.set_xlabel(x_label, fontsize=11)
            ax_rpt.set_ylabel(y_label, fontsize=11)
            ax_rpt.set_title(f"{res['file']} · HDU {res.get('hdu_index')}", fontsize=12)

            # Auto-adjust limits to data range
            ax_rpt.margins(x=0.02, y=0.05)
            ax_rpt.autoscale(enable=True, axis='both', tight=False)

            ax_rpt.legend(fontsize=10, loc='best')
            ax_rpt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig_rpt)
            buf.seek(0)

            img_path = os.path.join(tempfile.gettempdir(), f"{res['file']}_hdu{res.get('hdu_index')}_spectrum.png")
            with open(img_path, "wb") as fh:
                fh.write(buf.read())
            plots.append(img_path)

        # Collect 2D FITS images
        img_count_rpt = 0
        for r in results:
            if img_count_rpt >= MAX_IMAGES:
                break
            if not r.get("path"):
                continue
            try:
                with fits_open_smart(r["path"]) as hdul:
                    for idx, hdu in enumerate(hdul):
                        if img_count_rpt >= MAX_IMAGES:
                            break
                                                if hdu.data is None:
                            continue
                        plot_data = np.asarray(hdu.data, dtype=float)

                        if plot_data.ndim == 2:
                            display_data = plot_data
                        elif plot_data.ndim == 3:
                            display_data = plot_data[0]
                        elif plot_data.ndim == 4:
                            display_data = plot_data[0, 0]
                        else:
                            continue

                        img_path = ...
                        vmin = np.nanpercentile(display_data, 1)
                        vmax = np.nanpercentile(display_data, 99)
                        plt.imsave(img_path, np.clip(display_data, vmin, vmax), cmap="gray", origin="lower")
                            images.append(img_path)
                            img_count_rpt += 1
            except Exception as e:
                st.warning(f"Could not read images from {r.get('file')}: {e}")

        report_progress.progress(90, text="Compiling PDF…")

        # Sanitized metadata
        safe_title = f"AstroFlow Analysis Report - {rpt_target}".replace('\u2014', '-').replace('\u2013', '-')
        safe_notes = (rpt_notes or "").replace('\u2014', '-').replace('\u2013', '-').replace('\u2018', "'").replace('\u2019', "'")

        simbad_info = st.session_state.get("simbad_result") or {}
        report_metadata = {
            "title":     safe_title,
            "author":    rpt_author,
            "target":    rpt_target,
            "instrument": rpt_instrument,
            "notes":     safe_notes,
            "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "n_spectra": len(spec_results_for_report),
            "files":     list({r["file"] for r in spec_results_for_report}),
            "simbad_otype":  simbad_info.get("otype", ""),
            "simbad_sptype": simbad_info.get("sptype", ""),
            "simbad_dist":   simbad_info.get("distance", ""),
        }

        # Generate PDF
        try:
            pdf_path = generate_pdf_report(
                output_path=tmp_pdf,
                metadata=report_metadata,
                plots=plots,
                tables=[],      # Empty - no tables
                images=images,
            )
        except Exception as e:
            st.error(f"Failed to generate PDF report: {e}")
            report_progress.empty()
            st.stop()

        report_progress.progress(100, text="✅ Report ready.")

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                rpt_key_n = st.session_state.get("_rpt_dl_n", 0) + 1
                st.session_state["_rpt_dl_n"] = rpt_key_n
                st.download_button(
                    label="⬇ Download PDF Report",
                    data=f,
                    file_name=os.path.basename(pdf_path),
                    mime="application/pdf",
                    key=make_key('pdf_report', rpt_key_n)
                )
        else:
            st.error("PDF report was not generated.")

# ---------------------------
# Anomaly Detection tab
# ---------------------------
with tabs[5]:
    st.header("Anomaly Detection")
    st.markdown(
        "Detects **spikes**, **dips**, and **statistical outliers** in each spectrum. "
        "Tune thresholds below — lower z-threshold = more sensitive to outliers."
    )

    with st.sidebar.expander("Anomaly Detection Settings", expanded=False):
        z_thresh    = st.slider("Outlier z-threshold", 2, 10, 4,
                                help="Lower = catches more (and more noise). 4 is conservative.")
        dip_window  = st.slider("Dip median window (px)", 11, 501, 101, step=2,
                                help="Rolling window for local continuum estimation.")
        dip_depth   = st.number_input("Dip minimum depth fraction", min_value=0.0001,
                                      max_value=1.0, value=0.01, step=0.001,
                                      help="Fraction below local median to flag as a dip.")
        spike_window = st.slider("Spike window (px)", 3, 101, 11, step=2)
        spike_std    = st.slider("Spike std-factor", 2, 20, 6,
                                 help="How many σ above rolling median = spike.")
        min_prominence = st.slider("Min dip prominence (σ)", 0.5, 10.0, 2.0, step=0.5,
                                   help="Extra filter: dip must be this many σ deep to survive.")

    anomalies_all = []
    expected_keys = ["type", "wl", "index", "value"]  # defined here so the summary section always has it

    for ai, res in enumerate(results):
        if res.get("wl") is None or res.get("fl") is None:
            continue

        wl = res["wl"]
        fl = res["fl"]
        x_label = res.get("x_label", "Wavelength")
        y_label = res.get("y_label", "Flux")

        params = {
            "z_thresh":       z_thresh,
            "dip_window":     dip_window,
            "dip_depth":      dip_depth,
            "spike_window":   spike_window,
            "spike_std":      spike_std,
            "min_prominence": min_prominence,
        }
        anoms = detect_anomalies(wl, fl, params=params)

        for a in anoms:
            a["file"]      = res.get("file")
            a["hdu_index"] = res.get("hdu_index")

        anomalies_all += anoms

        st.subheader(f"{res['file']} (HDU {res.get('hdu_index')})")

        fig = plot_spectrum_interactive(
            wl, fl,
            title=f"{res['file']} (HDU {res.get('hdu_index')})",
            x_label=x_label,
            y_label=y_label
        )
        fig = annotate_plotly(fig, anoms)
        # ai in key guarantees uniqueness when multiple HDUs come from the same file
        st.plotly_chart(fig, width='stretch',
                        key=make_key(res['file'], res.get('hdu_index'), ai, 'anomaly_plot'))

        # Normalize anomalies
        normalized_anoms = [{k: a.get(k, np.nan) for k in expected_keys} for a in anoms]

        if normalized_anoms:
            df_anoms = pd.DataFrame(normalized_anoms)

            # Summary metrics
            anom_types = df_anoms["type"].value_counts()
            cols_anom = st.columns(len(anom_types))
            for ci, (atype, cnt) in enumerate(anom_types.items()):
                cols_anom[ci].metric(atype.capitalize(), cnt)

            st.dataframe(df_anoms.head(200), use_container_width=True)

            if enable_downloads:
                import json
                st.download_button(
                    f"Download anomalies JSON - {res['file']}",
                    json.dumps(anoms, indent=2, default=str).encode('utf-8'),
                    file_name=f"{res['file']}_hdu{res.get('hdu_index')}_anomalies.json",
                    mime="application/json",
                    key=make_key(res['file'], res.get('hdu_index'), ai, 'anoms_json')
                )
                st.download_button(
                    f"Download anomalies CSV - {res['file']}",
                    df_anoms.to_csv(index=False).encode('utf-8'),
                    file_name=f"{res['file']}_hdu{res.get('hdu_index')}_anomalies.csv",
                    mime="text/csv",
                    key=make_key(res['file'], res.get('hdu_index'), ai, 'anoms_csv')
                )
        else:
            st.info("No anomalies detected for this spectrum with current thresholds.")

    st.markdown("### Summary")
    st.write(f"Total anomalies detected across all files: **{len(anomalies_all)}**")

    if anomalies_all and enable_downloads:
        normalized_all = [
            {k: a.get(k, np.nan) for k in expected_keys + ["file", "hdu_index"]}
            for a in anomalies_all
        ]
        df_all = pd.DataFrame(normalized_all)
        st.download_button(
            "Download all anomalies (CSV)",
            df_all.to_csv(index=False).encode('utf-8'),
            file_name="astroflow_anomalies_all.csv",
            mime='text/csv',
            key=make_key('all', 'anomalies', 'csv')
        )
