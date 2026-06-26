# app.py (updated with dynamic axis labels)
from FitsFlow.csv_handler import ingest_csv_file
from FitsFlow.detectors import annotate_plotly
from FitsFlow.fields import map_columns
from FitsFlow.reporters import generate_pdf_report

import streamlit as st
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clip, mad_std
from scipy.signal import savgol_filter, find_peaks, peak_widths
try:
    from astropy import units as u
    from astropy.nddata import StdDevUncertainty
    from specutils import Spectrum1D
    from specutils.fitting import find_lines_threshold
except Exception:
    u = None
    StdDevUncertainty = None
    Spectrum1D = None
    find_lines_threshold = None
import pandas as pd
import tempfile, os, io, time, re, hashlib
from typing import Tuple
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from astroquery.mast import Observations

st.set_page_config(page_title="AstroFlow · FITSFlow", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
# Global safety limits
# ---------------------------
MAX_PRODUCTS   = 5          # Max MAST products to download in one click
MAX_FILE_MB    = 500        # Skip FITS files larger than this (MB)
MAX_IMAGES     = 100         # Max 2D HDU images rendered in Images tab
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

NON_SCIENCE_EXTNAMES = {
    # Data quality / flag arrays
    'DQ', 'DQ1', 'DQ2', 'DQ3',
    # Error / uncertainty arrays
    'ERR', 'ERR1', 'ERR2', 'SIGMA', 'NOISE', 'STDEV',
    # Variance arrays (JWST pipeline products)
    'VAR_POISSON', 'VAR_RNOISE', 'VAR_FLAT', 'VAR_RAMP',
    # Weight / exposure maps
    'WHT', 'WEIGHT', 'EXP', 'EXPTIME', 'CTX',
    # Background arrays
    'BKG', 'BACKGROUND',
    # Contamination / model
    'CONTAM', 'MODEL',
    # Kernel / PSF
    'KERNEL', 'PSF',
    # Sample / groupdq (JWST raw ramp data)
    'GROUPDQ', 'PIXELDQ',
    # Wavelength solution / reference arrays stored as calibration tables
    'WAVELENGTH', 'WCSCORR', 'HDRTAB', 'ASDF',
    # HST-specific calibration extensions
    'D2IMARR', 'WCSDVARR', 'SIPWCS',
}

SUPPORTED_SCIENCE_TABLE_MIN_ROWS = 5
MAX_MAST_RESULTS = 200
MAST_TIMEOUT_S = 25
ANOMALY_KEYS = ["type", "wl", "index", "value", "score", "prominence", "width", "method"]

def safe_names(arr):
    try:
        return list(arr.names)
    except Exception:
        return []


@contextmanager
def open_fits_best_effort(file_path):
    """
    Open FITS with a memmap fallback for scaled images.
    Some files with BZERO/BSCALE/BLANK keywords cannot be memory-mapped
    and must be reopened with memmap=False.
    """
    try:
        with fits.open(file_path, memmap=True) as hdul:
            yield hdul, True
            return
    except Exception as first_exc:
        msg = str(first_exc)
        if (
            "BZERO/BSCALE/BLANK" in msg
            or "Cannot load a memory-mapped image" in msg
            or "Set memmap=False" in msg
        ):
            with fits.open(file_path, memmap=False) as hdul:
                yield hdul, False
                return
        raise first_exc


def fits_skip_reason(hdu):
    """Return a human-readable skip reason for unsupported HDUs."""
    try:
        extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
        if extname in NON_SCIENCE_EXTNAMES:
            return f"non-science extension ({extname})"
        naxis = hdu.header.get("NAXIS", None)
        if naxis is not None:
            try:
                naxis = int(naxis)
            except Exception:
                naxis = None
        if naxis is not None and naxis > 2:
            return f"unsupported dimensionality (NAXIS={naxis})"
    except Exception:
        pass
    return None

def try_extract_spectrum(hdu):
    """
    Returns: (wl_array, fl_array, labels_dict)
    labels_dict contains keys: x_label, y_label

    Only extracts true 1D spectra or table-based wavelength/flux pairs.
    2D image HDUs are intentionally NOT collapsed into spectra, because
    that leads to scientific mislabeling (e.g. DQ arrays, detector images,
    or generic imaging extensions being plotted as if they were spectra).
    """
    default_labels = {"x_label": "Wavelength", "y_label": "Flux"}

    # Skip known non-science HDU types before touching the data.
    try:
        extname = str(hdu.header.get("EXTNAME", "")).strip().upper()
        if extname in NON_SCIENCE_EXTNAMES:
            return None, None, default_labels
    except Exception:
        pass

    try:
        data = hdu.data
    except Exception:
        return None, None, default_labels

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

            # Fallback: first two numeric columns from a sufficiently large table.
            names = safe_names(data)
            nums = [n for n in names if np.issubdtype(data[n].dtype, np.number)]
            if len(nums) >= 2:
                n_rows = len(data)
                if n_rows < SUPPORTED_SCIENCE_TABLE_MIN_ROWS:
                    return None, None, default_labels
                wl = np.array(data[nums[0]]).astype(float).flatten()
                fl = np.array(data[nums[1]]).astype(float).flatten()
                mask = np.isfinite(wl) & np.isfinite(fl)
                if np.any(mask):
                    labels = {"x_label": nums[0], "y_label": nums[1]}
                    return wl[mask], fl[mask], labels
    except Exception:
        pass

    # Only 1D arrays are treated as spectra. 2D arrays are left for the
    # Images tab, because collapsing them here can turn detector images,
    # DQ planes, and calibration arrays into fake "spectra".
    try:
        arr = np.array(data)
        if arr.ndim == 1:
            wl = np.arange(arr.size)
            fl = arr.astype(float)
            mask = np.isfinite(fl)
            return wl[mask], fl[mask], {"x_label": "Index", "y_label": "Value"}
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


def _interp_nans(y):
    y = np.asarray(y, dtype=float)
    if y.size == 0:
        return y
    good = np.isfinite(y)
    if good.sum() == 0:
        return y
    if good.sum() == 1:
        return np.full_like(y, y[good][0], dtype=float)
    x = np.arange(y.size)
    return np.interp(x, x[good], y[good])


def detect_anomalies(wl, fl, params=None):
    """
    Lightweight astronomy-oriented anomaly detection.

    Returns a list of dicts with schema:
    type, wl, index, value, score, prominence, width, method
    """
    p = {
        "clip_sigma": 3.0,
        "clip_iters": 5,
        "continuum_window": 101,
        "continuum_poly": 3,
        "mad_sigma": 5.0,
        "prominence_sigma": 4.0,
        "min_peak_distance": 5,
        "use_specutils": False,
        "specutils_noise_factor": 3.0,
    }
    if params:
        p.update(params)

    wl = np.asarray(wl, dtype=float)
    fl = np.asarray(fl, dtype=float)

    finite = np.isfinite(wl) & np.isfinite(fl)
    if finite.sum() < 10:
        return []

    wl = wl[finite]
    fl = fl[finite]

    clipped = sigma_clip(
        fl,
        sigma=float(p["clip_sigma"]),
        maxiters=int(p["clip_iters"]),
        cenfunc="median",
        masked=True,
        copy=True,
    )
    clipped_flux = clipped.filled(np.nan) if hasattr(clipped, "filled") else np.asarray(clipped, dtype=float)
    clipped_flux = _interp_nans(clipped_flux)

    cont = smooth_flux(
        clipped_flux.copy(),
        window=int(p["continuum_window"]),
        polyorder=int(p["continuum_poly"]),
    )
    cont = np.asarray(cont, dtype=float)
    cont = _interp_nans(cont)

    residual = fl - cont
    residual = np.asarray(residual, dtype=float)
    residual[~np.isfinite(residual)] = np.nan

    finite_resid = residual[np.isfinite(residual)]
    noise = mad_std(finite_resid, ignore_nan=True)
    if not np.isfinite(noise) or noise <= 0:
        noise = np.nanstd(finite_resid)
    if not np.isfinite(noise) or noise <= 0:
        noise = 1.0

    anomalies = []

    outlier_idx = np.where(np.abs(residual) >= float(p["mad_sigma"]) * noise)[0]
    for i in outlier_idx:
        anomalies.append({
            "type": "sigma_outlier",
            "wl": float(wl[i]),
            "index": int(i),
            "value": float(residual[i]),
            "score": float(abs(residual[i]) / noise),
            "prominence": np.nan,
            "width": np.nan,
            "method": "sigma_clip+mad",
        })

    peak_idx, peak_props = find_peaks(
        residual,
        prominence=float(p["prominence_sigma"]) * noise,
        distance=max(1, int(p["min_peak_distance"])),
    )
    if len(peak_idx) > 0:
        widths, _, _, _ = peak_widths(residual, peak_idx, rel_height=0.5)
        for j, i in enumerate(peak_idx):
            anomalies.append({
                "type": "emission_peak",
                "wl": float(wl[i]),
                "index": int(i),
                "value": float(residual[i]),
                "score": float(abs(residual[i]) / noise),
                "prominence": float(peak_props["prominences"][j]),
                "width": float(widths[j]),
                "method": "find_peaks+mad",
            })

    dip_idx, dip_props = find_peaks(
        -residual,
        prominence=float(p["prominence_sigma"]) * noise,
        distance=max(1, int(p["min_peak_distance"])),
    )
    if len(dip_idx) > 0:
        widths, _, _, _ = peak_widths(-residual, dip_idx, rel_height=0.5)
        for j, i in enumerate(dip_idx):
            anomalies.append({
                "type": "absorption_dip",
                "wl": float(wl[i]),
                "index": int(i),
                "value": float(residual[i]),
                "score": float(abs(residual[i]) / noise),
                "prominence": float(dip_props["prominences"][j]),
                "width": float(widths[j]),
                "method": "find_peaks+mad",
            })

    if (
        p.get("use_specutils")
        and Spectrum1D is not None
        and u is not None
        and StdDevUncertainty is not None
        and find_lines_threshold is not None
    ):
        try:
            spec = Spectrum1D(
                spectral_axis=wl * u.AA,
                flux=residual * u.one,
                uncertainty=StdDevUncertainty(np.full_like(residual, noise) * u.one),
            )
            lines = find_lines_threshold(spec, noise_factor=float(p["specutils_noise_factor"]))
            for row in lines:
                try:
                    idx = int(row["line_center_index"])
                    if 0 <= idx < len(wl):
                        anomalies.append({
                            "type": str(row["line_type"]),
                            "wl": float(wl[idx]),
                            "index": idx,
                            "value": float(residual[idx]),
                            "score": float(abs(residual[idx]) / noise),
                            "prominence": np.nan,
                            "width": np.nan,
                            "method": "specutils.find_lines_threshold",
                        })
                except Exception:
                    pass
        except Exception:
            pass

    seen = set()
    cleaned = []
    for a in anomalies:
        key = (a.get("type"), a.get("index"))
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(a)

    cleaned.sort(key=lambda x: (x.get("wl", np.nan), x.get("index", -1)))
    return cleaned

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

# ==========================================================
# MAST ARCHIVE INTEGRATION
# ==========================================================
def _mast_query_object(target_name, mission=None, radius="0.05 deg", max_results=MAX_MAST_RESULTS):
    """Internal helper for MAST search."""
    try:
        obs = Observations.query_object(
            target_name,
            radius=radius,
            pagesize=max_results,
        )
    except TypeError:
        obs = Observations.query_object(
            target_name,
            radius=radius,
        )

    if mission and mission != "All" and obs is not None and len(obs) > 0:
        obs = obs[obs["obs_collection"] == mission]

    if obs is not None and len(obs) > max_results:
        obs = obs[:max_results]

    return obs


def mast_search_target(target_name, mission=None, radius="0.05 deg"):
    """
    Search MAST archive for a target with a timeout so the UI does not
    spin forever on slow or overloaded archive responses.
    """
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_mast_query_object, target_name, mission, radius)
        try:
            return future.result(timeout=MAST_TIMEOUT_S)
        except FuturesTimeoutError:
            st.warning(
                f"MAST search timed out after {MAST_TIMEOUT_S} seconds. "
                "Try a more specific target name, a smaller radius, or retry later."
            )
            return None
        except Exception as e:
            st.error(f"MAST search failed: {e}")
            return None

def mast_download_products(observation_row):
    """
    Download science FITS products for a selected observation.
    Limits downloads to MAX_PRODUCTS to prevent runaway bulk downloads
    (e.g. PS1 mosaics with 60+ tiles).
    """
    try:
        products = Observations.get_product_list(
            observation_row
        )

        try:
            products = Observations.filter_products(
                products,
                productType="SCIENCE",
                extension="fits"
            )
        except Exception:
            products = Observations.filter_products(
                products,
                productType="SCIENCE"
            )

        n_total = len(products)
        if n_total == 0:
            st.warning("No SCIENCE FITS products found for this observation.")
            return None

        if n_total > MAX_PRODUCTS:
            st.warning(
                f"Found **{n_total}** science products. Downloading the first "
                f"**{MAX_PRODUCTS}** to avoid memory overload. "
                f"Use the [MAST Portal](https://mast.stsci.edu) for bulk downloads."
            )
            products = products[:MAX_PRODUCTS]

        manifest = Observations.download_products(
            products,
            cache=True
        )

        return manifest

    except Exception as e:
        st.error(f"MAST download failed: {e}")
        return None

@st.cache_data(ttl=3600)
def mast_import_fits(file_path):
    """
    Import downloaded FITS into AstroFlow format.
    Skips files larger than MAX_FILE_MB to prevent memory crashes.
    Opens with a memmap fallback for scaled images.
    """
    imported_results = []

    # File size guard
    try:
        file_mb = os.path.getsize(file_path) / (1024 * 1024)
    except OSError:
        file_mb = 0

    if file_mb > MAX_FILE_MB:
        st.warning(
            f"Skipping **{os.path.basename(file_path)}** "
            f"({file_mb:.0f} MB > {MAX_FILE_MB} MB limit). "
            f"Process locally for large imaging files."
        )
        return imported_results

    try:
        with open_fits_best_effort(file_path) as (hdul, used_memmap):
            for idx, hdu in enumerate(hdul):
                # Avoid processing known non-science HDUs.
                if fits_skip_reason(hdu) and hdu.header.get("NAXIS", 0) and int(hdu.header.get("NAXIS", 0)) > 2:
                    continue

                wl, fl, labels = try_extract_spectrum(hdu)

                if wl is None:
                    continue

                imported_results.append({
                    "file": os.path.basename(file_path),
                    "path": file_path,
                    "hdu_index": idx,
                    "header": dict(hdu.header),
                    "wl": np.array(wl, dtype=float),
                    "fl": np.array(fl, dtype=float),
                    "err": None,
                    "x_label": labels.get("x_label", "Wavelength"),
                    "y_label": labels.get("y_label", "Flux"),
                })

        if len(imported_results) == 0:
            st.warning(
                f"**{os.path.basename(file_path)}** did not contain an extractable 1D spectrum. "
                "This file may be image-only, metadata-only, or contain unsupported HDU types."
            )

    except Exception as e:
        st.warning(
            f"Could not import **{os.path.basename(file_path)}**: {e}"
        )

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
    enable_downloads = st.checkbox("Enable downloads", value=True)
    smoothing_window = st.slider("Smoothing window (odd)", 5, 501, 51, step=2)
    polyorder = st.slider("SavGol polyorder", 1, 5, 3)

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

    st.caption(f"Downloads limited to {MAX_PRODUCTS} products · files >{MAX_FILE_MB} MB skipped")
    st.caption("MAST searches can be slow on broad targets. Narrow the target name when possible.")

    mast_search_btn = st.button(
        "Search MAST"
    )

  
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
st.markdown(
    """
    <style>
        .af-hero {
            padding: 1.2rem 1.25rem;
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 18px;
            background: linear-gradient(135deg, rgba(10,18,35,0.98), rgba(20,34,58,0.96));
            color: white;
            margin-bottom: 0.9rem;
        }
        .af-hero h1 {
            margin: 0;
            font-size: 2rem;
            line-height: 1.15;
        }
        .af-hero p {
            margin: 0.45rem 0 0 0;
            font-size: 0.98rem;
            opacity: 0.9;
        }
        .af-pill {
            display: inline-block;
            padding: 0.28rem 0.7rem;
            margin: 0.2rem 0.35rem 0.2rem 0;
            border-radius: 999px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
            font-size: 0.82rem;
        }
        .af-panel {
            padding: 0.95rem 1rem;
            border: 1px solid rgba(120,120,120,0.18);
            border-radius: 14px;
            background: rgba(250,250,252,0.88);
            margin-bottom: 0.9rem;
        }
        .af-panel h3 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="af-hero">
        <h1>🔭 AstroFlow</h1>
        <p>A lightweight FITS and CSV processor for spectra, images, MAST archive retrieval, and report generation.</p>
        <div style="margin-top:0.8rem;">
            <span class="af-pill">JWST</span>
            <span class="af-pill">HST</span>
            <span class="af-pill">TESS</span>
            <span class="af-pill">Generic FITS</span>
            <span class="af-pill">CSV Spectra</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

mast_results = st.session_state.get("mast_results")
mast_imported_results = st.session_state.get("mast_imported_results", [])

if not uploaded and mast_results is None and len(mast_imported_results) == 0:
    st.info("Upload FITS or CSV spectral files to start, or search the MAST archive from the sidebar.")
    st.stop()

st.markdown(
    """
    <div class="af-panel">
        <h3>Data import</h3>
        <p style="margin:0;">
            Supported: <b>FITS</b> spectra, <b>FITS</b> images, and <b>CSV</b> spectral tables.
            Unsupported: calibration-only tables, non-science extensions, and files beyond the memory limits of the hosted deployment.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Save uploaded files and process each (FITS or CSV) ---
tmpdir = tempfile.mkdtemp()
file_paths = []
uploaded_results = []
processing_summary = {
    "processed": [],
    "skipped": [],
    "warnings": [],
    "errors": [],
}
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
            msg = (
                f"**{fname}** is {file_mb:.0f} MB (>{MAX_FILE_MB} MB limit) — skipped. "
                f"Consider pre-processing large files locally."
            )
            st.warning(msg)
            processing_summary["skipped"].append({"file": fname, "reason": "file too large", "mb": round(file_mb, 1)})
            continue

        lower = fname.lower()

        # CSV handling
        if lower.endswith(".csv"):
            try:
                csv_outputs = ingest_csv_file(dst, filename=fname)
                if not csv_outputs:
                    processing_summary["warnings"].append({"file": fname, "reason": "CSV parsed but no usable spectra were found"})
                for out in csv_outputs:
                    if out.get("wl") is not None:
                        out["wl"] = np.asarray(out["wl"], dtype=float)
                    if out.get("fl") is not None:
                        out["fl"] = np.asarray(out["fl"], dtype=float)
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

                if csv_outputs:
                    processing_summary["processed"].append({"file": fname, "type": "CSV", "items": len(csv_outputs)})
            except Exception as e:
                st.error(f"Failed to parse CSV {fname}: {e}")
                processing_summary["errors"].append({"file": fname, "reason": str(e)})
            continue

        # FITS handling — open with fallback for scaled images
        try:
            with open_fits_best_effort(dst) as (hdul, used_memmap):
                found_any = False
                unsupported_reasons = set()
                n_supported_hdus = 0
                n_skipped_hdus = 0

                for idx, hdu in enumerate(hdul):
                    reason = fits_skip_reason(hdu)
                    if reason:
                        unsupported_reasons.add(reason)

                    wl, fl, labels = try_extract_spectrum(hdu)
                    if wl is None:
                        n_skipped_hdus += 1
                        continue

                    found_any = True
                    n_supported_hdus += 1
                    uploaded_results.append({
                        "file": fname,
                        "path": dst,
                        "hdu_index": idx,
                        "header": dict(hdu.header) if hasattr(hdu, "header") else {},
                        "wl": np.array(wl, dtype=float),
                        "fl": np.array(fl, dtype=float),
                        "err": None,
                        "x_label": labels.get("x_label", "Wavelength"),
                        "y_label": labels.get("y_label", "Flux"),
                    })

                if not found_any:
                    if unsupported_reasons:
                        msg = (
                            f"**{fname}** was skipped because it contains only unsupported HDU types: "
                            f"{', '.join(sorted(unsupported_reasons))}. "
                            f"AstroFlow in the hosted Streamlit build supports 1D spectra and 2D images only."
                        )
                        st.warning(msg)
                        processing_summary["skipped"].append({
                            "file": fname,
                            "reason": "unsupported HDU types",
                            "details": ", ".join(sorted(unsupported_reasons)),
                            "skipped_hdus": n_skipped_hdus,
                        })
                    else:
                        msg = (
                            f"**{fname}** did not contain an extractable 1D spectrum. "
                            "It may be an image-only file, a metadata-only product, or a format AstroFlow does not treat as a spectrum."
                        )
                        st.warning(msg)
                        processing_summary["warnings"].append({"file": fname, "reason": "no extractable 1D spectrum"})
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
                else:
                    processing_summary["processed"].append({
                        "file": fname,
                        "type": "FITS",
                        "supported_hdus": n_supported_hdus,
                        "skipped_hdus": n_skipped_hdus,
                        "memmap": bool(used_memmap),
                    })
        except Exception as e:
            st.warning(f"Could not open **{fname}**: {e}")
            processing_summary["errors"].append({"file": fname, "reason": str(e)})

    progress.progress(100, text="✅ All files processed.")
    if status_text:
        status_text.empty()

mast_imported_results = st.session_state.get("mast_imported_results", [])
results = uploaded_results + mast_imported_results

# Dashboard summary cards
num_spectra = sum(1 for r in results if r.get("wl") is not None and r.get("fl") is not None)
num_images = sum(1 for r in results if r.get("path"))
num_skipped = len(processing_summary["skipped"])
num_warnings = len(processing_summary["warnings"])
num_errors = len(processing_summary["errors"])

st.markdown("### Session summary")
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Files", len(uploaded) if uploaded else 0)
s2.metric("Spectra", num_spectra)
s3.metric("Images", num_images)
s4.metric("Skipped", num_skipped)
s5.metric("Warnings", num_warnings + num_errors)

with st.expander("Processing details", expanded=False):
    st.write("Processed items")
    st.json(processing_summary["processed"] if processing_summary["processed"] else [])
    st.write("Skipped items")
    st.json(processing_summary["skipped"] if processing_summary["skipped"] else [])
    if processing_summary["warnings"]:
        st.write("Warnings")
        st.json(processing_summary["warnings"])
    if processing_summary["errors"]:
        st.write("Errors")
        st.json(processing_summary["errors"])

st.sidebar.markdown("---")
with st.sidebar.expander("Session Summary", expanded=False):
    st.metric("Spectra", num_spectra)
    st.metric("Files", len(results))
    st.metric("Images", num_images)
    st.metric("Skipped HDUs", num_skipped)
    st.metric("Warnings", num_warnings)
    st.metric("Errors", num_errors)

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
    show_smooth = st.checkbox("Show smoothed version", value=True, key="spectrum_smooth")

    for res in results:
        if res.get("wl") is None or res.get("fl") is None:
            continue

        label = f"{res['file']} (HDU {res.get('hdu_index')})"
        with st.expander(label, expanded=False):
            wl = res['wl']
            fl = res['fl']
            x_label = res.get("x_label", "Wavelength")
            y_label = res.get("y_label", "Flux")

            fl_smooth = None
            if show_smooth:
                fl_smooth = smooth_flux(fl.copy(), smoothing_window, polyorder)

            # === IMPROVED UNIQUE KEY ===
            unique_key = make_key(
                res.get('file', ''), 
                res.get('hdu_index', ''), 
                res.get('path', ''),      # path makes it more unique
                hashlib.md5(str(res.get('wl', ''))[:50].encode()).hexdigest()[:6]  # extra safety
            )

            fig = plot_spectrum_interactive(
                wl, fl, 
                fl_smooth=fl_smooth,
                err=res.get('err'),
                title=label,
                bands=None,
                show_bands_flag=False,
                show_error=False,
                x_label=x_label,
                y_label=y_label
            )
            st.plotly_chart(fig, use_container_width=True, key=unique_key)

            # Downloads
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
                    key=make_key(res.get('file'), res.get('hdu_index'), 'dl')
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
    seen_img_combos = set()
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

        try:
            with open_fits_best_effort(file_path) as (hdul, used_memmap):
                for idx, hdu in enumerate(hdul):
                    if image_render_count >= MAX_IMAGES:
                        break

                    combo = (file_path, idx)
                    if combo in seen_img_combos:
                        continue

                    if hdu.data is not None and hasattr(hdu.data, "shape") and hdu.data.ndim == 2:
                        seen_img_combos.add(combo)
                        found_image = True
                        image_render_count += 1
                        st.subheader(f"{r['file']} (HDU {idx}) — Image ({hdu.data.shape[0]}×{hdu.data.shape[1]} px)")

                        fig, ax = plt.subplots(figsize=(7, 5), dpi=120)
                        plot_data = hdu.data.astype(float)

                        try:
                            import matplotlib.colors as mcolors
                            if img_lognorm:
                                # Clip negatives for LogNorm
                                vmin = np.nanpercentile(plot_data[plot_data > 0], 1) if np.any(plot_data > 0) else 1e-6
                                vmax = np.nanpercentile(plot_data, 99)
                                norm = mcolors.LogNorm(vmin=max(vmin, 1e-10), vmax=max(vmax, 1e-9))
                            else:
                                vmin = np.nanpercentile(plot_data, 1)
                                vmax = np.nanpercentile(plot_data, 99)
                                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                            im = ax.imshow(plot_data, cmap=img_cmap, origin="lower", aspect="auto", norm=norm)
                        except Exception:
                            im = ax.imshow(plot_data, cmap=img_cmap, origin="lower", aspect="auto")

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
    st.markdown("Compile spectra plots and FITS images into a single PDF with a cover page and summary pages.")

    rpt_col1, rpt_col2 = st.columns(2)
    with rpt_col1:
        rpt_target = st.text_input("Target / Object name", value=mast_target if mast_target else "Unknown Target")
        rpt_author = st.text_input("Author(s)", value="AstroFlow User")
    with rpt_col2:
        rpt_instrument = st.text_input("Instrument / Mission", value="")
        rpt_notes = st.text_area("Report notes (optional)", value="", height=68)

    st.markdown("#### Report contents")
    st.caption("The report will include a cover page, a summary page, a methods page, then the selected spectra and images.")

    if st.button("Generate Report"):
        tmp_pdf = os.path.join(tempfile.gettempdir(), f"astroflow_report_{int(time.time())}.pdf")
        plots = []
        images = []

        spec_results_for_report = [r for r in results if r.get("wl") is not None and r.get("fl") is not None]
        report_metrics = {
            "n_files": len(results),
            "n_spectra": len(spec_results_for_report),
            "n_images": sum(1 for r in results if r.get("path")),
            "n_skipped": len(processing_summary["skipped"]) if "processing_summary" in locals() else 0,
            "n_warnings": len(processing_summary["warnings"]) if "processing_summary" in locals() else 0,
            "n_errors": len(processing_summary["errors"]) if "processing_summary" in locals() else 0,
        }

        report_progress = st.progress(0, text="Building report…")
        n_report_steps = max(len(spec_results_for_report) + 3, 1)

        report_dir = tempfile.mkdtemp()

        def _save_text_page(path, title, subtitle_lines, body_lines):
            fig, ax = plt.subplots(figsize=(8.27, 11.69), dpi=200)
            ax.axis("off")

            y = 0.95
            ax.text(0.5, y, title, ha="center", va="top", fontsize=22, fontweight="bold", transform=ax.transAxes)
            y -= 0.08

            for line in subtitle_lines:
                ax.text(0.5, y, line, ha="center", va="top", fontsize=11, transform=ax.transAxes)
                y -= 0.035

            y -= 0.03
            ax.hlines(y, 0.12, 0.88, transform=ax.transAxes, linewidth=1.0)
            y -= 0.05

            for line in body_lines:
                ax.text(0.12, y, line, ha="left", va="top", fontsize=11, transform=ax.transAxes)
                y -= 0.045

            plt.savefig(path, bbox_inches="tight", facecolor="white")
            plt.close(fig)

        # Cover page
        cover_path = os.path.join(report_dir, "astroflow_cover.png")
        cover_lines = [
            f"Target: {rpt_target}",
            f"Author(s): {rpt_author}",
            f"Instrument / Mission: {rpt_instrument or 'Not specified'}",
            f"Generated: {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}",
            "",
            f"Files analysed: {report_metrics['n_files']}",
            f"Spectra extracted: {report_metrics['n_spectra']}",
            f"Images rendered: {report_metrics['n_images']}",
            f"Skipped items: {report_metrics['n_skipped']}",
            f"Warnings: {report_metrics['n_warnings']}",
            f"Errors: {report_metrics['n_errors']}",
            "",
            "Methods:",
            "• Sigma clipping",
            "• MAD-based outlier detection",
            "• Savitzky–Golay continuum estimation",
            "• Peak prominence detection",
            "• Optional Specutils line finding",
        ]
        _save_text_page(
            cover_path,
            "AstroFlow",
            ["Astronomical FITS Processing and Spectral Analysis Report"],
            cover_lines,
        )
        plots.append(cover_path)

        # Executive summary page
        report_progress.progress(10, text="Creating summary pages…")
        summary_path = os.path.join(report_dir, "astroflow_summary.png")
        summary_lines = [
            f"Processed files: {report_metrics['n_files']}",
            f"Extracted spectra: {report_metrics['n_spectra']}",
            f"Rendered images: {report_metrics['n_images']}",
            f"Skipped items: {report_metrics['n_skipped']}",
            f"Warnings: {report_metrics['n_warnings']}",
            f"Errors: {report_metrics['n_errors']}",
            "",
            "Notes:",
            (rpt_notes or "No additional notes provided.").strip() or "No additional notes provided.",
        ]
        _save_text_page(
            summary_path,
            "Executive Summary",
            ["Overview of the current analysis session"],
            summary_lines,
        )
        plots.append(summary_path)

        # Methods page
        methods_path = os.path.join(report_dir, "astroflow_methods.png")
        methods_lines = [
            "Spectrum handling:",
            "• 1D arrays and table-based wavelength/flux pairs are extracted as spectra.",
            "• Unsupported HDU types and non-science extensions are skipped.",
            "",
            "Analysis methods:",
            "• Sigma clipping removes gross outliers before continuum estimation.",
            "• Savitzky–Golay smoothing estimates the local continuum.",
            "• Residuals are analysed with MAD-based thresholds.",
            "• Peak prominence is used for emission and absorption feature detection.",
            "",
            "Report generation:",
            "• Spectra plots and FITS images are compiled into a single PDF.",
            "• Duplicate images are removed using file+HDU deduplication.",
        ]
        _save_text_page(
            methods_path,
            "Methods",
            ["AstroFlow processing and detection workflow"],
            methods_lines,
        )
        plots.append(methods_path)

        # Optional processing log page
        log_path = os.path.join(report_dir, "astroflow_log.png")
        log_lines = [
            f"Skipped items: {len(processing_summary['skipped'])}",
            f"Warnings: {len(processing_summary['warnings'])}",
            f"Errors: {len(processing_summary['errors'])}",
            "",
            "Processing log excerpt:",
        ]
        if processing_summary["skipped"]:
            for item in processing_summary["skipped"][:8]:
                log_lines.append(f"• {item.get('file', 'Unknown')}: {item.get('reason', 'Skipped')}")
        else:
            log_lines.append("• No skipped items.")
        _save_text_page(
            log_path,
            "Processing Log",
            ["Concise summary of skipped or notable items"],
            log_lines,
        )
        plots.append(log_path)

        # Save 1D spectra as clean PNGs
        for ri, res in enumerate(spec_results_for_report):
            report_progress.progress(
                int((ri + 3) / n_report_steps * 80),
                text=f"Plotting spectrum {ri + 1}/{len(spec_results_for_report)}…"
            )
            wl = res["wl"]
            fl = res["fl"]
            x_label = res.get("x_label", "Wavelength")
            y_label = res.get("y_label", "Flux")

            fl_smooth_rpt = smooth_flux(fl.copy(), smoothing_window, polyorder) if show_smooth else None

            buf = io.BytesIO()
            fig_rpt, ax_rpt = plt.subplots(figsize=(8, 4.5), dpi=200)

            ax_rpt.plot(wl, fl, color="steelblue", alpha=0.8, linewidth=1.2, label="Raw")
            if fl_smooth_rpt is not None:
                ax_rpt.plot(wl, fl_smooth_rpt, color="darkred", linewidth=2.0, label="Smoothed")

            ax_rpt.set_xlabel(x_label, fontsize=11)
            ax_rpt.set_ylabel(y_label, fontsize=11)
            ax_rpt.set_title(f"{res['file']} · HDU {res.get('hdu_index')}", fontsize=12)

            ax_rpt.margins(x=0.02, y=0.05)
            ax_rpt.autoscale(enable=True, axis="both", tight=False)
            ax_rpt.legend(fontsize=10, loc="best")
            ax_rpt.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(buf, format="png", dpi=200, bbox_inches="tight")
            plt.close(fig_rpt)
            buf.seek(0)

            img_path = os.path.join(report_dir, f"{res['file']}_hdu{res.get('hdu_index')}_spectrum.png")
            with open(img_path, "wb") as fh:
                fh.write(buf.read())
            plots.append(img_path)

        # Collect 2D FITS images for report — deduplicate by file+hdu combo
        img_count_rpt = 0
        seen_img_combos = set()
        for r in results:
            if img_count_rpt >= MAX_IMAGES:
                break
            if not r.get("path"):
                continue
            try:
                with open_fits_best_effort(r["path"]) as (hdul, used_memmap):
                    for idx, hdu in enumerate(hdul):
                        if img_count_rpt >= MAX_IMAGES:
                            break
                        combo = (r["path"], idx)
                        if combo in seen_img_combos:
                            continue
                        if hdu.data is not None and hasattr(hdu.data, "shape") and hdu.data.ndim == 2:
                            seen_img_combos.add(combo)
                            img_path = os.path.join(report_dir, f"{r['file']}_hdu{idx}_image.png")
                            plot_data = hdu.data.astype(float)
                            vmin = np.nanpercentile(plot_data, 1)
                            vmax = np.nanpercentile(plot_data, 99)
                            plt.imsave(img_path, np.clip(plot_data, vmin, vmax), cmap="gray", origin="lower")
                            images.append(img_path)
                            img_count_rpt += 1
            except Exception as e:
                st.warning(f"Could not read images from {r.get('file')}: {e}")

        report_progress.progress(90, text="Compiling PDF…")

        safe_title = f"AstroFlow Analysis Report - {rpt_target}".replace("\u2014", "-").replace("\u2013", "-")
        safe_notes = (rpt_notes or "").replace("\u2014", "-").replace("\u2013", "-").replace("\u2018", "'").replace("\u2019", "'")

        report_metadata = {
            "title": safe_title,
            "author": rpt_author,
            "target": rpt_target,
            "instrument": rpt_instrument,
            "notes": safe_notes,
            "generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "n_spectra": len(spec_results_for_report),
            "files": list({r["file"] for r in spec_results_for_report}),
            "n_files": report_metrics["n_files"],
            "n_images": report_metrics["n_images"],
            "n_skipped": report_metrics["n_skipped"],
            "n_warnings": report_metrics["n_warnings"],
            "n_errors": report_metrics["n_errors"],
        }

        try:
            pdf_path = generate_pdf_report(
                output_path=tmp_pdf,
                metadata=report_metadata,
                plots=plots,
                tables=[],
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
                    key=make_key("pdf_report", rpt_key_n),
                )
        else:
            st.error("PDF report was not generated.")


# ---------------------------
# Anomaly Detection tab
# ---------------------------
with tabs[5]:
    st.header("Anomaly Detection")
    st.markdown(
        "Robust spectral QC: sigma clipping, MAD-based outliers, continuum residuals, "
        "peak prominence, and optional Specutils line finding."
    )

    with st.sidebar.expander("Anomaly Detection Settings", expanded=False):
        clip_sigma = st.slider("Sigma-clip threshold (σ)", 2.0, 8.0, 3.0, 0.5)
        clip_iters = st.slider("Sigma-clip iterations", 1, 10, 5)

        continuum_window = st.slider("Continuum window (px)", 11, 501, 101, step=2)
        continuum_poly = st.slider("Continuum polynomial order", 1, 5, 3)

        mad_sigma = st.slider("MAD outlier threshold (× MAD σ)", 2.0, 10.0, 5.0, 0.5)
        prominence_sigma = st.slider("Peak prominence (× MAD σ)", 1.0, 12.0, 4.0, 0.5)
        min_peak_distance = st.slider("Minimum peak spacing (px)", 1, 50, 5)

        use_specutils = st.checkbox("Use Specutils line finder", value=False)
        specutils_noise_factor = st.slider("Specutils noise factor", 1.0, 10.0, 3.0, 0.5)

    anomalies_all = []
    expected_keys = ANOMALY_KEYS

    for res in results:
        if res.get("wl") is None or res.get("fl") is None:
            continue

        wl = res["wl"]
        fl = res["fl"]
        x_label = res.get("x_label", "Wavelength")
        y_label = res.get("y_label", "Flux")

        params = {
            "clip_sigma": clip_sigma,
            "clip_iters": clip_iters,
            "continuum_window": continuum_window,
            "continuum_poly": continuum_poly,
            "mad_sigma": mad_sigma,
            "prominence_sigma": prominence_sigma,
            "min_peak_distance": min_peak_distance,
            "use_specutils": use_specutils,
            "specutils_noise_factor": specutils_noise_factor,
        }

        anoms = detect_anomalies(wl, fl, params=params)

        for a in anoms:
            a["file"] = res.get("file")
            a["hdu_index"] = res.get("hdu_index")

        anomalies_all += anoms

        st.subheader(f"{res['file']} (HDU {res.get('hdu_index')})")

        fig = plot_spectrum_interactive(
            wl,
            fl,
            title=f"{res['file']} (HDU {res.get('hdu_index')})",
            x_label=x_label,
            y_label=y_label
        )
        fig = annotate_plotly(fig, anoms)
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=make_key(res['file'], res.get('hdu_index'), 'anomaly_plot')
        )

        normalized_anoms = [{k: a.get(k, np.nan) for k in expected_keys} for a in anoms]

        if normalized_anoms:
            df_anoms = pd.DataFrame(normalized_anoms)
            st.dataframe(df_anoms.head(200), use_container_width=True)

            if enable_downloads:
                import json
                dl_key_json = make_key(res['file'], res.get('hdu_index'), 'anoms_json')
                st.download_button(
                    f"Download anomalies JSON - {res['file']}",
                    json.dumps(anoms, indent=2).encode('utf-8'),
                    file_name=f"{res['file']}_hdu{res.get('hdu_index')}_anomalies.json",
                    mime="application/json",
                    key=dl_key_json
                )

                dl_key_csv = make_key(res['file'], res.get('hdu_index'), 'anoms_csv')
                st.download_button(
                    f"Download anomalies CSV - {res['file']}",
                    df_anoms.to_csv(index=False).encode('utf-8'),
                    file_name=f"{res['file']}_hdu{res.get('hdu_index')}_anomalies.csv",
                    mime="text/csv",
                    key=dl_key_csv
                )
        else:
            st.write("No anomalies detected for this spectrum.")

    st.markdown("### Summary")
    st.write(f"Total anomalies detected across all files: {len(anomalies_all)}")

    if anomalies_all and enable_downloads:
        normalized_all = [{k: a.get(k, np.nan) for k in expected_keys + ["file", "hdu_index"]} for a in anomalies_all]
        df_all = pd.DataFrame(normalized_all)
        dl_key_all = make_key('all', 'anomalies', 'csv')
        st.download_button(
            "Download all anomalies (CSV)",
            df_all.to_csv(index=False).encode('utf-8'),
            file_name="astroflow_anomalies_all.csv",
            mime='text/csv',
            key=dl_key_all
        )
