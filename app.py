# Copyright (c) 2025 <Ali Nawaz>
# All rights reserved.
# Licensed for non-commercial evaluation and demonstration only.
# No copying, redistribution, or commercial use without written permission.
# Contact: <alinawaz9519@gmail.com>


# app.py (updated with dynamic axis labels)
from FitsFlow.csv_handler import ingest_csv_file
from FitsFlow.detectors import detect_anomalies, annotate_plotly
from FitsFlow.fields import detect_data_type, map_columns
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

st.set_page_config(page_title="AstroFlow · FITSFlow", layout="wide", initial_sidebar_state="expanded")

# ---------------------------
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
    try:
        arr = np.array(data)
        if arr.ndim == 1:
            wl = np.arange(arr.size)
            fl = arr.astype(float)
            mask = np.isfinite(fl)
            return wl[mask], fl[mask], {"x_label": "Index", "y_label": "Value"}
        elif arr.ndim == 2:
            fl = np.nanmean(arr, axis=0)
            wl = np.arange(fl.size)
            mask = np.isfinite(fl)
            return wl[mask], fl[mask], {"x_label": "Pixel", "y_label": "Mean(pixel rows)"}
    except Exception:
        pass

    return None, None, default_labels

def interp_to_reference(wl, fl, ref_wl):
    try:
        return np.interp(ref_wl, wl, fl, left=np.nan, right=np.nan)
    except Exception:
        return np.full_like(ref_wl, np.nan)

def smooth_flux(flux, window, polyorder):
    if len(flux) >= window and window % 2 == 1:
        return savgol_filter(flux, window, polyorder)
    return flux

def calc_snr_on_band(ref_wl, ref_flux, band_range: Tuple[float,float]):
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

DEFAULT_BANDS = {
    "H2O": (1.35, 1.45),
    "CH4": (1.60, 1.72),
    "CO2": (2.65, 2.75),
}

# ---------------------------
# Sidebar controls (UI)
# ---------------------------
st.sidebar.header("AstroFlow Controls")
st.sidebar.markdown("Upload FITS/CSV files and toggle analysis options.")

smoothing_enabled = st.sidebar.checkbox("Enable smoothing", value=True)
smoothing_window = st.sidebar.slider("Smoothing window (odd)", 5, 501, 51, step=2)
polyorder = st.sidebar.slider("SavGol polyorder", 1, 5, 3)

show_bands = st.sidebar.checkbox("Show molecular bands (overlay)", value=True)
selected_bands = st.sidebar.multiselect(
    "Select molecular bands to display",
    options=list(DEFAULT_BANDS.keys()),
    default=list(DEFAULT_BANDS.keys())
)

show_snr = st.sidebar.checkbox("Show SNR", value=False)
show_errorbars = st.sidebar.checkbox("Show error bars (if available)", value=False)
raw_only = st.sidebar.checkbox("Show raw data only (no smoothing/stacking overlays)", value=False)

stack_enabled = st.sidebar.checkbox("Enable stacking (multi-file)", value=True)
stack_method = st.sidebar.selectbox("Stack method", ["mean", "median"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("Display / export options")
enable_downloads = st.sidebar.checkbox("Enable downloads", value=True)
st.sidebar.markdown("---")
st.sidebar.caption("Prototype · AstroFlow / FutureMind")

# ---------------------------
# Main UI area
# ---------------------------
st.title("🔭 AstroFlow · FITSFlow Processor")
st.markdown("Upload FITS or CSV files (JWST/HST/TESS/generic). Tabs: Raw | Smoothed | Molecule Detection | Stacked | Table")

uploaded = st.file_uploader(
    "Upload one or more FITS/CSV files",
    type=["fits", "csv"],
    accept_multiple_files=True
)

if not uploaded:
    st.info("Upload FITS or CSV spectral files to start. Example FITS: K2-18b, GJ-1214b.")
    st.stop()

# --- Save uploaded files and process each (FITS or CSV) ---
tmpdir = tempfile.mkdtemp()
file_paths = []
results = []
nfiles = len(uploaded)
progress = st.progress(0)

for i, up in enumerate(uploaded, start=1):
    progress.progress(int((i-1)/nfiles*100))
    fname = up.name
    dst = os.path.join(tmpdir, fname)
    with open(dst, "wb") as f:
        f.write(up.getvalue())
    file_paths.append(dst)

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
                results.append(out)
        except Exception as e:
            st.error(f"Failed to parse CSV {fname}: {e}")
        continue

    # FITS handling
    try:
        with fits.open(dst, memmap=False) as hdul:
            found_any = False
            for idx, hdu in enumerate(hdul):
                wl, fl, labels = try_extract_spectrum(hdu)
                if wl is None:
                    continue
                found_any = True
                err = None
                results.append({
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
                results.append({
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

progress.progress(100)
time.sleep(0.2)

if len(results) == 0:
    st.error("No spectra could be extracted from uploaded files. You may upload pre-processed wavelength+flux CSVs.")
    st.stop()

tabs = st.tabs([
    "Raw Spectrum",
    "Smoothed",
    "Molecule Detection",
    "Stacked",
    "Data Table",
    "Downloads",
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
        for mol,(a,b) in bands.items():
            fig.add_vrect(x0=a, x1=b, fillcolor="LightSkyBlue", opacity=0.25, layer="below", line_width=0, annotation_text=mol, annotation_position="top left")
    fig.update_layout(title=title, xaxis_title=x_label, yaxis_title=y_label, template="plotly_white", height=400)
    return fig

# Raw tab
with tabs[0]:
    st.header("Raw Spectrum")
    for res in results:
        if res.get("wl") is None or res.get("fl") is None:
            continue

        label = f"{res['file']} (HDU {res['hdu_index']})"
        with st.expander(label, expanded=False):
            st.subheader("Header (partial)")
            hdr = res['header']
            keys_to_show = {k: hdr[k] for k in list(hdr.keys())[:20]}
            st.json(keys_to_show)

            wl = res['wl']; fl = res['fl']; err = res.get('err')
            x_label = res.get("x_label", "Wavelength")
            y_label = res.get("y_label", "Flux")

            fig = plot_spectrum_interactive(wl, fl, fl_smooth=None, err=err, title=label, bands=None, show_bands_flag=False, x_label=x_label, y_label=y_label)
            chart_key = make_key(res['file'], res['hdu_index'], 'plot', 'raw')
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            st.write(f"Data points: {len(wl)} | {x_label} range: {wl.min():.3g} – {wl.max():.3g}")

            if enable_downloads:
                df = pd.DataFrame({x_label: wl, y_label: fl})
                dl_key = make_key(res['file'], res['hdu_index'], 'download', 'raw_csv')
                st.download_button(f"Download CSV (raw) - {res['file']}", df.to_csv(index=False).encode('utf-8'), file_name=f"{res['file']}_hdu{res['hdu_index']}_raw.csv", mime='text/csv', key=dl_key)

# Smoothed tab
with tabs[1]:
    st.header("Smoothed Spectra")
    for res in results:
        if res.get("wl") is None or res.get("fl") is None:
            continue
        label = f"{res['file']} (HDU {res['hdu_index']})"
        with st.expander(label, expanded=False):
            wl = res['wl']; fl = res['fl']; err = res.get('err')
            x_label = res.get("x_label", "Wavelength")
            y_label = res.get("y_label", "Flux")
            if raw_only:
                st.info("Raw-only mode enabled. Toggle off to see smoothing.")
                fl_smooth = None
            else:
                fl_proc = fl.copy()
                fl_smooth = smooth_flux(fl_proc, smoothing_window, polyorder) if smoothing_enabled else None
            fig = plot_spectrum_interactive(wl, fl, fl_smooth=fl_smooth, err=err, title=label, bands=None, show_bands_flag=False, show_error=show_errorbars, x_label=x_label, y_label=y_label)
            chart_key = make_key(res['file'], res['hdu_index'], 'plot', 'smooth')
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            if enable_downloads:
                df = pd.DataFrame({x_label: wl, y_label: fl, f"{y_label}_smoothed": fl_smooth if fl_smooth is not None else fl})
                dl_key = make_key(res['file'], res['hdu_index'], 'download', 'smooth_csv')
                st.download_button(f"Download CSV (smoothed) - {res['file']}", df.to_csv(index=False).encode('utf-8'), file_name=f"{res['file']}_hdu{res['hdu_index']}_smoothed.csv", mime='text/csv', key=dl_key)

# Molecule Detection tab
with tabs[2]:
    st.header("Molecule Detection (band overlays)")
    active_bands = {mol: DEFAULT_BANDS[mol] for mol in selected_bands} if show_bands else {}

    for res in results:
        if res.get("wl") is None or res.get("fl") is None:
            continue
        label = f"{res['file']} (HDU {res['hdu_index']})"
        with st.expander(label, expanded=False):
            wl = res['wl']; fl = res['fl']
            x_label = res.get("x_label", "Wavelength")
            y_label = res.get("y_label", "Flux")
            if raw_only:
                fl_proc = fl
            else:
                fl_proc = smooth_flux(fl, smoothing_window, polyorder) if smoothing_enabled else fl
            fig = plot_spectrum_interactive(wl, fl, fl_smooth=fl_proc, err=res.get('err'), title=label, bands=active_bands, show_bands_flag=show_bands and not raw_only, show_error=show_errorbars, x_label=x_label, y_label=y_label)
            chart_key = make_key(res['file'], res['hdu_index'], 'plot', 'mol')
            st.plotly_chart(fig, use_container_width=True, key=chart_key)
            if show_snr and active_bands:
                snr_table = {mol: calc_snr_on_band(wl, fl_proc, rng) for mol,rng in active_bands.items()}
                st.subheader("SNR (approx)")
                st.json({k: float(np.round(v,3)) for k,v in snr_table.items()})
            if enable_downloads:
                df = pd.DataFrame({x_label: wl, y_label: fl, f"{y_label}_processed": fl_proc})
                dl_key = make_key(res['file'], res['hdu_index'], 'download', 'mol_csv')
                st.download_button(f"Download CSV (processed) - {res['file']}", df.to_csv(index=False).encode('utf-8'), file_name=f"{res['file']}_hdu{res['hdu_index']}_processed.csv", mime='text/csv', key=dl_key)

# Stacked tab
with tabs[3]:
    st.header("Stacked Spectrum")
    spec_results = [r for r in results if r.get("wl") is not None and r.get("fl") is not None]
    if len(spec_results) < 2 or not stack_enabled:
        st.info("Upload multiple spectra and enable stacking to see combined results.")
    else:
        # pick labels from first spectrum (best-effort)
        x_label = spec_results[0].get("x_label", "Wavelength")
        y_label = spec_results[0].get("y_label", "Flux")
        min_wl = min(np.nanmin(r['wl']) for r in spec_results)
        max_wl = max(np.nanmax(r['wl']) for r in spec_results)
        ref_wl = np.linspace(min_wl, max_wl, 2000)
        interp_fluxes = [interp_to_reference(r['wl'], r['fl'], ref_wl) for r in spec_results]
        arr = np.array(interp_fluxes)
        stacked = np.nanmedian(arr, axis=0) if stack_method == "median" else np.nanmean(arr, axis=0)
        if smoothing_enabled and not raw_only and len(stacked) >= smoothing_window:
            stacked_smooth = smooth_flux(stacked, smoothing_window, polyorder)
        else:
            stacked_smooth = stacked
        if True and not raw_only:
            if np.nanmax(stacked_smooth) != np.nanmin(stacked_smooth):
                stacked_norm = (stacked - np.nanmin(stacked)) / (np.nanmax(stacked) - np.nanmin(stacked))
                stacked_smooth = (stacked_smooth - np.nanmin(stacked_smooth)) / (np.nanmax(stacked_smooth) - np.nanmin(stacked_smooth)) if np.nanmax(stacked_smooth) != np.nanmin(stacked_smooth) else stacked_smooth
            else:
                stacked_norm = stacked
        else:
            stacked_norm = stacked

        bands_for_plot = {}
        if "H2O" in selected_bands: bands_for_plot["H2O"] = DEFAULT_BANDS["H2O"]
        if "CH4" in selected_bands: bands_for_plot["CH4"] = DEFAULT_BANDS["CH4"]
        if "CO2" in selected_bands: bands_for_plot["CO2"] = DEFAULT_BANDS["CO2"]

        fig_st = plot_spectrum_interactive(ref_wl, np.nan_to_num(stacked_norm), fl_smooth=stacked_smooth, err=None, title="Stacked Spectrum", bands=bands_for_plot, show_bands_flag=show_bands and not raw_only, show_error=False, x_label=x_label, y_label=y_label)
        st.plotly_chart(fig_st, use_container_width=True, key=make_key('stacked','plot'))
        if show_snr and bands_for_plot:
            st.subheader("Stacked SNR (approx)")
            st.json({mol: float(np.round(calc_snr_on_band(ref_wl, stacked_smooth, rng),4)) for mol,rng in bands_for_plot.items()})
        if enable_downloads:
            df_stack = pd.DataFrame({x_label: ref_wl, y_label: stacked_norm, f"{y_label}_smoothed": stacked_smooth})
            dl_key = make_key('stacked','download','csv','stacked_tab')
            st.download_button("Download stacked CSV", df_stack.to_csv(index=False).encode('utf-8'), file_name="stacked_spectrum.csv", mime='text/csv', key=dl_key)

# Data Table tab
with tabs[4]:
    st.header("Data Table")
    for r in results:
        label = f"{r['file']} (HDU {r.get('hdu_index')})"
        st.subheader(label)
        if r.get("wl") is not None and r.get("fl") is not None:
            df = pd.DataFrame({r.get("x_label","Wavelength"): r['wl'], r.get("y_label","Flux"): r['fl']})
        elif r.get("orig_df") is not None:
            df = r.get("orig_df")
        else:
            st.write("No 1D data for this file.")
            continue
        st.dataframe(df.head(500), use_container_width=True)
        if enable_downloads:
            dl_key = make_key(r.get('file'), r.get('hdu_index'), 'download', 'table_csv')
            st.download_button(f"Download CSV: {label}", df.to_csv(index=False).encode('utf-8'), file_name=f"{label}.csv", mime='text/csv', key=dl_key)

# Downloads tab
with tabs[5]:
    st.header("Downloads & Export")
    if enable_downloads:
        for r in results:
            if r.get("wl") is None or r.get("fl") is None:
                continue
            label = f"{r['file']}_hdu{r['hdu_index']}"
            df = pd.DataFrame({r.get("x_label","Wavelength"): r['wl'], r.get("y_label","Flux"): r['fl']})
            dl_key = make_key(label, 'download', 'csv')
            st.download_button(f"CSV: {label}", df.to_csv(index=False).encode('utf-8'), file_name=f"{label}.csv", mime='text/csv', key=dl_key)

        spec_results = [r for r in results if r.get("wl") is not None and r.get("fl") is not None]
        if len(spec_results) >= 2 and stack_enabled:
            min_wl = min(np.nanmin(r['wl']) for r in spec_results)
            max_wl = max(np.nanmax(r['wl']) for r in spec_results)
            ref_wl = np.linspace(min_wl, max_wl, 2000)
            interp_fluxes = [interp_to_reference(r['wl'], r['fl'], ref_wl) for r in spec_results]
            arr = np.array(interp_fluxes)
            stacked = np.nanmedian(arr, axis=0) if stack_method=="median" else np.nanmean(arr, axis=0)
            if np.nanmax(stacked) != np.nanmin(stacked):
                stacked = (stacked - np.nanmin(stacked)) / (np.nanmax(stacked) - np.nanmin(stacked))
            df_stack = pd.DataFrame({spec_results[0].get("x_label","Wavelength"): ref_wl, "stacked": stacked})
            dl_key = make_key('stacked','download','csv','downloads_tab')
            st.download_button("Download stacked CSV", df_stack.to_csv(index=False).encode('utf-8'), file_name="stacked_spectrum.csv", mime='text/csv', key=dl_key)
    else:
        st.info("Enable downloads in the sidebar to see export options.")

st.sidebar.success("Ready. Use the tabs to explore raw and processed data.")
st.caption("AstroFlow · FITSFlow MVP — upload data, toggle options, export results.")

# Images tab
with tabs[6]:
    st.header("FITS Images")
    found_image = False
    for r in results:
        if not r.get("path"):
            continue
        try:
            with fits.open(r["path"], memmap=False) as hdul:
                for idx, hdu in enumerate(hdul):
                    if hdu.data is not None and hasattr(hdu.data, "shape") and hdu.data.ndim == 2:
                        found_image = True
                        st.subheader(f"{r['file']} (HDU {idx}) — Image")
                        fig, ax = plt.subplots()
                        im = ax.imshow(hdu.data, cmap="gray", origin="lower", aspect="auto")
                        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                        st.pyplot(fig)
                        if enable_downloads:
                            buf = io.BytesIO()
                            fig.savefig(buf, format="png")
                            buf.seek(0)
                            dl_key = make_key(r['file'], idx, 'image_download', time.time())
                            st.download_button(label=f"Download Image (PNG) — {r['file']} HDU {idx}", data=buf, file_name=f"{r['file']}_hdu{idx}_image.png", mime="image/png", key=dl_key)
                        plt.close(fig)
        except Exception as e:
            st.warning(f"Could not open {r.get('file')} for images: {e}")
    if not found_image:
        st.info("No 2D images found in uploaded FITS files.")

# Reports tab
with tabs[7]:
    st.header("Generate PDF Report")
    st.markdown("Compile spectra, images, and tables into a single PDF.")

    if st.button("Generate Report"):
        tmp_pdf = os.path.join(tempfile.gettempdir(), f"astroflow_report_{int(time.time())}.pdf")
        plots = []
        images = []
        tables = []

        # Save 1D spectra as PNGs using Matplotlib
        for res in results:
            if res.get("wl") is None or res.get("fl") is None:
                continue
            wl, fl = res["wl"], res["fl"]
            x_label = res.get("x_label", "Wavelength")
            y_label = res.get("y_label", "Flux")
            buf = io.BytesIO()
            plt.figure(figsize=(6,4))
            plt.plot(wl, fl, color='blue')
            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f"{res['file']} HDU {res.get('hdu_index')}")
            plt.tight_layout()
            plt.savefig(buf, format="png")
            plt.close()
            buf.seek(0)
            img_path = os.path.join(tempfile.gettempdir(), f"{res['file']}_hdu{res.get('hdu_index')}_spectrum.png")
            with open(img_path, "wb") as fh:
                fh.write(buf.read())
            plots.append(img_path)

            # Save CSV for each
            df = pd.DataFrame({x_label: wl, y_label: fl})
            csv_path = os.path.join(tempfile.gettempdir(), f"{res['file']}_hdu{res.get('hdu_index')}.csv")
            df.to_csv(csv_path, index=False)
            tables.append(csv_path)

        # Collect 2D FITS images
        for r in results:
            if not r.get("path"):
                continue
            try:
                with fits.open(r["path"], memmap=False) as hdul:
                    for idx, hdu in enumerate(hdul):
                        if hdu.data is not None and hasattr(hdu.data, "shape") and hdu.data.ndim == 2:
                            img_path = os.path.join(tempfile.gettempdir(), f"{r['file']}_hdu{idx}_image.png")
                            plt.imsave(img_path, hdu.data, cmap="gray", origin="lower")
                            images.append(img_path)
            except Exception as e:
                st.warning(f"Could not read images from {r.get('file')}: {e}")

        # Generate PDF using reporters module
        try:
            pdf_path = generate_pdf_report(
                output_path=tmp_pdf,
                metadata={"title": "AstroFlow Report", "author": "AstroFlow"},
                plots=plots,
                tables=tables,
                images=images,
            )
        except Exception as e:
            st.error(f"Failed to generate PDF report: {e}")
            st.stop()

        if os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                st.download_button(label="Download PDF Report", data=f, file_name=os.path.basename(pdf_path), mime="application/pdf", key=make_key('pdf_report', int(time.time())))
        else:
            st.error("PDF report was not generated.")

# ---------------------------
# Anomalies tab
# ---------------------------
with tabs[8]:
    st.header("Anomaly Detection")
    st.markdown("Lightweight detectors: z-score outliers, local dips, spikes. Tune thresholds in the sidebar.")

    # Sidebar controls for anomaly detection
    st.sidebar.markdown("Anomaly detection settings")
    z_thresh = st.sidebar.slider("Outlier z-threshold", 3, 10, 4)
    dip_window = st.sidebar.slider("Dip median window (px)", 11, 501, 101, step=2)
    dip_depth = st.sidebar.number_input("Dip minimum depth fraction", min_value=0.0001, max_value=1.0, value=0.01, step=0.001)
    spike_window = st.sidebar.slider("Spike window", 3, 101, 11, step=2)
    spike_std = st.sidebar.slider("Spike std-factor", 2, 20, 6)

    anomalies_all = []

    for res in results:
        if res.get("wl") is None or res.get("fl") is None:
            continue

        wl = res["wl"]
        fl = res["fl"]
        x_label = res.get("x_label", "Wavelength")
        y_label = res.get("y_label", "Flux")

        params = {"z_thresh": z_thresh, "dip_window": dip_window, "dip_depth": dip_depth,
                  "spike_window": spike_window, "spike_std": spike_std}
        anoms = detect_anomalies(wl, fl, params=params)

        # Add file info to each anomaly
        for a in anoms:
            a["file"] = res.get("file")
            a["hdu_index"] = res.get("hdu_index")

        anomalies_all += anoms

        st.subheader(f"{res['file']} (HDU {res.get('hdu_index')})")

        # Plot spectrum with anomalies
        fig = plot_spectrum_interactive(wl, fl, title=f"{res['file']} (HDU {res.get('hdu_index')})",
                                        x_label=x_label, y_label=y_label)
        fig = annotate_plotly(fig, anoms)
        st.plotly_chart(fig, use_container_width=True, key=make_key(res['file'], res.get('hdu_index'), 'anomaly_plot'))

        # Normalize anomalies: ensure all expected keys exist
        expected_keys = ["type", "wl", "index", "value"]
        normalized_anoms = [{k: a.get(k, np.nan) for k in expected_keys} for a in anoms]

        if normalized_anoms:
            df_anoms = pd.DataFrame(normalized_anoms)
            st.table(df_anoms.head(200))

            if enable_downloads:
                # JSON download
                import json
                dl_key_json = make_key(res['file'], res.get('hdu_index'), 'anoms_json')
                st.download_button(
                    f"Download anomalies JSON - {res['file']}",
                    json.dumps(anoms, indent=2).encode('utf-8'),
                    file_name=f"{res['file']}_hdu{res.get('hdu_index')}_anomalies.json",
                    mime="application/json",
                    key=dl_key_json
                )

                # CSV download
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

    # Summary of all anomalies
    st.markdown("### Summary")
    st.write(f"Total anomalies detected across all files: {len(anomalies_all)}")

    if anomalies_all and enable_downloads:
        # Normalize all anomalies across all files
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
