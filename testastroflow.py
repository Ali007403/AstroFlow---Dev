"""
Tests for AstroFlow core logic.

These tests import the real functions from app.py while stubbing the
Streamlit UI layer and the optional FitsFlow package imports so the module
can be loaded without launching the app or touching the network.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class _ProgressStub:
    def progress(self, *args, **kwargs):
        return self


class _SidebarStub:
    def header(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def expander(self, *args, **kwargs):
        return _NullContext()

    def checkbox(self, *args, **kwargs):
        return kwargs.get("value", False)

    def slider(self, *args, **kwargs):
        if "value" in kwargs:
            return kwargs["value"]
        if len(args) >= 4:
            return args[3]
        return None

    def text_input(self, *args, **kwargs):
        return kwargs.get("value", "")

    def selectbox(self, label, options, index=0, **kwargs):
        if not options:
            return None
        return options[index]

    def button(self, *args, **kwargs):
        return False

    def metric(self, *args, **kwargs):
        return None


class StreamlitStub(types.ModuleType):
    def __init__(self):
        super().__init__("streamlit")
        self.session_state = {}
        self.sidebar = _SidebarStub()

    def set_page_config(self, *args, **kwargs):
        return None

    def cache_data(self, *args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def stop(self):
        raise SystemExit

    def file_uploader(self, *args, **kwargs):
        return []

    def progress(self, *args, **kwargs):
        return _ProgressStub()

    def empty(self):
        return _NullContext()

    def expander(self, *args, **kwargs):
        return _NullContext()

    def columns(self, n):
        return [_NullContext() for _ in range(n)]

    def tabs(self, labels):
        return [_NullContext() for _ in labels]

    def dataframe(self, *args, **kwargs):
        return None

    def plotly_chart(self, *args, **kwargs):
        return None

    def pyplot(self, *args, **kwargs):
        return None

    def download_button(self, *args, **kwargs):
        return None

    def rerun(self):
        return None

    def title(self, *args, **kwargs):
        return None

    def header(self, *args, **kwargs):
        return None

    def subheader(self, *args, **kwargs):
        return None

    def markdown(self, *args, **kwargs):
        return None

    def info(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def success(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def write(self, *args, **kwargs):
        return None


def _install_stub_modules():
    """Install minimal modules required so app.py can be imported."""
    # streamlit
    sys.modules["streamlit"] = StreamlitStub()

    # FitsFlow package stubs
    fitsflow = types.ModuleType("FitsFlow")
    fitsflow.__path__ = []
    sys.modules["FitsFlow"] = fitsflow

    csv_handler = types.ModuleType("FitsFlow.csv_handler")
    csv_handler.ingest_csv_file = lambda *args, **kwargs: []
    sys.modules["FitsFlow.csv_handler"] = csv_handler

    detectors = types.ModuleType("FitsFlow.detectors")
    detectors.annotate_plotly = lambda fig, *args, **kwargs: fig
    sys.modules["FitsFlow.detectors"] = detectors

    def map_columns(df):
        cols = [str(c).upper() for c in getattr(df, "columns", [])]
        mapping = {}
        for candidate in ("WAVELENGTH", "WAVE", "LAMBDA", "WLEN", "TIME", "X"):
            if candidate in cols:
                mapping["wavelength" if candidate != "TIME" else "time"] = candidate
                break
        for candidate in ("FLUX", "FLUX_DENSITY", "SPECTRUM", "INTENSITY", "VALUE", "SAP_FLUX", "Y"):
            if candidate in cols:
                mapping["flux" if candidate != "VALUE" else "value"] = candidate
                break
        return mapping

    fields = types.ModuleType("FitsFlow.fields")
    fields.map_columns = map_columns
    sys.modules["FitsFlow.fields"] = fields

    reporters = types.ModuleType("FitsFlow.reporters")
    reporters.generate_pdf_report = lambda *args, **kwargs: None
    sys.modules["FitsFlow.reporters"] = reporters

    # astroquery.mast
    astroquery = types.ModuleType("astroquery")
    astroquery.__path__ = []
    sys.modules["astroquery"] = astroquery

    mast = types.ModuleType("astroquery.mast")
    mast.Observations = MagicMock()
    sys.modules["astroquery.mast"] = mast


@pytest.fixture(scope="module")
def app():
    _install_stub_modules()
    spec = importlib.util.spec_from_file_location("app", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    try:
        assert spec.loader is not None
        spec.loader.exec_module(module)
    except SystemExit:
        # Expected: app.py calls st.stop() once the stubbed UI has no files.
        pass
    return module


def test_make_key_is_deterministic(app):
    a = app.make_key("file.fits", 1, "/tmp/x")
    b = app.make_key("file.fits", 1, "/tmp/x")
    c = app.make_key("file.fits", 2, "/tmp/x")
    assert a == b
    assert a != c


def test_fits_skip_reason_blocks_non_science_extensions(app):
    hdu = fits.ImageHDU(np.ones((2, 2)))
    hdu.header["EXTNAME"] = "DQ"
    assert "non-science extension" in app.fits_skip_reason(hdu)


def test_fits_skip_reason_blocks_high_dimensional_data(app):
    hdu = fits.ImageHDU(np.ones((2, 2, 2)))
    assert "unsupported dimensionality" in app.fits_skip_reason(hdu)


def test_try_extract_spectrum_accepts_1d_arrays(app):
    hdu = fits.PrimaryHDU(np.array([1.0, 2.0, 3.0]))
    wl, fl, labels = app.try_extract_spectrum(hdu)
    assert np.array_equal(wl, np.array([0, 1, 2]))
    assert np.allclose(fl, np.array([1.0, 2.0, 3.0]))
    assert labels["x_label"] == "Index"
    assert labels["y_label"] == "Value"


def test_try_extract_spectrum_blocks_non_science_hdus(app):
    hdu = fits.ImageHDU(np.array([1.0, 2.0, 3.0]))
    hdu.header["EXTNAME"] = "ERR"
    wl, fl, labels = app.try_extract_spectrum(hdu)
    assert wl is None and fl is None
    assert labels["x_label"] == "Wavelength"
    assert labels["y_label"] == "Flux"


def test_try_extract_spectrum_reads_table_columns(app):
    table = Table({"WAVELENGTH": [1.0, 2.0, 3.0], "FLUX": [10.0, 11.0, 12.0]})
    hdu = fits.BinTableHDU(table)
    wl, fl, labels = app.try_extract_spectrum(hdu)
    assert np.allclose(wl, np.array([1.0, 2.0, 3.0]))
    assert np.allclose(fl, np.array([10.0, 11.0, 12.0]))
    assert labels["x_label"].upper() == "WAVELENGTH"
    assert labels["y_label"].upper() == "FLUX"


def test_build_stacked_spectrum_mean_and_median(app):
    wl = np.linspace(1.0, 2.0, 100)
    spectra = [
        {"wl": wl, "fl": np.ones_like(wl)},
        {"wl": wl, "fl": np.ones_like(wl) * 3},
    ]
    ref_wl, mean_stack = app.build_stacked_spectrum(spectra, method="mean")
    _, median_stack = app.build_stacked_spectrum(spectra, method="median")
    assert len(ref_wl) == 2000
    assert np.nanmean(mean_stack) == pytest.approx(2.0, rel=1e-3)
    assert np.nanmean(median_stack) == pytest.approx(2.0, rel=1e-3)


def test_smooth_flux_handles_nans(app):
    fl = np.ones(100, dtype=float)
    fl[20:30] = np.nan
    smoothed = app.smooth_flux(fl, window=11, polyorder=3)
    assert len(smoothed) == len(fl)
    assert np.isfinite(smoothed).sum() > 80


def test_detect_anomalies_finds_spike(app):
    rng = np.random.default_rng(42)
    wl = np.linspace(1.0, 2.0, 400)
    fl = np.ones_like(wl) + rng.normal(0, 0.02, len(wl))
    fl[200] += 1.0
    anomalies = app.detect_anomalies(wl, fl, params={"mad_sigma": 4.0, "prominence_sigma": 4.0})
    assert isinstance(anomalies, list)
    assert any(abs(a["index"] - 200) <= 2 for a in anomalies)


def test_calc_snr_on_band_returns_positive_value(app):
    ref_wl = np.linspace(1.0, 3.0, 500)
    ref_flux = np.ones_like(ref_wl)
    ref_flux[(ref_wl > 1.4) & (ref_wl < 1.6)] += 0.2
    snr = app.calc_snr_on_band(ref_wl, ref_flux, (1.4, 1.6))
    assert snr >= 0


def test_open_fits_best_effort_opens_real_fits(tmp_path, app):
    path = tmp_path / "simple.fits"
    fits.PrimaryHDU(np.arange(16).reshape(4, 4)).writeto(path)

    with app.open_fits_best_effort(str(path)) as (hdul, used_memmap):
        assert len(hdul) == 1
        assert hdul[0].data.shape == (4, 4)
        assert isinstance(used_memmap, bool)
