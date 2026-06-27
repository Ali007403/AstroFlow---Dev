title: 'AstroFlow: An interactive pipeline for astronomical FITS and CSV spectral analysis'
tags:
  - Python
  - astronomy
  - spectroscopy
  - FITS
  - Streamlit
  - MAST
authors:
  - name: Ali Nawaz
    orcid: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: Independent Researcher, AstroFlow Project
    index: 1
date: 2025
bibliography: paper.bib
---

# Summary

AstroFlow is an open-source Streamlit application for loading, inspecting, and
reporting on heterogeneous astronomical data in FITS and CSV formats. It
provides a six-tab interactive interface covering spectral extraction, FITS
image display, MAST archive search and download, Savitzky--Golay smoothing,
robust anomaly detection, and automated PDF report generation. AstroFlow is
designed for exploratory analysis of mixed data products from missions such as
JWST, HST, and TESS, while remaining lightweight enough to run on a hosted
Streamlit Cloud deployment. The software is intended for researchers and
students who need a practical, reproducible workflow for quickly inspecting
heterogeneous astronomical data without writing custom reduction scripts.

# Statement of Need

Astronomical analysis often begins with a practical file-handling problem rather
than a novel algorithm: data arrive in many formats, with many extension types,
and with different conventions for wavelength, flux, quality flags, images, and
metadata. Researchers working across multiple missions frequently need to inspect
files from JWST, HST, TESS, or public archives such as MAST, then move between
separate tools for plotting, filtering, anomaly inspection, and reporting.

AstroFlow addresses this workflow gap by providing a single interface for common
inspection tasks without requiring the user to write any Python. It is designed
for users who need to:

- load heterogeneous FITS and CSV files without writing code,
- inspect extracted spectra and 2D image HDUs,
- search and import public archive products directly from MAST [@mast],
- generate reproducible PDF reports from a single session, and
- download processed outputs in CSV, PNG, and PDF formats.

The target audience is researchers, students, and scientific developers who
routinely handle mixed astronomical data products and need a fast interactive
inspection tool. AstroFlow is not intended to replace mission-specific reduction
pipelines; it fills the gap between raw archive data and more specialised
downstream analysis.

# State of the Field

AstroFlow builds on well-established scientific Python tools but differs in how
it combines them into a single interactive workflow. Astropy [@astropy2022]
provides FITS I/O and robust statistical utilities; SciPy [@scipy2020] provides
smoothing and peak detection; Streamlit provides the web interface; and
astroquery [@astroquery2019] provides the MAST archive connection. These tools
are widely used individually, but they do not, by themselves, constitute a
complete end-to-end application for archive search, multi-file inspection, and
PDF reporting.

`specutils` [@specutils] offers low-level spectral analysis primitives for
Python but requires users to write scripts and does not provide a graphical
multi-file workflow or PDF reporting. CCDProc [@ccdproc] focuses on CCD image
reduction rather than spectral inspection. No existing open-source tool
currently combines interactive multi-instrument FITS ingestion, MAST archive
integration, robust anomaly detection, and automated report generation in a
single zero-code interface.

AstroFlow's contribution is an integrated workflow that makes heterogeneous
astronomical data easier to inspect and document, drawing on the analysis
concepts common to `specutils` and `astropy` but wrapping them in an accessible
application layer.

# Software Design

AstroFlow is organised as a modular Python package (`FitsFlow`) with a
Streamlit front end (`app.py`). The six interface tabs are: MAST Archive,
Spectrum, Data Table, Images, Reports, and Anomalies.

## Data Ingestion and Classification

Uploaded files are written to a temporary local path and opened with a
safety-first FITS loader (`open_fits_best_effort`) that attempts a
memory-mapped open first and falls back to `memmap=False` when FITS scaling
keywords (BZERO, BSCALE, BLANK) prevent memory mapping. HDUs are classified
before any data is loaded by checking the `EXTNAME` header keyword against a
blocklist of known non-science extension types. Quality arrays (DQ, GROUPDQ,
PIXELDQ), error arrays (ERR, VAR\_POISSON, VAR\_RNOISE, VAR\_FLAT), weight
maps (WHT, CTX), and calibration tables (WCSCORR, HDRTAB, D2IMARR) are skipped
so that they are never plotted as science spectra. This prevents the common
failure mode in heterogeneous FITS collections where bitmask or calibration
arrays are accidentally rendered as line plots.

For spectra, AstroFlow prefers explicit wavelength--flux column mappings
detected via `map_columns` in the `FitsFlow.fields` module. When those mappings
are absent, conservative fallback rules handle 1D image arrays and tables with
sufficient numeric content. Tables with fewer than five rows are treated as
calibration metadata and skipped. This guards against metadata reference tables
(such as EXTVER/CRVAL1 pairs) being extracted as diagonal-line "spectra".

## Visualisation and Analysis

The spectrum viewer provides interactive Plotly plots [@plotly] with optional
Savitzky--Golay smoothing [@savgol1964] via `scipy.signal.savgol_filter`.
Smoothing window length and polynomial order are user-adjustable through sidebar
sliders.

The anomaly detector in `FitsFlow.detectors` combines four complementary
methods. First, `astropy.stats.sigma_clip` [@astropy2022] removes gross
outliers. Second, a continuum model is estimated with `scipy.signal.savgol_filter`
and subtracted to produce a residual spectrum. Third, outlier points in the
residual are flagged using the median absolute deviation (MAD) via
`astropy.stats.mad_std`. Fourth, `scipy.signal.find_peaks` with prominence
thresholds identifies emission-like spikes and absorption-like dips. Optionally,
`specutils.fitting.find_lines_threshold` [@specutils] can be enabled for
continuum-subtracted line finding on data with explicit wavelength calibration.
This approach is intentionally lightweight, flagging suspicious features without
introducing machine-learning dependencies or black-box classification.

## Reporting

The `FitsFlow.reporters` module compiles spectral plots and FITS images into a
multi-page PDF via ReportLab. The report includes a cover page with session
metadata, a summary page, a methods page, and a processing log, followed by
spectral plots (raw and smoothed overlaid) and FITS image panels. Duplicate
images are prevented through file-path-plus-HDU-index deduplication. Report
generation handles per-spectrum exceptions individually so that one
unrenderable HDU does not abort the full report.

## Archive Integration

The MAST Archive tab uses `astroquery.mast.Observations` [@astroquery2019] to
search by target name and mission, display matching observations, and download
selected science FITS products with a progress bar. Downloaded products are
imported into the same analysis session as uploaded files. The MAST workflow
includes a configurable timeout and a warning when large product counts are
returned.

# Validation

AstroFlow has been tested on real datasets spanning six instrument configurations
and file structures. End-to-end sessions were run for the K2-18 and GJ-1214b
systems, each combining files from multiple sources in a single upload:

| File type | Source | HDUs extracted |
|-----------|--------|---------------|
| JWST MIRI/LRS slitless (`_x1dints.fits`) | MAST Programme 2722 | 30 spectral HDUs, 5--14 μm |
| HST WFC3 G141 grism (`_flt.fits`) | MAST Programme 13665 | SCI extensions, 1.1--1.7 μm |
| TESS light curve (`_lc.fits`) | MAST | SAP\_FLUX time series |
| TESS DVT (`_dvt.fits`) | MAST | TIMECORR table |
| Fermi-LAT photon event file | Public archive | Event table, MET timestamps |
| IRTF Spectral Library (`M0.5V`, `M0IIIb`) | IRTF | 1D stellar spectra |

In each case, AstroFlow correctly extracted science HDUs, skipped calibration
and quality extensions, produced interactive spectra, and generated a complete
multi-page PDF report without crashing. The MIRI slitless spectrum for K2-18
(Programme 2722, @madhusudhan2023) shows the expected rising 5--14 μm continuum
shape across all 30 integration HDUs, consistent with published results. The HST
WFC3 G141 grism for K2-18 (Programme 13665, @kreidberg2014) recovers the
characteristic near-infrared spectral trace shape in the SCI extensions. The
anomaly detector correctly identifies flux spikes in raw HST FLT data consistent
with known cosmic ray signatures in WFC3 grism observations.

These runs demonstrate that AstroFlow is a reusable analysis tool rather than a
single-dataset demonstration, and that its HDU classification logic operates
robustly across heterogeneous FITS archives.

# Limitations and Future Work

AstroFlow focuses on 1D spectra and 2D image HDUs and is not designed to reduce
raw 3D or 4D data cubes, all calibration products, or every possible FITS
convention. Wavelength unit inference is heuristic; physical unit calibration is
not yet enforced for all WCS variants. On hosted Streamlit Cloud deployments,
available RAM is approximately 1 GB, which limits processing of very large FITS
files; the application issues a warning when files exceed 500 MB. Anomaly
detection thresholds are user-adjustable but not yet automatically optimised.

Future work will focus on more explicit WCS-based wavelength calibration, more
instrument-aware spectrum extraction for common JWST and HST data models,
improved anomaly scoring with uncertainty propagation, and extended support for
additional data products where they can be handled safely.

# AI Usage Disclosure

This manuscript was drafted with AI assistance for structure and language, then
reviewed, edited, and validated by the author. The author made all scientific
and architectural decisions and is responsible for the correctness of the
manuscript and software.

# Acknowledgements

The author thanks the open-source astronomy community and the maintainers of
Astropy, SciPy, Streamlit, Plotly, astroquery, and ReportLab.

# References
