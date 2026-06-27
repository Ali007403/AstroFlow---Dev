# AstroFlow

AstroFlow is an interactive Streamlit application for loading, inspecting, analyzing, and reporting on astronomical FITS and CSV data. It is designed for quick, reproducible exploration of heterogeneous data products from missions and archives such as JWST, HST, TESS, and MAST.

## What AstroFlow does

- Loads FITS and CSV files for spectral and table-based analysis
- Detects and skips common non-science FITS extensions
- Plots spectra interactively
- Applies Savitzky–Golay smoothing
- Flags statistically unusual points and emission/absorption-like features
- Searches MAST and imports science products
- Generates multi-page PDF reports
- Exports spectra, tables, and images for download

## Installation

```bash
git clone https://github.com/OWNER/REPO.git
cd REPO
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the app

```bash
streamlit run app.py
```

## Quick start

1. Open the app in your browser.
2. Upload one or more FITS or CSV files.
3. Use the tabs to inspect spectra, tables, images, anomalies, and reports.
4. Optionally search MAST from the sidebar and import products into the same session.

## Development and testing

Run the test suite locally with:

```bash
pytest -q
```

Recommended continuous integration: add a GitHub Actions workflow that installs the dependencies and runs `pytest -q` on every push and pull request.

## Repository structure

```text
app.py                # Streamlit application entry point
FitsFlow/             # Core processing modules
README.md             # Project overview and installation
paper.md              # JOSS manuscript
paper.bib             # Bibliography for the paper
requirements.txt      # Runtime dependencies
test_astroflow.py     # Automated tests
```

## Citation

If you use AstroFlow in your work, please cite the software using `CITATION.cff` and the JOSS paper once published.

## Limitations

AstroFlow is optimized for 1D spectra, tabular data, and 2D image HDUs. It is not intended to replace mission-specific reduction pipelines or to process every possible FITS convention.

## License

MIT License.
