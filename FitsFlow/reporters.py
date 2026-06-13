"""
Report generation utilities for AstroFlow.

Generates PDF reports that collect:
- Spectral plots (PNG)
- Telescope images from FITS HDUs (PNG)
- Data tables (CSV → rendered as tables)
"""

from fpdf import FPDF
import os, csv, datetime

class PDFReport(FPDF):
    def header(self):
        # Title at the top of each page
        if hasattr(self, "title") and self.page_no() > 1:
            self.set_font("Arial", "B", 12)
            self.cell(0, 10, self.title, 0, 1, "C")
        self.ln(5)

    def footer(self):
        # Page numbers
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", 0, 0, "C")


def generate_pdf_report(output_path, metadata, plots, tables, images):
    """
    Build a PDF report with given assets.

    Args:
        output_path (str): File path for the PDF.
        metadata (dict): Info like {"title": "AstroFlow Report"}.
        plots (list[str]): Paths to PNG spectral plots.
        tables (list[str]): Paths to CSV tables.
        images (list[str]): Paths to PNG FITS images.

    Returns:
        str: Path to the generated PDF.
    """
    pdf = PDFReport()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.title = metadata.get("title", "AstroFlow Report")

    # --- Cover page ---
    pdf.add_page()
    pdf.set_font("Arial", "B", 20)
    pdf.cell(0, 20, pdf.title, ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
    pdf.ln(10)
    if "author" in metadata:
        pdf.cell(0, 10, f"Author: {metadata['author']}", ln=True, align="C")

    # --- Spectra plots ---
    if plots:
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Spectral Plots", ln=True)
        pdf.ln(5)

        for p in plots:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, os.path.basename(p), ln=True)
            pdf.image(p, x=15, w=180)
            pdf.ln(5)

    # --- FITS Images ---
    if images:
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "FITS Images", ln=True)
        pdf.ln(5)

        for img in images:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, os.path.basename(img), ln=True)
            pdf.image(img, x=30, w=150)
            pdf.ln(5)

    # --- Data Tables ---
    if tables:
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Data Tables", ln=True)
        pdf.ln(5)

        for t in tables:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 8, os.path.basename(t), ln=True)
            pdf.ln(2)

            with open(t, "r") as fh:
                reader = csv.reader(fh)
                rows = list(reader)

            # Limit rows for readability
            max_rows = 20
            display_rows = rows[:max_rows]

            pdf.set_font("Courier", "", 8)
            col_width = pdf.w / (len(display_rows[0]) + 1)
            for row in display_rows:
                line = " | ".join(row)
                pdf.multi_cell(0, 5, line)

            if len(rows) > max_rows:
                pdf.set_font("Arial", "I", 8)
                pdf.cell(0, 5, f"... ({len(rows)-max_rows} more rows not shown)", ln=True)

            pdf.ln(4)

    # Save
    pdf.output(output_path)
    return output_path
