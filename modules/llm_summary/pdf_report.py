"""modules/llm_summary/pdf_report.py
Generates a styled PDF from the summary text using reportlab.
"""
from __future__ import annotations

import io
import re
import textwrap
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    Table,
    TableStyle,
)
from reportlab.lib.enums import TA_LEFT, TA_CENTER


# ── Colour palette ────────────────────────────────────────────────────────
AMBER      = colors.HexColor("#f59e0b")
AMBER_DARK = colors.HexColor("#d97706")
DARK_BG    = colors.HexColor("#0f172a")
SLATE      = colors.HexColor("#334155")
TEXT_CLR   = colors.HexColor("#1e293b")
MUTED      = colors.HexColor("#64748b")


def _build_styles():
    """Return a dict of named ParagraphStyles for the PDF."""
    base = getSampleStyleSheet()

    return {
        "title": ParagraphStyle(
            "PDFTitle",
            parent=base["Heading1"],
            fontSize=18,
            leading=22,
            textColor=DARK_BG,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "subtitle": ParagraphStyle(
            "PDFSubtitle",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=MUTED,
            spaceAfter=14,
        ),
        "section_heading": ParagraphStyle(
            "PDFSectionH",
            parent=base["Heading2"],
            fontSize=12,
            leading=15,
            textColor=AMBER_DARK,
            spaceBefore=16,
            spaceAfter=6,
            fontName="Helvetica-Bold",
            borderPadding=(0, 0, 0, 4),
        ),
        "sub_heading": ParagraphStyle(
            "PDFSubH",
            parent=base["Heading3"],
            fontSize=10,
            leading=13,
            textColor=TEXT_CLR,
            spaceBefore=10,
            spaceAfter=4,
            fontName="Helvetica-Bold",
        ),
        "body": ParagraphStyle(
            "PDFBody",
            parent=base["Normal"],
            fontSize=9,
            leading=14,
            textColor=TEXT_CLR,
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "PDFBullet",
            parent=base["Normal"],
            fontSize=9,
            leading=13,
            textColor=TEXT_CLR,
            leftIndent=14,
            spaceAfter=3,
            bulletIndent=4,
        ),
    }


def _escape_xml(text: str) -> str:
    """Escape XML entities for reportlab Paragraph."""
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def _inline_format(text: str) -> str:
    """Convert **bold** and *italic* to reportlab XML tags."""
    text = _escape_xml(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__(.+?)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"(?<!_)_([^_]+?)_(?!_)", r"<i>\1</i>", text)
    return text


def generate_pdf(summary_text: str, filename: str = "SolarPV_Report") -> io.BytesIO:
    """
    Generate a styled A4 PDF from the summary text.
    Returns a BytesIO buffer containing the PDF data.
    """
    buf = io.BytesIO()
    styles = _build_styles()

    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=20 * mm,
        bottomMargin=18 * mm,
        title=filename,
        author="SolaritivityPlus",
    )

    story = []

    # ── Header ────────────────────────────────────────────────────────────
    story.append(Paragraph("☀ Solar Panel Inspection Report", styles["title"]))
    story.append(
        Paragraph(
            f"Generated on {datetime.now().strftime('%d %B %Y at %H:%M')}  •  SolaritivityPlus AI Engine",
            styles["subtitle"],
        )
    )
    story.append(
        HRFlowable(
            width="100%", thickness=1.5, color=AMBER, spaceAfter=12, spaceBefore=2
        )
    )

    # ── Parse the summary text into sections ──────────────────────────────
    text = summary_text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove separator lines
    text = re.sub(r"^={3,}.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}\s*$", "", text, flags=re.MULTILINE)

    # Split by SECTION N:
    section_pattern = re.compile(
        r"(?:^|\n)\s*SECTION\s+\d+\s*:\s*(.*)", re.IGNORECASE
    )
    matches = list(section_pattern.finditer(text))

    if not matches:
        # No sections found — dump as body text
        for line in text.strip().split("\n"):
            line = line.strip()
            if not line:
                continue
            story.append(Paragraph(_inline_format(line), styles["body"]))
    else:
        for i, m in enumerate(matches):
            title = m.group(1).strip().rstrip("=-").strip()
            start = text.index("\n", m.start() + 1) if "\n" in text[m.start() + 1 :] else m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            body = text[start:end].strip()

            # Section heading
            story.append(Paragraph(_inline_format(title.upper()), styles["section_heading"]))

            # Process body lines
            for line in body.split("\n"):
                stripped = line.strip()
                if not stripped or re.match(r"^[=\-]{3,}$", stripped):
                    continue
                if re.match(
                    r"^(SOLAR PANEL INSPECTION REPORT|Image Reference:|END OF REPORT|Rules:|\[DEFECT BLOCK)",
                    stripped,
                    re.IGNORECASE,
                ):
                    continue

                # Sub-headings
                if re.match(
                    r"^(?:\d+[a-z]?[.)\s]|Defect Name|What This Defect|Detection Evidence|Root Causes|Consequences|Defect Priority|Immediate Actions|Short-Term|Preventive|Carbon Emission|Risk Level|Risk Justification|CO2|Power and Efficiency|Operational)",
                    stripped,
                    re.IGNORECASE,
                ):
                    story.append(Paragraph(_inline_format(stripped), styles["sub_heading"]))
                    continue

                # Bullet items
                if re.match(r"^\s*[-•]\s+", line):
                    content = re.sub(r"^[-•]\s+", "", stripped)
                    story.append(
                        Paragraph(f"▸ {_inline_format(content)}", styles["bullet"])
                    )
                    continue

                # Regular paragraph
                story.append(Paragraph(_inline_format(stripped), styles["body"]))

    # ── Footer separator ──────────────────────────────────────────────────
    story.append(Spacer(1, 16))
    story.append(
        HRFlowable(
            width="100%", thickness=1, color=SLATE, spaceAfter=6, spaceBefore=4
        )
    )
    story.append(
        Paragraph(
            "This report was generated by <b>SolaritivityPlus</b> using Gemini AI. "
            "Data sourced from detections.csv and carbon.csv.",
            styles["subtitle"],
        )
    )

    doc.build(story)
    buf.seek(0)
    return buf
