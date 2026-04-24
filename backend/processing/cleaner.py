"""
cleaner.py — Normalise and fix extracted field values.

Handles:
  - Whitespace normalisation
  - Common OCR substitution errors (0↔O, 1↔I, etc.)
  - Month name standardisation
  - Degree label normalisation
  - Course code formatting
"""

from __future__ import annotations

import re
from typing import Optional

from backend.models.schema import (
    QuestionPaperRecord,
    DegreeType,
    MONTH_ORDER,
)
from backend.utils.logger import get_logger

logger = get_logger("cleaner")

# ---------------------------------------------------------------------------
# OCR character confusion map (applied to course codes & titles)
# ---------------------------------------------------------------------------
_OCR_CHAR_MAP: dict[str, str] = {
    "0": "O",   # digit zero  ↔ letter O  (context-dependent, applied carefully)
    "|": "I",   # pipe        → I
    "l": "l",   # already fine
    "—": "-",   # em-dash     → hyphen
    "–": "-",   # en-dash     → hyphen
    "\u00a0": " ",   # non-breaking space
    "\t": " ",
}

# Degree alias map: maps OCR-garbled variants → canonical DegreeType values
_DEGREE_ALIASES: dict[str, str] = {
    "b.ca": DegreeType.BCA,
    "bca":  DegreeType.BCA,
    "b.sc": DegreeType.BSC,
    "bsc":  DegreeType.BSC,
    "b.tech": DegreeType.BTECH,
    "btech":  DegreeType.BTECH,
    "b.e":    DegreeType.BTECH,
    "m.tech": DegreeType.MTECH,
    "mtech":  DegreeType.MTECH,
    "m.e":    DegreeType.MTECH,
    "minor":  DegreeType.MINORS,
    "minors": DegreeType.MINORS,
}

# Month alias map: OCR mistakes → canonical month name
_MONTH_ALIASES: dict[str, str] = {
    "janaury":   "January",
    "januray":   "January",
    "febuary":   "February",
    "feburary":  "February",
    "marchh":    "March",
    "apirl":     "April",
    "aprll":     "April",
    "junee":     "June",
    "julv":      "July",
    "augst":     "August",
    "septembar": "September",
    "septemher": "September",
    "octobor":   "October",
    "novembar":  "November",
    "novemher":  "November",
    "decembar":  "December",
    "decemher":  "December",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def clean_record(record: QuestionPaperRecord) -> QuestionPaperRecord:
    """
    Return a *new* record with all fields cleaned.
    The original is not mutated.
    """
    data = record.model_dump()

    data["school"]       = _clean_text(data.get("school"))
    data["course_title"] = _clean_text(data.get("course_title"))
    data["course_code"]  = _clean_course_code(data.get("course_code"))
    data["degree"]       = _normalise_degree(data.get("degree"))
    data["month"]        = _normalise_month(data.get("month"))
    data["semester"]     = _clean_semester(data.get("semester"))

    return QuestionPaperRecord(**data)


# ---------------------------------------------------------------------------
# Field-level cleaners
# ---------------------------------------------------------------------------

def _clean_text(value: Optional[str]) -> Optional[str]:
    """
    Normalise whitespace, strip, and fix common unicode issues.
    """
    if not value:
        return value
    # Replace non-breaking spaces, tabs, em-dashes
    for bad, good in _OCR_CHAR_MAP.items():
        if bad in (" ", "-"):   # skip char-level substitution for common chars
            continue
        value = value.replace(bad, good)
    # Collapse multiple spaces
    value = re.sub(r"[ ]{2,}", " ", value)
    # Strip
    value = value.strip()
    return value or None


def _clean_course_code(code: Optional[str]) -> Optional[str]:
    if not code:
        return code
    # Remove spaces, uppercase
    code = re.sub(r"\s+", "", code).upper()
    # Fix digit/letter confusion in the letter prefix only
    # (e.g. "CS24O1" → "CS2401" — O→0 in numeric portion)
    # Split into letter prefix + numeric suffix
    m = re.match(r"^([A-Z]+)(\d+[A-Z]?)$", code)
    if m:
        prefix, suffix = m.group(1), m.group(2)
        # In suffix: O→0, I→1, S→5
        suffix = suffix.replace("O", "0").replace("I", "1").replace("S", "5")
        code = prefix + suffix
    return code


def _normalise_degree(degree: Optional[str]) -> Optional[str]:
    if not degree:
        return degree
    key = degree.strip().lower().replace(" ", "")
    canonical = _DEGREE_ALIASES.get(key)
    if canonical:
        return canonical
    # Return title-cased original if not found
    return degree.strip().title()


def _normalise_month(month: Optional[str]) -> Optional[str]:
    if not month:
        return month
    # Try alias map first
    key = month.strip().lower()
    alias = _MONTH_ALIASES.get(key)
    if alias:
        return alias
    # Try capitalising and checking against known months
    capitalised = month.strip().capitalize()
    if capitalised in MONTH_ORDER:
        return capitalised
    # Try abbreviation expansion
    for full in MONTH_ORDER:
        if capitalised.startswith(full[:3]):
            return full
    logger.debug("Could not normalise month: '%s'", month)
    return month.strip().capitalize()


def _clean_semester(sem: Optional[str]) -> Optional[str]:
    if not sem:
        return sem
    # Strip whitespace, uppercase
    sem = sem.strip().upper()
    
    # Common words → Roman
    word_map = {
        "FIRST": "I", "SECOND": "II", "THIRD": "III", "FOURTH": "IV",
        "FIFTH": "V", "SIXTH": "VI", "SEVENTH": "VII", "EIGHTH": "VIII"
    }
    if sem in word_map:
        return word_map[sem]
    
    # Remove "SEM" or "SEMESTER" if it's still there
    sem = re.sub(r"SEM(?:ESTER)?", "", sem).strip()
    
    # Only keep Roman numerals, Digits, and ordinal suffixes
    sem = re.sub(r"[^IVX0-9]", "", sem)
    return sem or None
