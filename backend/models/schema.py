"""
schema.py — Pydantic data models for OCRExtract.
Defines the canonical shape of an extracted question paper record.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    HIGH   = "high"
    MEDIUM = "medium"
    LOW    = "low"


class DegreeType(str, Enum):
    BCA    = "BCA"
    BSC    = "BSc"
    BTECH  = "BTech"
    MTECH  = "MTech"
    MINORS = "Minors"
    OTHER  = "Other"


MONTH_ORDER = {
    "January": 1, "February": 2, "March": 3,   "April":    4,
    "May":     5, "June":     6, "July":  7,   "August":   8,
    "September": 9, "October": 10, "November": 11, "December": 12,
}

DEGREE_ORDER = {
    DegreeType.BCA:    1,
    DegreeType.BSC:    2,
    DegreeType.BTECH:  3,
    DegreeType.MTECH:  4,
    DegreeType.MINORS: 5,
    DegreeType.OTHER:  6,
}


# ---------------------------------------------------------------------------
# Core record
# ---------------------------------------------------------------------------

class QuestionPaperRecord(BaseModel):
    """
    One row in the final Excel output.
    All fields are optional to support partial extraction — confidence
    indicates how complete / reliable the record is.
    """

    school:       Optional[str] = Field(None, description="School / department name")
    semester:     Optional[str] = Field(None, description="Semester label e.g. 'III'")
    degree:       Optional[str] = Field(None, description="Degree programme e.g. 'BTech'")
    course_code:  Optional[str] = Field(None, description="Course code e.g. 'CS2401'")
    course_title: Optional[str] = Field(None, description="Full course title")
    month:        Optional[str] = Field(None, description="Exam month e.g. 'November'")
    year:         Optional[int] = Field(None, description="Exam year e.g. 2023")

    # Internal tracking fields (not exported to Excel)
    source_file:  Optional[str] = Field(None, description="Original Drive file name")
    confidence:   ConfidenceLevel = Field(ConfidenceLevel.HIGH, description="Extraction confidence")
    flags:        list[str] = Field(default_factory=list, description="Validation / QC flags")

    @field_validator("year")
    @classmethod
    def year_must_be_reasonable(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and not (1990 <= v <= 2100):
            raise ValueError(f"Year {v} is outside reasonable range (1990–2100)")
        return v

    @field_validator("month")
    @classmethod
    def month_must_be_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in MONTH_ORDER:
            raise ValueError(f"'{v}' is not a recognised month name")
        return v

    def is_valid(self) -> bool:
        """A record is considered valid if it has at least course_code and year."""
        return bool(self.course_code and self.year)

    def to_excel_row(self) -> dict:
        """Return a dict keyed by the Excel column headers."""
        return {
            "School":       self.school       or "",
            "SEM":          self.semester     or "",
            "Degree":       self.degree       or "",
            "Course Code":  self.course_code  or "",
            "Course Title": self.course_title or "",
            "Month":        self.month        or "",
            "Year":         self.year         or "",
            "Source File":  self.source_file  or "",
        }


# ---------------------------------------------------------------------------
# Progress / status snapshot
# ---------------------------------------------------------------------------

class ProcessingStatus(BaseModel):
    """Live state snapshot served by GET /status."""

    total_files:    int = 0
    processed_files: int = 0
    current_file:   Optional[str] = None
    extracted_rows: int = 0
    errors:         int = 0
    excel_url:      Optional[str] = None
    recent_records: list[dict] = Field(default_factory=list)
    logs:           list[str] = Field(default_factory=list)
    is_running:     bool = False
    is_complete:    bool = False

    @property
    def progress_pct(self) -> float:
        if self.total_files == 0:
            return 0.0
        return round((self.processed_files / self.total_files) * 100, 1)
