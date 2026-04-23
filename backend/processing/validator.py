"""
validator.py — Post-cleaning validation gate.

Rules:
  - course_code must exist and match expected pattern
  - year must be in range 1990–2099
  - month (if present) must be a recognised month name
  - Records failing mandatory checks are flagged, not silently dropped
"""

from __future__ import annotations

import re
from typing import List, Tuple

from backend.models.schema import QuestionPaperRecord, ConfidenceLevel, MONTH_ORDER
from backend.utils.logger import get_logger

logger = get_logger("validator")

# Strict course-code pattern (after cleaning)
RE_VALID_CODE = re.compile(r"^[A-Z]{2,5}\d{3,5}[A-Z]?$")

VALID_YEAR_RANGE = range(1990, 2100)


def validate_records(
    records: List[QuestionPaperRecord],
) -> Tuple[List[QuestionPaperRecord], List[QuestionPaperRecord]]:
    """
    Split *records* into (valid, invalid) lists.

    A record is *valid* if it has a properly formatted course_code
    AND a year within the acceptable range.

    Invalid records are returned separately (rather than dropped) so
    the caller can log or inspect them.
    """
    valid:   List[QuestionPaperRecord] = []
    invalid: List[QuestionPaperRecord] = []

    for record in records:
        issues = _check(record)
        if issues:
            # Attach issues as flags and downgrade confidence
            updated = record.model_copy(
                update={
                    "flags": record.flags + issues,
                    "confidence": ConfidenceLevel.LOW,
                }
            )
            invalid.append(updated)
            logger.warning(
                "Invalid record from '%s': %s",
                record.source_file or "unknown",
                ", ".join(issues),
            )
        else:
            valid.append(record)

    logger.info(
        "Validation: %d valid / %d invalid out of %d records.",
        len(valid), len(invalid), len(records),
    )
    return valid, invalid


# ---------------------------------------------------------------------------
# Individual check helpers
# ---------------------------------------------------------------------------

def _check(record: QuestionPaperRecord) -> List[str]:
    """Return a list of issue strings (empty = all good)."""
    issues: List[str] = []

    # Mandatory: course code
    if not record.course_code:
        issues.append("course_code_missing")
    elif not RE_VALID_CODE.match(record.course_code):
        issues.append(f"course_code_invalid:{record.course_code!r}")

    # Mandatory: year
    if record.year is None:
        issues.append("year_missing")
    elif record.year not in VALID_YEAR_RANGE:
        issues.append(f"year_out_of_range:{record.year}")

    # Optional but validated if present: month
    if record.month and record.month not in MONTH_ORDER:
        issues.append(f"month_unrecognised:{record.month!r}")

    return issues
