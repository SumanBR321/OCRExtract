"""
extractor.py — Hybrid extraction engine.

Strategy (in priority order):
  1. Regex patterns for course code, year, month, time/marks boundaries
  2. Fuzzy matching for OCR-corrupted month names
  3. Multi-paper split: one PDF page may contain multiple exam papers
  4. LLM fallback placeholder (stub — wire in any API you like)
"""

from __future__ import annotations

import re
from typing import List

from rapidfuzz import process as fuzz_process
from groq import Groq

from backend.config import settings
from backend.models.schema import QuestionPaperRecord, ConfidenceLevel, MONTH_ORDER
from backend.utils.logger import get_logger

logger = get_logger("extractor")

# ---------------------------------------------------------------------------
# Compiled regex patterns
# ---------------------------------------------------------------------------

# Course codes like CS2401, BCA301, MT101, ECE4402 etc.
RE_COURSE_CODE = re.compile(
    r"\b([A-Z]{2,5}\s?\d{3,5}[A-Z]?)\b"
)

# 4-digit year in the range 1990–2099
RE_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")

# Month names (full and abbreviated)
_MONTH_NAMES_FULL  = list(MONTH_ORDER.keys())
_MONTH_NAMES_ABBR  = [m[:3] for m in _MONTH_NAMES_FULL]
_ALL_MONTHS = _MONTH_NAMES_FULL + _MONTH_NAMES_ABBR
RE_MONTH = re.compile(
    r"\b(" + "|".join(_MONTH_NAMES_FULL + _MONTH_NAMES_ABBR) + r")\b",
    re.IGNORECASE,
)

# Semester: Roman (I-VIII) or Digits (1-8), optionally with SEM/SEMESTER
RE_SEMESTER = re.compile(
    r"(?:SEM(?:ESTER)?|SESSION)\s*[:\-]?\s*([IVX]{1,5}|[1-8](?:ST|ND|RD|TH)?)\b",
    re.IGNORECASE,
)

# Alternative: "Third Semester", "Fourth Sem"
RE_SEMESTER_WORDS = re.compile(
    r"\b(FIRST|SECOND|THIRD|FOURTH|FIFTH|SIXTH|SEVENTH|EIGHTH)\s+SEM(?:ESTER)?\b",
    re.IGNORECASE,
)

# Course title: common labels
RE_COURSE_TITLE = re.compile(
    r"(?:Subject|Course|Paper|Title|Branch)\s*[:\-]\s*(.+)",
    re.IGNORECASE,
)

# Multi-paper split keywords
_SPLIT_KEYWORDS = re.compile(
    r"(?:Time\s*:|Max(?:imum)?\s*Marks|Full\s*Marks|Duration\s*:)",
    re.IGNORECASE,
)

# Fuzzy matching threshold (0–100)
FUZZY_THRESHOLD = 75


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_records(
    raw_text: str,
    source_file: str = "",
    hint_school: str = "",
    hint_degree: str = "",
    hint_semester: str = "",
) -> List[QuestionPaperRecord]:
    """
    Parse *raw_text* (combined OCR output) into one or more records.

    Hints come from the Drive folder hierarchy and are used as fallbacks
    when regex cannot find the values in the text itself.
    """
    # Split into per-paper sections first
    sections = _split_into_sections(raw_text)
    logger.debug("Split into %d section(s) for '%s'.", len(sections), source_file)

    records: List[QuestionPaperRecord] = []
    for section in sections:
        record = _extract_one(
            section,
            source_file=source_file,
            hint_school=hint_school,
            hint_degree=hint_degree,
            hint_semester=hint_semester,
        )
        records.append(record)

    return records


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_into_sections(text: str) -> List[str]:
    """
    Split the combined OCR text into individual exam-paper blocks using
    boundary keywords (Time:, Max Marks, etc.).
    If no boundary is found, return the whole text as a single block.
    """
    positions = [m.start() for m in _SPLIT_KEYWORDS.finditer(text)]
    if len(positions) < 2:
        return [text]

    # Each section starts at a keyword boundary
    sections = []
    for i, pos in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(text)
        sections.append(text[pos:end].strip())
    return sections or [text]


def _extract_one(
    text: str,
    source_file: str,
    hint_school: str,
    hint_degree: str,
    hint_semester: str,
) -> QuestionPaperRecord:
    """Extract a single record from one paper section."""
    flags: list[str] = []
    confidence = ConfidenceLevel.HIGH

    # -- Course Code --
    cc_match = RE_COURSE_CODE.search(text)
    course_code = _normalise_code(cc_match.group(1)) if cc_match else None
    if not course_code:
        flags.append("missing_course_code")
        confidence = ConfidenceLevel.LOW

    # -- Year (Only look in the top 1000 characters - the header) --
    header_text = text[:1000]
    year_matches = RE_YEAR.findall(header_text)
    year = int(year_matches[0]) if year_matches else None
    
    if not year:
        flags.append("missing_year")
        confidence = ConfidenceLevel.LOW

    # -- Month --
    month = _extract_month(text)
    if not month:
        flags.append("missing_month")
        if confidence == ConfidenceLevel.HIGH:
            confidence = ConfidenceLevel.MEDIUM

    # -- Semester --
    sem_match = RE_SEMESTER.search(text)
    if not sem_match:
        sem_match = RE_SEMESTER_WORDS.search(text)
    
    semester = sem_match.group(1).upper() if sem_match else None
    
    # Fallback to hint or filename for semester
    if not semester:
        semester = hint_semester or _parse_sem_from_filename(source_file)
    
    if not semester:
        flags.append("missing_semester")

    # -- Course Title --
    title_match = RE_COURSE_TITLE.search(text)
    course_title = _clean_title(title_match.group(1)) if title_match else None
    
    # Heuristic fallback for Title: look for prominent lines before split keywords
    if not course_title:
        course_title = _heuristic_title_search(text)

    # -- LLM fallback if any major fields are missing --
    if not (course_code and course_title and year and semester):
        llm_data = _llm_fallback(text)
        if llm_data:
            course_code  = course_code  or llm_data.get("course_code")
            course_title = course_title or llm_data.get("course_title")
            year         = year         or llm_data.get("year")
            month        = month        or llm_data.get("month")
            semester     = semester     or llm_data.get("semester")
            flags.append("llm_assisted")
            confidence = ConfidenceLevel.MEDIUM

    return QuestionPaperRecord(
        school       = hint_school   or None,
        semester     = semester,
        degree       = hint_degree   or None,
        course_code  = course_code,
        course_title = course_title,
        month        = month,
        year         = year,
        source_file  = source_file,
        confidence   = confidence,
        flags        = flags,
    )


def _extract_month(text: str) -> str | None:
    """
    Try exact regex first; fall back to fuzzy matching on each token.
    Returns the canonical month name (e.g. 'November') or None.
    """
    # Exact match
    m = RE_MONTH.search(text)
    if m:
        raw = m.group(1).capitalize()
        # Normalise abbreviation → full name
        for full in _MONTH_NAMES_FULL:
            if raw.startswith(full[:3]):
                return full
        return raw if raw in MONTH_ORDER else None

    # Fuzzy fallback — check each whitespace token
    for token in re.findall(r"[A-Za-z]{3,}", text):
        result = fuzz_process.extractOne(
            token.capitalize(),
            _MONTH_NAMES_FULL,
            score_cutoff=FUZZY_THRESHOLD,
        )
        if result:
            logger.debug("Fuzzy month: '%s' → '%s' (score=%d)", token, result[0], result[1])
            return result[0]

    return None


def _normalise_code(raw: str) -> str:
    """Remove internal spaces from codes like 'CS 2401' → 'CS2401'."""
    return re.sub(r"\s+", "", raw).upper()


def _clean_title(raw: str) -> str:
    """Strip trailing punctuation / extra whitespace / OCR noise from a title."""
    # Remove leading/trailing non-alphanumeric junk
    cleaned = re.sub(r"^[^A-Za-z0-9]+|[^A-Za-z0-9)]+$", "", raw.strip())
    # If the title is just a code, ignore it
    if RE_COURSE_CODE.fullmatch(cleaned):
        return None
    return cleaned if len(cleaned) > 2 else None


def _parse_sem_from_filename(filename: str) -> str | None:
    """Extract semester hint from filename like 'SOB-VI-SEM-...'"""
    m = re.search(r"-([IVX]{1,5}|[1-8])-SEM", filename, re.IGNORECASE)
    return m.group(1).upper() if m else None


def _heuristic_title_search(text: str) -> str | None:
    """
    Look for a line that looks like a course title.
    Usually it's between the Examination line and the Time line.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    
    # Find the boundary: Time/Marks/Duration
    boundary_idx = -1
    for i, line in enumerate(lines):
        if _SPLIT_KEYWORDS.search(line):
            boundary_idx = i
            break
    
    if boundary_idx > 0:
        # Search backwards from the boundary for a prominent line (all caps or distinct)
        # Skip lines that look like Degree or Semester or Date
        for i in range(boundary_idx - 1, -1, -1):
            line = lines[i]
            # Ignore if it's too short, just a code, or contains "Examination"
            if len(line) < 5: continue
            if "EXAMINATION" in line.upper(): continue
            if "SEMESTER" in line.upper(): continue
            if RE_COURSE_CODE.search(line): continue
            
            return _clean_title(line)
            
    return None


# ---------------------------------------------------------------------------
# LLM fallback stub
# ---------------------------------------------------------------------------

def _llm_fallback(text: str) -> dict:
    """
    Call Groq API to extract structured data from OCR text.
    """
    if not settings.groq_api_key:
        return {}

    try:
        client = Groq(api_key=settings.groq_api_key)
        
        prompt = f"""
        Extract exam paper metadata from the following OCR text.
        Return ONLY a JSON object with these keys: 
        "course_code", "course_title", "year" (int), "month", "semester" (e.g. "III" or "3rd").
        
        Text:
        {text[:4000]}
        """
        
        completion = client.chat.completions.create(
            model=settings.groq_model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        
        import json
        return json.loads(completion.choices[0].message.content)
    except Exception as e:
        logger.error(f"Groq extraction failed: {str(e)}")
        return {}
