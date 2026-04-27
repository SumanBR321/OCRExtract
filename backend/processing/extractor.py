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
import json
import time
from typing import List

import httpx
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
    hint_month: str = "",
    hint_year: str = "",
) -> List[QuestionPaperRecord]:
    """
    Parse *raw_text* (combined OCR output) into one or more records.
    """
    # Split into per-paper sections first
    sections = _split_into_sections(raw_text)
    logger.debug("Split into %d section(s) for '%s'.", len(sections), source_file)

    # Parallelize extraction if multiple sections exist
    from concurrent.futures import ThreadPoolExecutor
    
    def _safe_extract(section):
        try:
            return _extract_one(
                section,
                source_file=source_file,
                hint_school=hint_school,
                hint_degree=hint_degree,
                hint_semester=hint_semester,
                hint_month=hint_month,
                hint_year=hint_year,
            )
        except Exception as exc:
            logger.warning(f"Failed to extract record from section in {source_file}: {exc}")
            return QuestionPaperRecord(
                source_file=source_file,
                school=hint_school,
                degree=hint_degree,
                flags=["extraction_error", str(exc)]
            )

    with ThreadPoolExecutor(max_workers=4) as executor:
        records = list(executor.map(_safe_extract, sections))

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
    hint_month: str,
    hint_year: str,
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

    # -- Year --
    # Prioritize filename hint as ground truth, fallback to regex
    year = int(hint_year) if (hint_year and hint_year.isdigit()) else None
    if not year:
        header_text = text[:1000]
        year_matches = RE_YEAR.findall(header_text)
        year = int(year_matches[0]) if year_matches else None
    
    if not year:
        flags.append("missing_year")
        confidence = ConfidenceLevel.LOW

    # -- Month --
    # Prioritize filename hint as ground truth
    month = _normalise_month_name(hint_month) if hint_month else None
    if not month:
        month = _extract_month(text)

    if not month:
        flags.append("missing_month")
        if confidence == ConfidenceLevel.HIGH:
            confidence = ConfidenceLevel.MEDIUM

    # -- Semester --
    # Prioritize hint_semester or filename parse
    semester = hint_semester or _parse_sem_from_filename(source_file)
    
    # Fallback to OCR only if filename hints are missing
    if not semester:
        sem_match = RE_SEMESTER.search(text)
        if not sem_match:
            sem_match = RE_SEMESTER_WORDS.search(text)
        semester = sem_match.group(1).upper() if sem_match else None
    
    if not semester:
        flags.append("missing_semester")

    # -- Course Title --
    title_match = RE_COURSE_TITLE.search(text)
    course_title = _clean_title(title_match.group(1)) if title_match else None
    
    # Heuristic fallback for Title is DISABLED because it often mistakenly grabs exam questions.
    # We will let the Groq LLM cleanly extract the title instead.
    # if not course_title:
    #     course_title = _heuristic_title_search(text)

    # -- LLM fallback if any major fields are missing --
    if not (course_code and course_title and year and semester):
        llm_data = _llm_fallback(
            text,
            hint_year=year,
            hint_month=month,
            hint_sem=semester
        )
        if llm_data:
            course_code  = course_code  or llm_data.get("course_code")
            course_title = course_title or llm_data.get("course_title")
            
            # Sanitize year from LLM
            llm_year = llm_data.get("year")
            if llm_year and str(llm_year).isdigit():
                year = year or int(llm_year)
            
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


def _normalise_month_name(raw: str) -> str | None:
    """Convert 'DEC' or 'december' to 'December'."""
    raw = raw.strip().capitalize()
    if raw in MONTH_ORDER:
        return raw
    # Check abbreviations
    for full in MONTH_ORDER.keys():
        if raw.startswith(full[:3]):
            return full
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

# ---------------------------------------------------------------------------
# LLM fallback with token chunking
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    """
    Approximate token count. 
    Rule of thumb: 1 token ~= 4 characters for English text.
    We'll use a slightly more conservative 3.5 to be safe.
    """
    return int(len(text) / 3.5)

def _chunk_text(text: str, max_tokens: int = 5500) -> List[str]:
    """
    Split text into chunks that are approximately within the token limit.
    Tries to split at line boundaries.
    """
    max_chars = int(max_tokens * 3.5)
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    lines = text.splitlines()
    current_chunk = []
    current_chars = 0
    
    for line in lines:
        # If a single line is longer than max_chars, we have to split it anyway
        if len(line) > max_chars:
            # Flush current chunk if any
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
                current_chars = 0
            # Split the long line into pieces
            for i in range(0, len(line), max_chars):
                chunks.append(line[i:i+max_chars])
            continue

        if current_chars + len(line) + 1 > max_chars:
            chunks.append("\n".join(current_chunk))
            current_chunk = [line]
            current_chars = len(line) + 1
        else:
            current_chunk.append(line)
            current_chars += len(line) + 1
            
    if current_chunk:
        chunks.append("\n".join(current_chunk))
        
    return chunks

def _llm_fallback(
    text: str,
    hint_year: int | None = None,
    hint_month: str | None = None,
    hint_sem: str | None = None,
) -> dict:
    """
    Call Groq API to extract structured data from OCR text.
    Handles large texts by chunking into < 6000 tokens.
    """
    if not settings.groq_api_key:
        return {}

    # Final merged data
    merged_data = {
        "course_code": None,
        "course_title": None,
        "year": hint_year,
        "month": hint_month,
        "semester": hint_sem
    }

    # Split text into chunks of ~5500 tokens to stay well under the 6000 limit
    chunks = _chunk_text(text, max_tokens=5500)
    logger.info(f"Processing LLM fallback in {len(chunks)} chunk(s).")

    try:
        custom_client = httpx.Client()
        client = Groq(api_key=settings.groq_api_key, http_client=custom_client)
        
        # We only process up to 3 chunks to avoid excessive costs/time, 
        # as metadata is almost always in the first few pages.
        for i, chunk in enumerate(chunks[:3]):
            if i > 0:
                # Small sleep between chunks to avoid bursting rate limits
                time.sleep(2)
                
            context = f"""
            Chunk {i+1} of {len(chunks)}
            Known Metadata:
            - Year: {merged_data['year'] or "Unknown"}
            - Month: {merged_data['month'] or "Unknown"}
            - Semester: {merged_data['semester'] or "Unknown"}
            """

            prompt = f"""
            Extract exam paper metadata from the following OCR text chunk.
            
            {context}
            
            Strict Extraction Rules:
            1. "course_code": Concise alphanumeric code (e.g., "CS2401"). No spaces.
            2. "course_title": The SUBJECT NAME only. 
               - CRITICAL: Never include exam instructions (e.g., "Time: 3 hrs", "Max Marks", "Note:").
               - CRITICAL: Never include parts of questions.
               - CLEAN: Remove OCR noise like extra symbols or leading numbers.
            3. "year": 4-digit number.
            4. "month": Canonical month name (e.g., "December").
            5. "semester": Roman numeral or simple ordinal (e.g., "III", "3rd").
            6. "Logic Check": If the text contains multiple papers, only extract the one relevant to this chunk's header.
            
            Return ONLY a JSON object:
            {{"course_code": "...", "course_title": "...", "year": ..., "month": "...", "semester": "..."}}
            Use null for any field that is missing or unclear.
            
            Text:
            {chunk}
            """
            
            chunk_extracted = {}
            for attempt in range(3):
                try:
                    completion = client.chat.completions.create(
                        model=settings.groq_model,
                        messages=[{"role": "user", "content": prompt}],
                        response_format={"type": "json_object"},
                        temperature=0.1,
                    )
                    chunk_extracted = json.loads(completion.choices[0].message.content)
                    break
                except Exception as e:
                    err_msg = str(e)
                    if "429" in err_msg or "Rate limit" in err_msg:
                        wait_time = 15 * (attempt + 1)
                        logger.warning(f"Groq rate limit hit for chunk {i+1}. Waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.error(f"Groq chunk {i+1} failed: {err_msg}")
                        break
            
            # Merge extracted data
            if chunk_extracted:
                for key in ["course_code", "course_title", "year", "month", "semester"]:
                    val = chunk_extracted.get(key)
                    if not val or val == "null":
                        continue
                        
                    # Basic cleaning for strings
                    if isinstance(val, str):
                        val = val.strip()
                        if key == "course_code":
                            val = re.sub(r"\s+", "", val).upper()
                        elif key == "course_title":
                            # Secondary check to strip instructions if LLM included them
                            val = re.sub(r"(?i)(Time|Max|Min|Note|Instruction).*[:\-].*", "", val).strip()
                    
                    # Only update if we don't already have a value for this field
                    if merged_data[key] is None:
                        merged_data[key] = val
            
            # Optimization: If we have core metadata, stop processing chunks
            if merged_data["course_code"] and merged_data["course_title"] and merged_data["year"]:
                break

        return merged_data

    except Exception as e:
        logger.error(f"Groq extraction process failed: {str(e)}")
        return merged_data
