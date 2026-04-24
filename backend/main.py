"""
main.py — FastAPI application entry point.

Endpoints:
  GET  /          → serve frontend (via static files)
  GET  /status    → live progress snapshot
  POST /start     → kick off the processing pipeline
  GET  /download  → download the generated Excel file
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.drive.drive_client import DriveClient
from backend.models.schema import ProcessingStatus, QuestionPaperRecord
from backend.processing.cleaner import clean_record
from backend.processing.extractor import extract_records
from backend.processing.ocr_engine import run_ocr_on_images
from backend.processing.pdf_to_image import pdf_to_images
from backend.processing.preprocess import preprocess_image
from backend.processing.validator import validate_records
from backend.services.excel_writer import write_excel
from backend.state.progress_tracker import tracker
from backend.utils.logger import get_logger

logger = get_logger("main")

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OCRExtract API",
    description="Scanned question-paper OCR pipeline with live progress tracking.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if _FRONTEND_DIR.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


# ---------------------------------------------------------------------------
# State shared across requests
# ---------------------------------------------------------------------------

_excel_output_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse, include_in_schema=False)
async def root():
    index = _FRONTEND_DIR / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return {"message": "OCRExtract API is running. Visit /docs for API reference."}


@app.get("/status", response_model=ProcessingStatus, summary="Live progress snapshot")
async def get_status():
    """Returns the current pipeline progress as a JSON object."""
    return tracker.snapshot()


@app.post("/start", summary="Start the OCR pipeline")
async def start_pipeline(background_tasks: BackgroundTasks):
    """
    Begins the full pipeline asynchronously.
    """
    if tracker.is_running():
        raise HTTPException(status_code=409, detail="Pipeline is already running.")

    if not settings.drive_root_folder_id:
        raise HTTPException(
            status_code=400,
            detail="DRIVE_ROOT_FOLDER_ID is not configured. Set it in your .env file.",
        )

    tracker.reset()
    background_tasks.add_task(_run_pipeline)
    return {"message": "Pipeline started."}


@app.post("/stop", summary="Stop the OCR pipeline")
async def stop_pipeline():
    """Signals the running pipeline to stop safely."""
    tracker.stop()
    return {"message": "Stop signal sent to pipeline."}


@app.get("/download", summary="Download the generated Excel file")
async def download_excel():
    """Returns the most recently generated Excel file."""
    global _excel_output_path
    if _excel_output_path is None or not _excel_output_path.exists():
        raise HTTPException(status_code=404, detail="No Excel file has been generated yet.")
    return FileResponse(
        str(_excel_output_path),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=_excel_output_path.name,
    )


# ---------------------------------------------------------------------------
# Pipeline implementation
# ---------------------------------------------------------------------------

def _run_pipeline() -> None:
    """
    Full pipeline executed in a background thread.
    Catches all exceptions to ensure the tracker always reaches a terminal
    state so the frontend doesn't spin forever.
    """
    global _excel_output_path

    try:
        logger.info("=== Pipeline starting ===")
        tracker.log("🚀 Pipeline starting…")

        # ── 1. Connect to Drive and enumerate PDFs ──────────────────────────
        tracker.log("📂 Connecting to Google Drive…")
        drive = DriveClient(
            credentials_path=settings.credentials_path,
            download_dir=settings.downloads_path,
        )

        tracker.log(f"📂 Traversing folder: {settings.drive_root_folder_id}")
        pdf_files = list(drive.iter_pdfs(settings.drive_root_folder_id))
        total = len(pdf_files)

        # ── Checkpoints & Persistence ──────────────────────────────────────
        checkpoint_path = Path("output/processed_files.json")
        backup_path     = Path("output/records_backup.json")
        final_excel_path = Path("output/OCRExtract_Final_Results.xlsx")
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        processed_set = set()
        all_valid:   list = []
        all_invalid: list = []

        import json
        # Load processed filenames
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, "r") as f:
                    processed_set = set(json.load(f))
            except: pass

        # Load existing records to resume building the Excel
        import pandas as pd
        if backup_path.exists():
            try:
                with open(backup_path, "r") as f:
                    data = json.load(f)
                    all_valid   = [QuestionPaperRecord(**r) for r in data.get("valid", [])]
                    all_invalid = [QuestionPaperRecord(**r) for r in data.get("invalid", [])]
                tracker.log(f"🔄 Resumed: {len(processed_set)} files done, {len(all_valid)} rows reloaded from backup.")
            except Exception as e:
                logger.error(f"Failed to load backup: {e}")

        # SAFETY: If backup is empty but Excel exists, try to reload from Excel
        if not all_valid and final_excel_path.exists():
            try:
                tracker.log("📂 Backup empty. Attempting to reload from Excel file…")
                # Read all sheets
                all_sheets = pd.read_excel(final_excel_path, sheet_name=None)
                for sheet_name, df in all_sheets.items():
                    if sheet_name == "_Flagged": continue
                    for _, row in df.iterrows():
                        # Map Excel columns back to schema
                        all_valid.append(QuestionPaperRecord(
                            school=str(row.get("School", "")),
                            course_code=str(row.get("Course Code", "")),
                            course_title=str(row.get("Course Title", "")),
                            year=int(row["Year"]) if pd.notnull(row.get("Year")) else None,
                            month=str(row.get("Month", "")),
                            semester=str(row.get("SEM", "")),
                            source_file=str(row.get("Source File", "")),
                        ))
                tracker.log(f"✅ Recovered {len(all_valid)} rows from Excel.")
            except Exception as e:
                logger.warning(f"Excel recovery skipped: {e}")

        if total == 0:
            tracker.log("⚠️  No PDF files found in the specified Drive folder.")
            tracker.complete()
            return

        tracker.start(total)
        tracker.log(f"✅ Found {total} PDF file(s).")

        # ── 2. Process PDFs Concurrently ───────────────────────────────────
        # We use a ThreadPoolExecutor to handle multiple files at once.
        # Max 3 concurrent files to avoid VRAM congestion on the RTX 3050.
        from concurrent.futures import ThreadPoolExecutor
        import threading
        
        results_lock = threading.Lock()
        gpu_lock = threading.Lock()

        def _process_single_pdf(drive_file):
            if tracker.is_stopped():
                return

            # Skip if already done
            if drive_file.name in processed_set:
                tracker.file_done()
                return

            try:
                tracker.log(f"📄 Start: {drive_file.name}")

                # Download (Fresh client per thread for safety)
                thread_drive = DriveClient(
                    credentials_path=settings.credentials_path,
                    download_dir=settings.downloads_path,
                )
                local_path = thread_drive.download(drive_file)

                # PDF → images
                images = pdf_to_images(local_path, dpi=settings.pdf_dpi)

                # GPU operations (Preprocess & OCR) are locked to prevent VRAM OOM
                with gpu_lock:
                    # Preprocess (now GPU-accelerated internally)
                    processed = [preprocess_image(img) for img in images]

                    # OCR (uses GPU)
                    raw_text = run_ocr_on_images(processed, tesseract_cmd=settings.tesseract_cmd)

                # Extract (now parallelized internally)
                records = extract_records(
                    raw_text,
                    source_file=drive_file.name,
                    hint_school=drive_file.school,
                    hint_degree=drive_file.degree,
                    hint_semester=drive_file.semester,
                    hint_month=drive_file.month,
                    hint_year=drive_file.year,
                )

                # Clean & Validate
                cleaned = [clean_record(r) for r in records]
                valid, invalid = validate_records(cleaned)

                # Thread-safe update of shared results
                with results_lock:
                    all_valid.extend(valid)
                    all_invalid.extend(invalid)
                    processed_set.add(drive_file.name)
                    
                    # Persist checkpoints
                    with open(checkpoint_path, "w") as f:
                        json.dump(list(processed_set), f)
                    with open(backup_path, "w") as f:
                        json.dump({
                            "valid":   [r.model_dump() for r in all_valid],
                            "invalid": [r.model_dump() for r in all_invalid]
                        }, f)
                    
                    # Incremental Save (passing only the new records to append)
                    write_excel(valid, invalid, output_path=final_excel_path, append=True)
                    global _excel_output_path
                    _excel_output_path = final_excel_path

                tracker.add_rows(len(valid))
                tracker.add_records(valid)
                tracker.log(f"✅ Done: {drive_file.name} ({len(valid)} rows)")

            except Exception as exc:
                msg = f"{drive_file.name}: {exc}"
                tracker.add_error(msg)
                logger.error("Error processing %s:\n%s", drive_file.name, traceback.format_exc())
            finally:
                tracker.file_done()
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=3) as executor:
            list(executor.map(_process_single_pdf, pdf_files))

        tracker.set_excel_url("/download")

        # ── 3. Final Excel Polish ────────────────────────────────────────────
        tracker.log("📝 Finalizing Excel file…")
        # Write the full consolidated list one last time to ensure everything is sorted and styled
        _excel_output_path = write_excel(all_valid, all_invalid, output_path=final_excel_path, append=True)
        tracker.log(f"✅ Excel saved → {_excel_output_path.name}")

        tracker.complete()
        tracker.log("🎉 Pipeline complete!")
        logger.info("=== Pipeline complete ===")

    except Exception as exc:
        tracker.add_error(f"Fatal pipeline error: {exc}")
        logger.critical("Fatal error in pipeline:\n%s", traceback.format_exc())
        tracker.complete()


# ---------------------------------------------------------------------------
# Dev runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info",
    )
