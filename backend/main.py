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
from backend.models.schema import ProcessingStatus
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
        if backup_path.exists():
            try:
                with open(backup_path, "r") as f:
                    data = json.load(f)
                    all_valid   = [QuestionPaperRecord(**r) for r in data.get("valid", [])]
                    all_invalid = [QuestionPaperRecord(**r) for r in data.get("invalid", [])]
                tracker.log(f"🔄 Resumed: {len(processed_set)} files already done, {len(all_valid)} rows reloaded.")
            except Exception as e:
                logger.error(f"Failed to load backup: {e}")

        if total == 0:
            tracker.log("⚠️  No PDF files found in the specified Drive folder.")
            tracker.complete()
            return

        tracker.start(total)
        tracker.log(f"✅ Found {total} PDF file(s).")

        # ── 2. Process each PDF ─────────────────────────────────────────────
        for drive_file in pdf_files:
            if tracker.is_stopped():
                tracker.log("🛑 Pipeline stopped by user.")
                break
            
            # Skip if already done
            if drive_file.name in processed_set:
                tracker.file_done()
                continue

            try:
                tracker.set_current_file(drive_file.name)
                tracker.log(f"📄 Processing: {drive_file.name}")

                # Download
                tracker.log(f"  ⬇️  Downloading…")
                local_path = drive.download(drive_file)

                # PDF → images
                tracker.log(f"  🖼️  Converting PDF to images @ {settings.pdf_dpi} DPI…")
                images = pdf_to_images(local_path, dpi=settings.pdf_dpi)

                # Preprocess
                tracker.log(f"  🔧 Preprocessing {len(images)} page(s)…")
                processed = [preprocess_image(img) for img in images]

                # OCR
                tracker.log(f"  🔍 Running OCR (8 cores)…")
                raw_text = run_ocr_on_images(processed, tesseract_cmd=settings.tesseract_cmd)

                # Extract
                tracker.log(f"  📊 Extracting structured data (AI active)…")
                records = extract_records(
                    raw_text,
                    source_file=drive_file.name,
                    hint_school=drive_file.school,
                    hint_degree=drive_file.degree,
                    hint_semester=drive_file.semester,
                    hint_month=drive_file.month,
                    hint_year=drive_file.year,
                )

                # Clean
                cleaned = [clean_record(r) for r in records]

                # Validate
                valid, invalid = validate_records(cleaned)
                all_valid.extend(valid)
                all_invalid.extend(invalid)

                tracker.add_rows(len(valid))
                tracker.add_records(valid) # Push to UI preview
                tracker.log(
                    f"  ✅ Extracted {len(valid)} valid row(s), "
                    f"{len(invalid)} flagged."
                )

                # Mark as done and Save Backup
                processed_set.add(drive_file.name)
                with open(checkpoint_path, "w") as f:
                    json.dump(list(processed_set), f)
                
                with open(backup_path, "w") as f:
                    json.dump({
                        "valid":   [r.model_dump() for r in all_valid],
                        "invalid": [r.model_dump() for r in all_invalid]
                    }, f)

            except Exception as exc:
                msg = f"{drive_file.name}: {exc}"
                tracker.add_error(msg)
                logger.error("Error processing %s:\n%s", drive_file.name, traceback.format_exc())
            finally:
                tracker.file_done()

                # Incremental Save to the FIXED filename
                if all_valid or all_invalid:
                    write_excel(all_valid, all_invalid, output_path=final_excel_path)
                    global _excel_output_path
                    _excel_output_path = final_excel_path
                    tracker.set_excel_url("/download")
                    tracker.log(f"🔄 Excel updated: {len(all_valid)} rows total.")

        # ── 3. Write Excel ───────────────────────────────────────────────────
        tracker.log("📝 Writing Excel file…")
        _excel_output_path = write_excel(all_valid, all_invalid)
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
