"""
config.py — Central configuration loaded from environment variables.
Copy .env.example → .env and fill in your values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All runtime configuration in one place.
    Values are read from environment variables (case-insensitive).
    A .env file in the project root is also supported.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Google Drive ────────────────────────────────────────────────────────
    google_credentials_path: str = "credentials/service_account.json"
    drive_root_folder_id:    str = ""          # REQUIRED — set in .env

    # ── Tesseract & Poppler ───────────────────────────────────────────────
    # Windows examples: 
    # TESSERACT_CMD: "C:/Program Files/Tesseract-OCR/tesseract.exe"
    # POPPLER_PATH:  "C:/poppler/Library/bin"
    tesseract_cmd: Optional[str] = None
    poppler_path:  Optional[str] = None

    # ── Groq ────────────────────────────────────────────────────────────────
    groq_api_key: Optional[str] = None
    groq_model:   str = "llama-3.1-8b-instant"

    # ── Processing ──────────────────────────────────────────────────────────
    pdf_dpi:         int  = 300
    download_dir:    str  = "downloads"
    output_dir:      str  = "output"

    # ── Server ──────────────────────────────────────────────────────────────
    host:  str = "0.0.0.0"
    port:  int = 8000
    debug: bool = False

    # ── CORS ────────────────────────────────────────────────────────────────
    # Comma-separated list of allowed origins for the frontend
    cors_origins: str = "http://localhost:5500,http://127.0.0.1:5500,http://localhost:8000"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def credentials_path(self) -> Path:
        return Path(self.google_credentials_path)

    @property
    def downloads_path(self) -> Path:
        return Path(self.download_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)


# Module-level singleton — import this everywhere
settings = Settings()
