"""
drive_client.py — Google Drive API integration.
Recursively traverses a folder, yields PDF file metadata,
and downloads files to a local temp directory.

Folder hierarchy expected:
    <root>/
        <School>/
            <Degree>/
                <Semester>/
                    *.pdf
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Generator

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from backend.utils.logger import get_logger

logger = get_logger("drive_client")

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


class DriveFile:
    """Lightweight descriptor for a Drive PDF file."""

    __slots__ = ("file_id", "name", "school", "degree", "semester", "month", "year", "local_path")

    def __init__(
        self,
        file_id: str,
        name: str,
        school: str = "",
        degree: str = "",
        semester: str = "",
        month: str = "",
        year: str = "",
        local_path: Path | None = None,
    ) -> None:
        self.file_id    = file_id
        self.name       = name
        self.school     = school
        self.degree     = degree
        self.semester   = semester
        self.month      = month
        self.year       = year
        self.local_path = local_path

    def __repr__(self) -> str:
        return (
            f"<DriveFile {self.name!r} school={self.school!r} "
            f"degree={self.degree!r} sem={self.semester!r} "
            f"month={self.month!r} year={self.year!r}>"
        )


class DriveClient:
    """
    Wraps Google Drive API v3.

    Parameters
    ----------
    credentials_path : str | Path
        Path to the service-account JSON key file.
    download_dir : str | Path
        Local directory where PDFs are saved.
    """

    def __init__(
        self,
        credentials_path: str | Path,
        download_dir: str | Path,
    ) -> None:
        self._creds_path  = Path(credentials_path)
        self._download_dir = Path(download_dir)
        self._download_dir.mkdir(parents=True, exist_ok=True)
        self._service = self._build_service()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_service(self):
        creds = service_account.Credentials.from_service_account_file(
            str(self._creds_path), scopes=SCOPES
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        logger.info("Google Drive service authenticated via service account.")
        return service

    def _list_children(self, folder_id: str) -> list[dict]:
        """Return all children of a folder (files + subfolders)."""
        results = []
        page_token = None
        query = f"'{folder_id}' in parents and trashed = false"
        while True:
            response = (
                self._service.files()
                .list(
                    q=query,
                    spaces="drive",
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageToken=page_token,
                    pageSize=1000,
                )
                .execute()
            )
            results.extend(response.get("files", []))
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return results

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def iter_pdfs(
        self,
        root_folder_id: str,
    ) -> Generator[DriveFile, None, None]:
        """
        Recursively walk the Drive folder tree.
        Yields a DriveFile for every PDF encountered.

        Hierarchy (up to 3 levels deep below root):
            root/School/Degree/Semester/file.pdf
        """
        logger.info("Traversing Drive folder: %s", root_folder_id)

        def _walk(
            folder_id: str,
            depth: int,
            school: str,
            degree: str,
            semester: str,
        ) -> Generator[DriveFile, None, None]:
            children = self._list_children(folder_id)
            for child in children:
                mime  = child["mimeType"]
                name  = child["name"]
                fid   = child["id"]

                if mime == "application/vnd.google-apps.folder":
                    # Assign hierarchy labels based on depth
                    if depth == 0:
                        yield from _walk(fid, depth + 1, name, "", "")
                    elif depth == 1:
                        yield from _walk(fid, depth + 1, school, name, "")
                    elif depth == 2:
                        yield from _walk(fid, depth + 1, school, degree, name)
                    else:
                        # Deeper nesting — keep labels and recurse
                        yield from _walk(fid, depth + 1, school, degree, semester)

                elif mime == "application/pdf" or name.lower().endswith(".pdf"):
                    # Attempt to parse metadata from filename
                    # Format: SOB-III-SEM-DEC-2022-BBA.pdf
                    f_school, f_sem, f_month, f_year, f_degree = school, semester, "", "", degree
                    
                    # Split by dash and remove .pdf
                    parts = name.rsplit(".", 1)[0].split("-")
                    if len(parts) >= 5:
                        # Try to match the pattern: SCHOOL-SEM-SEM-MONTH-YEAR-DEGREE
                        # Example: SOB-III-SEM-DEC-2022-BBA.pdf (6 parts)
                        # Example: SOB-III-SEM-DEC-2022.pdf (5 parts)
                        f_school = parts[0]
                        f_sem    = parts[1]
                        # parts[2] is usually "SEM"
                        f_month  = parts[3]
                        f_year   = parts[4]
                        if len(parts) >= 6:
                            f_degree = parts[5]

                    yield DriveFile(
                        file_id=fid,
                        name=name,
                        school=f_school,
                        degree=f_degree,
                        semester=f_sem,
                        month=f_month,
                        year=f_year,
                    )

        yield from _walk(root_folder_id, 0, "", "", "")

    def download(self, drive_file: DriveFile) -> Path:
        """
        Download a DriveFile to the local download directory.
        Returns the local Path.
        """
        local_path = self._download_dir / drive_file.name
        if local_path.exists():
            logger.debug("Cache hit — skipping download: %s", drive_file.name)
            drive_file.local_path = local_path
            return local_path

        logger.info("Downloading: %s", drive_file.name)
        request = self._service.files().get_media(fileId=drive_file.file_id)
        buf = io.BytesIO()
        downloader = MediaIoBaseDownload(buf, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

        local_path.write_bytes(buf.getvalue())
        drive_file.local_path = local_path
        logger.info("Saved: %s (%d bytes)", local_path.name, local_path.stat().st_size)
        return local_path
