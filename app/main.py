"""FastAPI application entry point."""

import logging
import traceback
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, UploadFile
from sqlalchemy.orm import Session

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

from app.database import create_tables, get_db
from app.models import JobStatus, TranscriptionJob, TranscriptSegment
from app.schemas import JobResponse, SegmentResponse, TranscriptResponse
from app.transcriber import TranscriptionService

load_dotenv()

UPLOADS_DIR = Path("uploads")
ALLOWED_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg"}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create database tables on startup."""
    create_tables()
    yield


app = FastAPI(title="Audio Transcription API", lifespan=lifespan)

# Initialise once at module load so the heavy models are not reloaded per request.
_transcription_service: TranscriptionService | None = None


def get_transcription_service() -> TranscriptionService:
    """Return a shared TranscriptionService instance, creating it on first call."""
    global _transcription_service
    if _transcription_service is None:
        _transcription_service = TranscriptionService()
    return _transcription_service


# ---------------------------------------------------------------------------
# Background task
# ---------------------------------------------------------------------------

def _run_transcription(job_id: uuid.UUID, audio_path: str) -> None:
    """Transcribe *audio_path* and persist results; updates job status throughout.

    Runs in a background thread via FastAPI's BackgroundTasks.

    Args:
        job_id: UUID of the TranscriptionJob row to update.
        audio_path: Filesystem path to the saved audio file.
    """
    # Background tasks need their own DB session.
    from app.database import SessionLocal

    logger.info("Background transcription started for job %s (%s)", job_id, audio_path)
    db: Session = SessionLocal()
    try:
        job: TranscriptionJob | None = db.get(TranscriptionJob, job_id)
        if job is None:
            logger.warning("Job %s not found in database; aborting.", job_id)
            return

        job.status = JobStatus.processing
        db.commit()
        logger.info("Job %s status → processing", job_id)

        service = get_transcription_service()
        segments = service.transcribe(audio_path)
        logger.info("Job %s transcription produced %d segment(s)", job_id, len(segments))

        for seg in segments:
            db.add(
                TranscriptSegment(
                    job_id=job_id,
                    speaker_label=seg["speaker"],
                    start_time=seg["start_time"],
                    end_time=seg["end_time"],
                    text=seg["text"],
                )
            )

        job.status = JobStatus.complete
        db.commit()
        logger.info("Job %s status → complete", job_id)

    except Exception as exc:
        logger.error(
            "Job %s failed with error: %s\n%s",
            job_id,
            exc,
            traceback.format_exc(),
        )
        db.rollback()
        job = db.get(TranscriptionJob, job_id)
        if job is not None:
            job.status = JobStatus.failed
            db.commit()
        raise RuntimeError(f"Transcription failed for job {job_id}: {exc}") from exc
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Return service liveness status."""
    return {"status": "ok"}


@app.post("/upload", response_model=JobResponse, status_code=202)
async def upload_audio(
    file: UploadFile,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
) -> TranscriptionJob:
    """Accept an audio file, persist it, and queue transcription in the background.

    Args:
        file: Uploaded audio file (mp3, wav, m4a, ogg).
        background_tasks: FastAPI background task runner.
        db: Injected database session.

    Returns:
        The newly created TranscriptionJob (status will be ``pending``).

    Raises:
        HTTPException 400: If the file extension is not allowed.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    job_id = uuid.uuid4()
    # Prefix the saved filename with the job UUID to avoid collisions.
    safe_filename = f"{job_id}{suffix}"
    audio_path = UPLOADS_DIR / safe_filename

    UPLOADS_DIR.mkdir(exist_ok=True)
    contents = await file.read()
    audio_path.write_bytes(contents)

    job = TranscriptionJob(
        id=job_id,
        filename=file.filename or safe_filename,
        status=JobStatus.pending,
    )
    db.add(job)
    db.commit()
    db.refresh(job)

    background_tasks.add_task(_run_transcription, job_id, str(audio_path))

    return job


@app.get("/jobs/{job_id}", response_model=TranscriptResponse)
def get_job(
    job_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> TranscriptResponse:
    """Return job status and, once complete, all transcript segments.

    Args:
        job_id: UUID of the transcription job.
        db: Injected database session.

    Returns:
        Job metadata and segment list.

    Raises:
        HTTPException 404: If no job with *job_id* exists.
    """
    job: TranscriptionJob | None = db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    return TranscriptResponse(
        job=JobResponse.model_validate(job),
        segments=[SegmentResponse.model_validate(s) for s in job.segments],
    )


@app.get("/jobs/{job_id}/search", response_model=list[SegmentResponse])
def search_transcript(
    job_id: uuid.UUID,
    q: str,
    db: Session = Depends(get_db),
) -> list[TranscriptSegment]:
    """Search transcript segments for *q* using a case-insensitive ILIKE query.

    Args:
        job_id: UUID of the transcription job to search within.
        q: Keyword or phrase to search for.
        db: Injected database session.

    Returns:
        Matching segments ordered by start time.

    Raises:
        HTTPException 404: If no job with *job_id* exists.
        HTTPException 400: If *q* is empty.
    """
    if not q.strip():
        raise HTTPException(status_code=400, detail="Search query 'q' must not be empty.")

    job: TranscriptionJob | None = db.get(TranscriptionJob, job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    matches: list[TranscriptSegment] = (
        db.query(TranscriptSegment)
        .filter(
            TranscriptSegment.job_id == job_id,
            TranscriptSegment.text.ilike(f"%{q}%"),
        )
        .order_by(TranscriptSegment.start_time)
        .all()
    )
    return matches
