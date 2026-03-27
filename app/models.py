"""SQLAlchemy ORM models for transcription jobs and transcript segments."""

import uuid
from datetime import datetime, timezone
from enum import Enum as PyEnum

from sqlalchemy import DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class JobStatus(str, PyEnum):
    """Allowed status values for a TranscriptionJob."""

    pending = "pending"
    processing = "processing"
    complete = "complete"
    failed = "failed"


def _now() -> datetime:
    """Return the current UTC datetime."""
    return datetime.now(timezone.utc)


class TranscriptionJob(Base):
    """Represents a single audio transcription request."""

    __tablename__ = "transcription_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    filename: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(
        String, nullable=False, default=JobStatus.pending
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now, onupdate=_now
    )

    segments: Mapped[list["TranscriptSegment"]] = relationship(
        "TranscriptSegment", back_populates="job", cascade="all, delete-orphan"
    )


class TranscriptSegment(Base):
    """A single speaker-labelled segment produced by diarization."""

    __tablename__ = "transcript_segments"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("transcription_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    speaker_label: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_now
    )

    job: Mapped["TranscriptionJob"] = relationship(
        "TranscriptionJob", back_populates="segments"
    )
