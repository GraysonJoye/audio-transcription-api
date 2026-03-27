"""Pydantic response schemas for the transcription API."""

import uuid
from datetime import datetime

from pydantic import BaseModel


class JobResponse(BaseModel):
    """Summary of a transcription job."""

    id: uuid.UUID
    filename: str
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class SegmentResponse(BaseModel):
    """A single diarized transcript segment."""

    speaker_label: str
    start_time: float
    end_time: float
    text: str

    model_config = {"from_attributes": True}


class TranscriptResponse(BaseModel):
    """Full transcript: job metadata plus all segments."""

    job: JobResponse
    segments: list[SegmentResponse]
