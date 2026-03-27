"""Pytest configuration and shared fixtures.

Sets DATABASE_URL to an in-memory SQLite database *before* any app module is
imported so that app/database.py never tries to reach PostgreSQL.
"""

import io
import os
import sys
import types
import wave

# Must be set before any app import so database.py picks up the SQLite URL
# instead of the PostgreSQL URL from .env.  load_dotenv() will not override
# an env var that is already present in os.environ.
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ.setdefault("HUGGINGFACE_TOKEN", "test-token")

# ---------------------------------------------------------------------------
# Stub heavy ML dependencies so they don't need to be installed to run tests.
# These must be inserted into sys.modules before any app module is imported.
# ---------------------------------------------------------------------------

def _make_stub(name: str) -> types.ModuleType:
    """Return an empty module stub registered under *name*."""
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# whisper
_whisper = _make_stub("whisper")
_whisper.load_model = lambda *a, **kw: None  # type: ignore[attr-defined]

# pyannote.audio (package + sub-module)
_pyannote = _make_stub("pyannote")
_pyannote_audio = _make_stub("pyannote.audio")
_pyannote.audio = _pyannote_audio  # type: ignore[attr-defined]
_pyannote_audio.Pipeline = type("Pipeline", (), {"from_pretrained": staticmethod(lambda *a, **kw: None)})  # type: ignore[attr-defined]

from typing import Generator
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

import app.database as db_module
from app.database import Base, get_db
from app.main import app

# ---------------------------------------------------------------------------
# SQLite test engine
# ---------------------------------------------------------------------------

# StaticPool ensures every call to connect() returns the *same* underlying
# connection, which is required for a shared in-memory SQLite database.
TEST_ENGINE = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(
    autocommit=False, autoflush=False, bind=TEST_ENGINE
)

# Patch the module-level objects so that _run_transcription's
# `from app.database import SessionLocal` also gets the test session factory.
db_module.engine = TEST_ENGINE
db_module.SessionLocal = TestingSessionLocal


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_wav_bytes() -> bytes:
    """Return a minimal valid WAV file (10 ms of silence, mono 16-bit 16 kHz)."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 160)
    return buf.getvalue()


MINIMAL_WAV: bytes = _make_wav_bytes()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _create_tables():
    """Create all tables before each test and drop them afterwards."""
    Base.metadata.create_all(bind=TEST_ENGINE)
    yield
    Base.metadata.drop_all(bind=TEST_ENGINE)


@pytest.fixture()
def db_session() -> Generator[Session, None, None]:
    """Yield a SQLAlchemy session backed by the in-memory SQLite database."""
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def mock_transcriber(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Patch get_transcription_service so no ML models are loaded.

    Returns a MagicMock whose .transcribe() returns an empty list, preventing
    any attempt to load Whisper or pyannote during tests.
    """
    mock_service = MagicMock()
    mock_service.transcribe.return_value = []
    monkeypatch.setattr("app.main.get_transcription_service", lambda: mock_service)
    return mock_service


@pytest.fixture()
def client(mock_transcriber: MagicMock) -> Generator[TestClient, None, None]:
    """Return a TestClient with the database dependency overridden to use SQLite."""

    def override_get_db() -> Generator[Session, None, None]:
        db = TestingSessionLocal()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app, raise_server_exceptions=True) as test_client:
        yield test_client
    app.dependency_overrides.clear()
