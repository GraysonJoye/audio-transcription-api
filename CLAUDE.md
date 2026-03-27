# Audio Transcription API

## Project Overview
A FastAPI backend that transcribes audio files and identifies speakers using
OpenAI Whisper (local) and pyannote.audio. Results are stored in PostgreSQL.

## Tech Stack
- Python 3.11
- FastAPI
- PostgreSQL (via Docker)
- SQLAlchemy ORM
- OpenAI Whisper (local, free)
- pyannote.audio 3.1

## Commands
- Start services: docker-compose up -d
- Run app: uvicorn app.main:app --reload
- Run tests: pytest

## Style Guidelines
- Use type hints everywhere
- Follow PEP 8
- Write docstrings for all functions
- Keep environment variables in .env (never commit this file)
