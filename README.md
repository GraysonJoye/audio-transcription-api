# Audio Transcription API

A REST API that converts audio recordings into text transcripts and identifies who is speaking at each moment.

---

## Features

- **Speech-to-Text Transcription** — Converts `.mp3`, `.wav`, `.m4a`, and `.ogg` audio files into written text using OpenAI Whisper, which runs entirely on your machine (no external API calls or fees).
- **Speaker Diarization** — Automatically labels each segment of a transcript with the speaker who said it (e.g. "Speaker 1", "Speaker 2"), making it easy to follow multi-person conversations.
- **Asynchronous Processing** — Audio files are queued in the background after upload. You receive an immediate response with a job ID, then poll for results when ready — no waiting around for long files to process.
- **Keyword Search** — Search within any completed transcript to find every moment a word or phrase was spoken, along with the speaker and timestamp.
- **Persistent Storage** — All jobs and transcripts are saved to a PostgreSQL database, so results are available any time after processing completes.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | [FastAPI](https://fastapi.tiangolo.com/) |
| Speech-to-Text | [OpenAI Whisper](https://github.com/openai/whisper) (runs locally) |
| Speaker Identification | [pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio) |
| Database | PostgreSQL 15 |
| ORM | SQLAlchemy |
| Containerization | Docker / Docker Compose |
| Language | Python 3.11 |

---

## Getting Started

### Prerequisites

Before you begin, make sure you have installed:

- [Python 3.11](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/)
- A [Hugging Face](https://huggingface.co/) account (free) — required to download the speaker diarization model

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd audio-transcription-api
```

### 2. Set up environment variables

Copy the example environment file and fill in your values:

```bash
cp .env.example .env
```

Open `.env` and set the following:

```env
DATABASE_URL=postgresql://postgres:password@localhost:5433/transcription_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=password
POSTGRES_DB=transcription_db
HUGGINGFACE_TOKEN=your_token_here
```

To get your Hugging Face token:
1. Create a free account at [huggingface.co](https://huggingface.co/)
2. Go to **Settings > Access Tokens** and create a new token
3. Accept the terms for the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1) model

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Note:** The `torch` and `torchaudio` packages are large. The first install may take several minutes.

### 4. Start the database

```bash
docker-compose up -d
```

This starts a PostgreSQL 15 instance on port `5433`.

### 5. Run the API

```bash
uvicorn app.main:app --reload
```

The API will be available at `http://localhost:8000`.

You can explore the interactive API docs at `http://localhost:8000/docs`.

### 6. Run tests

```bash
pytest
```

---

## API Endpoints

### `GET /health`

Check that the service is running.

**Response**
```json
{
  "status": "ok"
}
```

---

### `POST /upload`

Upload an audio file to be transcribed. Processing happens in the background — this endpoint returns immediately with a job ID.

**Accepted formats:** `.mp3`, `.wav`, `.m4a`, `.ogg`

**Request**

Send the file as `multipart/form-data` with the field name `file`.

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@interview.mp3"
```

**Response** — `202 Accepted`
```json
{
  "id": "3f1b2c4d-5678-4abc-9def-0123456789ab",
  "filename": "interview.mp3",
  "status": "pending",
  "created_at": "2026-03-27T10:30:45.123456+00:00"
}
```

Save the `id` — you will need it to retrieve results.

---

### `GET /jobs/{job_id}`

Retrieve the status and transcript for a job. Once the job status is `complete`, the `segments` array is populated with the full transcript.

**Job statuses:**

| Status | Meaning |
|---|---|
| `pending` | Uploaded, waiting to be processed |
| `processing` | Transcription and speaker identification in progress |
| `complete` | Finished — transcript is ready |
| `failed` | An error occurred during processing |

**Example request**
```bash
curl http://localhost:8000/jobs/3f1b2c4d-5678-4abc-9def-0123456789ab
```

**Response**
```json
{
  "job": {
    "id": "3f1b2c4d-5678-4abc-9def-0123456789ab",
    "filename": "interview.mp3",
    "status": "complete",
    "created_at": "2026-03-27T10:30:45.123456+00:00"
  },
  "segments": [
    {
      "speaker_label": "Speaker 1",
      "start_time": 0.5,
      "end_time": 4.1,
      "text": "Welcome to the interview. Can you tell us about yourself?"
    },
    {
      "speaker_label": "Speaker 2",
      "start_time": 4.8,
      "end_time": 9.3,
      "text": "Sure, I have been working in software engineering for five years."
    }
  ]
}
```

---

### `GET /jobs/{job_id}/search?q={keyword}`

Search a completed transcript for a keyword or phrase. Returns every segment where the term appears, along with the speaker and timestamps.

**Query parameters:**

| Parameter | Required | Description |
|---|---|---|
| `q` | Yes | The word or phrase to search for (case-insensitive) |

**Example request**
```bash
curl "http://localhost:8000/jobs/3f1b2c4d-5678-4abc-9def-0123456789ab/search?q=software"
```

**Response**
```json
[
  {
    "speaker_label": "Speaker 2",
    "start_time": 4.8,
    "end_time": 9.3,
    "text": "Sure, I have been working in software engineering for five years."
  }
]
```

---

## Project Structure

```
audio-transcription-api/
├── app/
│   ├── main.py          # API routes and application entry point
│   ├── models.py        # Database table definitions (SQLAlchemy)
│   ├── schemas.py       # Request and response data shapes (Pydantic)
│   ├── database.py      # Database connection setup
│   └── transcriber.py   # Whisper + pyannote audio processing logic
├── tests/
│   ├── conftest.py      # Shared test fixtures
│   └── test_api.py      # API endpoint tests
├── uploads/             # Uploaded audio files (not committed to git)
├── docker-compose.yml   # PostgreSQL database service
├── requirements.txt     # Python dependencies
├── .env.example         # Template for environment variables
└── CLAUDE.md            # Project instructions for AI assistants
```
