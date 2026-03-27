"""API integration tests using an in-memory SQLite database."""

import uuid

from fastapi.testclient import TestClient

from tests.conftest import MINIMAL_WAV


class TestHealth:
    def test_returns_200_and_ok(self, client: TestClient) -> None:
        """GET /health should respond with 200 and {"status": "ok"}."""
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestUpload:
    def test_no_file_returns_422(self, client: TestClient) -> None:
        """POST /upload with no file field should return 422 (missing required field)."""
        response = client.post("/upload")

        assert response.status_code == 422

    def test_unsupported_extension_returns_400(self, client: TestClient) -> None:
        """POST /upload with a disallowed extension should return 400."""
        response = client.post(
            "/upload",
            files={"file": ("audio.txt", b"not audio", "text/plain")},
        )

        assert response.status_code == 400
        assert ".txt" in response.json()["detail"]

    def test_valid_wav_returns_202_with_job_id(self, client: TestClient) -> None:
        """POST /upload with a valid WAV file should return 202 and a job_id UUID."""
        response = client.post(
            "/upload",
            files={"file": ("test.wav", MINIMAL_WAV, "audio/wav")},
        )

        assert response.status_code == 202
        body = response.json()
        assert "id" in body
        # Ensure the returned id is a valid UUID
        uuid.UUID(body["id"])
        assert body["filename"] == "test.wav"
        assert body["status"] == "pending"

    def test_valid_wav_creates_pending_job(self, client: TestClient) -> None:
        """The job returned immediately should have status 'pending'."""
        response = client.post(
            "/upload",
            files={"file": ("speech.wav", MINIMAL_WAV, "audio/wav")},
        )

        assert response.status_code == 202
        assert response.json()["status"] == "pending"


class TestGetJob:
    def test_unknown_job_id_returns_404(self, client: TestClient) -> None:
        """GET /jobs/{id} for a non-existent job should return 404."""
        missing_id = uuid.uuid4()
        response = client.get(f"/jobs/{missing_id}")

        assert response.status_code == 404

    def test_invalid_uuid_returns_422(self, client: TestClient) -> None:
        """GET /jobs/not-a-uuid should return 422 (path param validation)."""
        response = client.get("/jobs/not-a-uuid")

        assert response.status_code == 422

    def test_known_job_returns_200(self, client: TestClient) -> None:
        """GET /jobs/{id} for an existing job should return 200 with job and segments."""
        upload = client.post(
            "/upload",
            files={"file": ("clip.wav", MINIMAL_WAV, "audio/wav")},
        )
        job_id = upload.json()["id"]

        response = client.get(f"/jobs/{job_id}")

        assert response.status_code == 200
        body = response.json()
        assert body["job"]["id"] == job_id
        assert isinstance(body["segments"], list)


class TestSearch:
    def test_unknown_job_id_returns_404(self, client: TestClient) -> None:
        """GET /jobs/{id}/search for a non-existent job should return 404."""
        missing_id = uuid.uuid4()
        response = client.get(f"/jobs/{missing_id}/search", params={"q": "hello"})

        assert response.status_code == 404

    def test_empty_query_returns_400(self, client: TestClient) -> None:
        """GET /jobs/{id}/search?q= with blank q should return 400."""
        upload = client.post(
            "/upload",
            files={"file": ("clip.wav", MINIMAL_WAV, "audio/wav")},
        )
        job_id = upload.json()["id"]

        response = client.get(f"/jobs/{job_id}/search", params={"q": "   "})

        assert response.status_code == 400

    def test_no_matches_returns_empty_list(self, client: TestClient) -> None:
        """Searching a job with no segments should return an empty list."""
        upload = client.post(
            "/upload",
            files={"file": ("clip.wav", MINIMAL_WAV, "audio/wav")},
        )
        job_id = upload.json()["id"]

        response = client.get(f"/jobs/{job_id}/search", params={"q": "hello"})

        assert response.status_code == 200
        assert response.json() == []
