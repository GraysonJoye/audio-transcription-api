"""Audio transcription and speaker diarization service."""

import logging
import os
from typing import Any

import torch
import torchaudio
import whisper
from dotenv import load_dotenv
from pyannote.audio import Pipeline

load_dotenv()

logger = logging.getLogger(__name__)

_TARGET_SAMPLE_RATE = 16_000


class TranscriptionService:
    """Combines Whisper transcription with pyannote speaker diarization."""

    def __init__(self) -> None:
        """Load the Whisper model and initialise the pyannote diarization pipeline."""
        huggingface_token = os.environ.get("HUGGINGFACE_TOKEN")
        if not huggingface_token:
            raise RuntimeError(
                "HUGGINGFACE_TOKEN environment variable is not set. "
                "Create a token at https://huggingface.co/settings/tokens and "
                "accept the pyannote/speaker-diarization-3.1 model licence."
            )

        try:
            self.whisper_model = whisper.load_model("base")
            logger.info("Whisper 'base' model loaded successfully.")
        except Exception as exc:
            raise RuntimeError(f"Failed to load Whisper 'base' model: {exc}") from exc

        try:
            self.diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                token=huggingface_token,
            )
            logger.info("pyannote diarization pipeline loaded successfully.")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load pyannote speaker-diarization-3.1 pipeline: {exc}. "
                "Ensure you have accepted the model licence on Hugging Face."
            ) from exc

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def transcribe(self, audio_path: str) -> list[dict[str, Any]]:
        """Transcribe *audio_path* and label each segment with a speaker.

        Steps:
        1. Load the audio file into a waveform tensor (avoids any ffmpeg
           subprocess calls from Whisper or pyannote).
        2. Run Whisper on the pre-loaded numpy array.
        3. Run pyannote diarization on the pre-loaded waveform tensor.
        4. Merge: assign each Whisper segment the speaker whose turn has the
           greatest overlap with that segment's time range.

        Args:
            audio_path: Absolute or relative path to the audio file.

        Returns:
            A list of dicts, each with keys:
            ``speaker``, ``start_time``, ``end_time``, ``text``.
        """
        logger.info("Loading audio from '%s'", audio_path)
        waveform, sample_rate = self._load_audio(audio_path)
        duration = waveform.shape[-1] / sample_rate
        logger.info(
            "Audio loaded: %d samples at %d Hz (%.1f s)",
            waveform.shape[-1],
            sample_rate,
            duration,
        )

        whisper_segments = self._run_whisper(waveform)
        diarization = self._run_diarization(waveform, sample_rate)
        speaker_turns = self._extract_speaker_turns(diarization)
        return self._merge(whisper_segments, speaker_turns)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_audio(audio_path: str) -> tuple[torch.Tensor, int]:
        """Load an audio file and return a mono 16 kHz waveform tensor.

        Uses ``torchaudio.load`` which ships with bundled ffmpeg binaries on
        Windows, so no separate ffmpeg installation is required.  The returned
        tensor is a float32 mono waveform of shape ``(1, N)`` at 16 kHz.

        Args:
            audio_path: Path to the audio file (wav, mp3, m4a, ogg, flac …).

        Returns:
            Tuple of ``(waveform, sample_rate)`` where ``waveform`` has shape
            ``(1, N)`` and ``sample_rate`` is ``_TARGET_SAMPLE_RATE`` (16000).

        Raises:
            RuntimeError: If the file cannot be loaded or decoded.
        """
        try:
            waveform, orig_sr = torchaudio.load(audio_path)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load audio file '{audio_path}': {exc}. "
                "Ensure the file is a valid audio format (wav, mp3, m4a, ogg)."
            ) from exc

        logger.debug(
            "Raw audio: %d channel(s), %d Hz, %d samples",
            waveform.shape[0],
            orig_sr,
            waveform.shape[-1],
        )

        # Convert to mono by averaging channels.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # Resample to 16 kHz if needed.
        if orig_sr != _TARGET_SAMPLE_RATE:
            logger.debug(
                "Resampling audio from %d Hz to %d Hz", orig_sr, _TARGET_SAMPLE_RATE
            )
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sr, new_freq=_TARGET_SAMPLE_RATE
            )
            waveform = resampler(waveform)

        # Ensure float32 (torchaudio may return int16 for some formats).
        waveform = waveform.to(torch.float32)

        return waveform, _TARGET_SAMPLE_RATE

    def _run_whisper(self, waveform: torch.Tensor) -> list[dict[str, Any]]:
        """Run Whisper on a pre-loaded waveform tensor.

        Passes the audio as a 1-D float32 numpy array so Whisper does not
        attempt to invoke ffmpeg internally.

        Args:
            waveform: Mono waveform tensor of shape ``(1, N)`` at 16 kHz.

        Returns:
            List of Whisper segment dicts containing at minimum
            ``start``, ``end``, and ``text``.

        Raises:
            RuntimeError: If Whisper fails to process the audio.
        """
        # Whisper expects a 1-D float32 numpy array sampled at 16 kHz.
        audio_np = waveform.squeeze(0).numpy().astype("float32")
        logger.info(
            "Running Whisper transcription on %.1f s of audio",
            len(audio_np) / _TARGET_SAMPLE_RATE,
        )

        try:
            result = self.whisper_model.transcribe(
                audio_np,
                word_timestamps=True,
                verbose=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Whisper transcription failed: {exc}") from exc

        segments: list[dict[str, Any]] = result.get("segments", [])
        logger.info("Whisper returned %d segment(s)", len(segments))
        if not segments:
            return []
        return segments

    def _run_diarization(self, waveform: torch.Tensor, sample_rate: int) -> Any:
        """Run pyannote diarization on a pre-loaded waveform tensor.

        Passes the audio as ``{"waveform": tensor, "sample_rate": int}`` so
        pyannote does not attempt to decode the file from disk (which would
        trigger its own audio-backend lookup and potentially fail on Windows).

        Args:
            waveform: Mono waveform tensor of shape ``(1, N)``.
            sample_rate: Sample rate in Hz (should be 16000).

        Returns:
            A pyannote diarization result (``Annotation`` or ``DiarizeOutput``
            depending on the installed pyannote version).

        Raises:
            RuntimeError: If diarization fails.
        """
        logger.info("Running pyannote speaker diarization")
        try:
            return self.diarization_pipeline(
                {"waveform": waveform, "sample_rate": sample_rate}
            )
        except Exception as exc:
            raise RuntimeError(f"Speaker diarization failed: {exc}") from exc

    @staticmethod
    def _extract_speaker_turns(diarization: Any) -> list[dict[str, Any]]:
        """Convert a pyannote diarization result into a plain list of speaker turns.

        Handles both ``pyannote.audio.pipelines.speaker_diarization.DiarizeOutput``
        (a dataclass whose ``speaker_diarization`` field is a
        ``pyannote.core.Annotation``) and a bare ``pyannote.core.Annotation``
        returned by older pipeline versions.  Both expose ``itertracks``.

        Args:
            diarization: Diarization result returned by the pipeline.

        Returns:
            List of dicts with keys ``start``, ``end``, ``speaker``.
        """
        # DiarizeOutput is a dataclass; unwrap the Annotation it contains.
        annotation = (
            diarization.speaker_diarization
            if hasattr(diarization, "speaker_diarization")
            else diarization
        )

        turns: list[dict[str, Any]] = []
        for segment, _track, speaker in annotation.itertracks(yield_label=True):
            turns.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "speaker": speaker,
                }
            )
        logger.info("Diarization identified %d speaker turn(s)", len(turns))
        return turns

    @staticmethod
    def _overlap(
        seg_start: float,
        seg_end: float,
        turn_start: float,
        turn_end: float,
    ) -> float:
        """Return the duration of overlap between two time intervals.

        Args:
            seg_start: Start of the Whisper segment.
            seg_end: End of the Whisper segment.
            turn_start: Start of the speaker turn.
            turn_end: End of the speaker turn.

        Returns:
            Overlap duration in seconds (0.0 if there is no overlap).
        """
        return max(0.0, min(seg_end, turn_end) - max(seg_start, turn_start))

    def _assign_speaker(
        self,
        seg_start: float,
        seg_end: float,
        speaker_turns: list[dict[str, Any]],
    ) -> str:
        """Find the speaker with the most overlap for a given time range.

        Falls back to ``"UNKNOWN"`` when no speaker turn overlaps at all.

        Args:
            seg_start: Start time of the Whisper segment.
            seg_end: End time of the Whisper segment.
            speaker_turns: List of speaker-turn dicts from :meth:`_extract_speaker_turns`.

        Returns:
            Speaker label string.
        """
        best_speaker = "UNKNOWN"
        best_overlap = 0.0

        for turn in speaker_turns:
            overlap = self._overlap(seg_start, seg_end, turn["start"], turn["end"])
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = turn["speaker"]

        return best_speaker

    def _merge(
        self,
        whisper_segments: list[dict[str, Any]],
        speaker_turns: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge Whisper segments with speaker diarization results.

        Each Whisper segment is assigned the speaker whose turn overlaps it the
        most.  Consecutive segments with the same speaker are **not** merged so
        that fine-grained timestamps are preserved for the caller.

        Args:
            whisper_segments: Raw Whisper segment dicts.
            speaker_turns: Speaker-turn dicts from :meth:`_extract_speaker_turns`.

        Returns:
            List of dicts with keys ``speaker``, ``start_time``, ``end_time``,
            ``text``.
        """
        merged: list[dict[str, Any]] = []

        for seg in whisper_segments:
            start: float = seg["start"]
            end: float = seg["end"]
            text: str = seg["text"].strip()

            if not text:
                continue

            speaker = self._assign_speaker(start, end, speaker_turns)

            merged.append(
                {
                    "speaker": speaker,
                    "start_time": start,
                    "end_time": end,
                    "text": text,
                }
            )

        logger.info("Merged into %d final segment(s)", len(merged))
        return merged
