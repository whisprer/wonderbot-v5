from __future__ import annotations

from typing import List
import math
import time

from .base import SensorObservation, SensorStatus
from ..perception import SpeechTranscriber


class MicrophoneUnavailableError(RuntimeError):
    pass


class SoundDeviceMicrophoneAdapter:
    name = "microphone"

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        window_seconds: float = 0.35,
        rms_threshold: float = 0.03,
        peak_threshold: float = 0.12,
        min_salience: float = 0.10,
        transcriber: SpeechTranscriber | None = None,
        transcript_salience_threshold: float = 0.22,
        transcript_min_chars: int = 4,
        transcript_cooldown_seconds: float = 0.75,
    ) -> None:
        try:
            import sounddevice as sd  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:
            raise MicrophoneUnavailableError(
                "sounddevice and numpy are not installed. Install with: pip install -e .[audio]"
            ) from exc
        self._sd = sd
        self._np = np
        self.sample_rate = sample_rate
        self.channels = channels
        self.window_seconds = window_seconds
        self.rms_threshold = rms_threshold
        self.peak_threshold = peak_threshold
        self.min_salience = min_salience
        self.transcriber = transcriber
        self.transcript_salience_threshold = transcript_salience_threshold
        self.transcript_min_chars = transcript_min_chars
        self.transcript_cooldown_seconds = transcript_cooldown_seconds
        self._prev_rms = 0.0
        self._prev_zcr = 0.0
        self._last_text = ""
        self._last_transcript = ""
        self._last_transcript_at = 0.0

    def record(self, seconds: float | None = None):
        duration = self.window_seconds if seconds is None else seconds
        frames = max(1, int(self.sample_rate * duration))
        recording = self._sd.rec(frames, samplerate=self.sample_rate, channels=self.channels, dtype="float32")
        self._sd.wait()
        return recording

    def poll(self) -> List[SensorObservation]:
        sample = self.record()
        mono = self._to_mono(sample)
        if mono.size == 0:
            return []
        rms = float(math.sqrt(float((mono * mono).mean())))
        peak = float(self._np.max(self._np.abs(mono)))
        signs = mono[:-1] * mono[1:]
        zcr = float((signs < 0).mean()) if signs.size else 0.0
        delta_rms = abs(rms - self._prev_rms)
        delta_zcr = abs(zcr - self._prev_zcr)
        salience = min(1.0, max(rms * 6.0, peak * 4.0, delta_rms * 8.0, delta_zcr * 2.2))
        self._prev_rms = rms
        self._prev_zcr = zcr
        if salience < self.min_salience:
            return []

        texture = "voice-like banding" if 0.03 <= zcr <= 0.22 else ("noisy texture" if zcr > 0.22 else "low-frequency texture")
        if peak >= self.peak_threshold * 1.6:
            event = "a sharp transient"
        elif rms >= self.rms_threshold and 0.03 <= zcr <= 0.22:
            event = "voice-like audio activity"
        elif rms >= self.rms_threshold:
            event = "rising ambient audio"
        else:
            event = "a faint audio change"
        text = f"microphone hears {event} with {texture}."
        metadata = {
            "rms": round(rms, 6),
            "peak": round(peak, 6),
            "zcr": round(zcr, 6),
            "delta_rms": round(delta_rms, 6),
        }
        transcript = self._maybe_transcribe(mono, salience, event, zcr)
        if transcript:
            text = f'microphone hears {event} and catches speech: "{transcript}".'
            metadata["transcript"] = transcript
        if text == self._last_text and salience < 0.45:
            return []
        self._last_text = text
        return [
            SensorObservation(
                source=self.name,
                text=text,
                salience=round(salience, 6),
                metadata=metadata,
            )
        ]

    def status(self) -> SensorStatus:
        detail = "microphone adapter active"
        if self.transcriber is not None:
            detail += f"; speech transcription via {getattr(self.transcriber, 'model_name', 'transcriber')}"
        return SensorStatus(source=self.name, enabled=True, available=True, detail=detail)

    def close(self) -> None:
        return None

    def _to_mono(self, sample):
        data = self._np.asarray(sample, dtype="float32")
        if data.ndim == 1:
            return data
        if data.ndim == 2:
            return data.mean(axis=1)
        return data.reshape(-1)

    def _maybe_transcribe(self, mono, salience: float, event: str, zcr: float) -> str | None:
        if self.transcriber is None or salience < self.transcript_salience_threshold:
            return None
        if event != "voice-like audio activity" and not (0.03 <= zcr <= 0.22 and salience >= self.transcript_salience_threshold * 1.2):
            return None
        now = time.monotonic()
        if self._last_transcript and (now - self._last_transcript_at) < self.transcript_cooldown_seconds:
            return None
        try:
            result = self.transcriber.transcribe(mono, sample_rate=self.sample_rate)
        except Exception:
            return None
        if result is None:
            return None
        text = _clean_transcript(result.text)
        if len(text) < self.transcript_min_chars:
            return None
        if _normalize_text(text) == _normalize_text(self._last_transcript):
            return None
        self._last_transcript = text
        self._last_transcript_at = now
        return text



def _clean_transcript(text: str) -> str:
    return " ".join(str(text).strip().split())



def _normalize_text(text: str) -> str:
    return _clean_transcript(text).lower()
