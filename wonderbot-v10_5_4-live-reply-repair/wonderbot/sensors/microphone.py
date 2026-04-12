from __future__ import annotations

from collections import deque
from typing import List
import math
import threading
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
        device: str = '',
        latency: str = 'high',
        rolling_seconds: float = 4.0,
        transcript_window_seconds: float = 1.8,
        preamp_gain: float = 1.0,
        agc_target_rms: float = 0.08,
        agc_max_gain: float = 8.0,
        startup_grace_seconds: float = 0.40,
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
        self.device = device.strip()
        self.latency = latency
        self.rolling_seconds = max(1.0, float(rolling_seconds))
        self.transcript_window_seconds = max(self.window_seconds, float(transcript_window_seconds))
        self.preamp_gain = max(0.0, float(preamp_gain))
        self.agc_target_rms = max(0.0, float(agc_target_rms))
        self.agc_max_gain = max(1.0, float(agc_max_gain))
        self.startup_grace_seconds = max(0.0, float(startup_grace_seconds))
        self._prev_rms = 0.0
        self._prev_zcr = 0.0
        self._last_text = ""
        self._last_transcript = ""
        self._last_transcript_at = 0.0
        self._last_transcribe_attempt_at = 0.0
        self._lock = threading.Lock()
        self._chunks: deque = deque()
        self._buffered_frames = 0
        self._stream = None
        self._stream_started_at = 0.0
        self._stream_error: str | None = None
        self._resolved_sample_rate = int(sample_rate)
        self._resolved_channels = int(channels)
        self._resolved_device = self.device
        self._buffer_frames = max(1, int(self.rolling_seconds * max(1, self.sample_rate)))
        self._ensure_stream_started()

    def _ensure_stream_started(self) -> None:
        if getattr(self, '_stream', None) is not None:
            return
        if getattr(self, '_stream_error', None) is not None:
            raise MicrophoneUnavailableError(self._stream_error)
        if not hasattr(self, '_sd'):
            return
        try:
            device, sample_rate, channels = self._resolve_stream_params()
            self._resolved_device = device or ''
            self._resolved_sample_rate = sample_rate
            self._resolved_channels = channels
            self._buffer_frames = max(1, int(self.rolling_seconds * self._resolved_sample_rate))
            self._stream = self._sd.InputStream(
                samplerate=self._resolved_sample_rate,
                channels=self._resolved_channels,
                dtype='float32',
                device=device or None,
                latency=self.latency,
                callback=self._audio_callback,
            )
            self._stream.start()
            self._stream_started_at = time.monotonic()
        except Exception as exc:  # pragma: no cover - exercised through status / runtime behavior
            self._stream_error = f'failed to open microphone stream: {exc}'
            raise MicrophoneUnavailableError(self._stream_error) from exc

    def _resolve_stream_params(self) -> tuple[str, int, int]:
        device = self.device or ''
        channels = max(1, int(self.channels))
        preferred = max(1, int(self.sample_rate))
        try:
            self._sd.check_input_settings(device=device or None, samplerate=preferred, channels=channels)
            return device, preferred, channels
        except Exception:
            info = self._sd.query_devices(device or None, 'input')
            max_input = int(info.get('max_input_channels') or channels or 1)
            channels = min(channels, max_input) if max_input > 0 else channels
            default_sr = int(round(float(info.get('default_samplerate') or preferred)))
            try:
                self._sd.check_input_settings(device=device or None, samplerate=default_sr, channels=channels)
                return device, default_sr, channels
            except Exception as exc:
                raise MicrophoneUnavailableError(
                    f'input device {device or "default"!r} rejected sample rates {preferred} and {default_sr}: {exc}'
                ) from exc

    def _audio_callback(self, indata, frames, time_info, status) -> None:  # pragma: no cover - callback path
        _ = frames, time_info
        if status:
            self._stream_error = str(status)
        mono = self._to_mono(indata)
        if mono.size == 0:
            return
        mono = mono.copy()
        with self._lock:
            self._chunks.append(mono)
            self._buffered_frames += int(mono.size)
            while self._buffered_frames > self._buffer_frames and self._chunks:
                removed = self._chunks.popleft()
                self._buffered_frames -= int(removed.size)

    def _snapshot_recent(self, seconds: float):
        frame_count = max(1, int(seconds * max(1, int(getattr(self, '_resolved_sample_rate', self.sample_rate)))))
        with self._lock:
            if not self._chunks:
                return self._np.zeros((0,), dtype='float32')
            chunks = list(self._chunks)
        data = self._np.concatenate(chunks).astype('float32', copy=False)
        if data.size <= frame_count:
            return data
        return data[-frame_count:]

    def record(self, seconds: float | None = None):
        self._ensure_stream_started()
        duration = self.window_seconds if seconds is None else float(seconds)
        return self._snapshot_recent(duration)

    def poll(self) -> List[SensorObservation]:
        self._ensure_stream_started()
        if getattr(self, '_stream_error', None) is not None:
            raise MicrophoneUnavailableError(self._stream_error)
        if getattr(self, '_stream_started_at', 0.0) and (time.monotonic() - self._stream_started_at) < getattr(self, 'startup_grace_seconds', 0.0):
            return []
        mono = self.record()
        mono = self._to_mono(mono)
        resolved_rate = int(getattr(self, '_resolved_sample_rate', self.sample_rate))
        min_frames = max(4, int(resolved_rate * min(self.window_seconds, 0.08)))
        if mono.size < min_frames:
            return []
        analysis = self._prepare_signal(mono)
        if analysis.size == 0:
            return []
        rms = float(math.sqrt(float((analysis * analysis).mean())))
        peak = float(self._np.max(self._np.abs(analysis)))
        signs = analysis[:-1] * analysis[1:]
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
            "sample_rate": int(getattr(self, '_resolved_sample_rate', self.sample_rate)),
            "device": getattr(self, '_resolved_device', '') or 'default',
        }
        transcript_info = self._maybe_transcribe(salience, event, zcr)
        metadata["stt_state"] = transcript_info["state"]
        metadata["stt_detail"] = transcript_info["detail"]
        if transcript_info["transcript"]:
            transcript = str(transcript_info["transcript"])
            text = f'microphone hears {event} and catches speech: "{transcript}". STT: transcript accepted.'
            metadata["transcript"] = transcript
        elif transcript_info["state"] == "transcript-rejected":
            text = f"microphone hears {event} with {texture}. STT: transcript rejected."
        elif transcript_info["state"] == "speech-attempted":
            text = f"microphone hears {event} with {texture}. STT: speech attempted."
        else:
            text = f"microphone hears {event} with {texture}. STT: sound only."
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
        if getattr(self, '_stream_error', None) is not None:
            return SensorStatus(source=self.name, enabled=True, available=False, detail=self._stream_error)
        resolved_device = getattr(self, '_resolved_device', '') or 'default'
        resolved_rate = int(getattr(self, '_resolved_sample_rate', self.sample_rate))
        detail = (
            f"microphone stream active ({resolved_device} @ {resolved_rate} Hz, "
            f"rolling={float(getattr(self, 'rolling_seconds', self.window_seconds)):.1f}s, window={self.window_seconds:.2f}s)"
        )
        if self.transcriber is not None:
            detail += f"; speech transcription via {getattr(self.transcriber, 'model_name', 'transcriber')}"
        return SensorStatus(source=self.name, enabled=True, available=True, detail=detail)

    def close(self) -> None:
        stream = getattr(self, '_stream', None)
        self._stream = None
        if stream is None:
            return None
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        return None

    def _prepare_signal(self, mono):
        data = self._np.asarray(mono, dtype='float32').reshape(-1)
        if data.size == 0:
            return data
        preamp_gain = float(getattr(self, 'preamp_gain', 1.0))
        agc_target_rms = float(getattr(self, 'agc_target_rms', 0.0))
        agc_max_gain = float(getattr(self, 'agc_max_gain', 8.0))
        if preamp_gain != 1.0:
            data = data * preamp_gain
        rms = float(math.sqrt(float((data * data).mean()))) if data.size else 0.0
        if rms > 1e-8 and agc_target_rms > 0.0:
            auto_gain = min(agc_max_gain, agc_target_rms / rms)
            data = data * auto_gain
        data = self._np.tanh(data)
        return data.astype('float32', copy=False)

    def _to_mono(self, sample):
        data = self._np.asarray(sample, dtype='float32')
        if data.ndim == 1:
            return data
        if data.ndim == 2:
            return data.mean(axis=1)
        return data.reshape(-1)

    def _maybe_transcribe(self, salience: float, event: str, zcr: float) -> dict[str, object]:
        info: dict[str, object] = {"state": "sound-only", "detail": "no speech attempt", "transcript": None}
        if self.transcriber is None:
            info["detail"] = "transcriber disabled"
            return info
        if salience < self.transcript_salience_threshold:
            info["detail"] = "below transcript salience threshold"
            return info
        if event != "voice-like audio activity" and not (0.03 <= zcr <= 0.22 and salience >= self.transcript_salience_threshold * 1.1):
            info["detail"] = "signal did not look speech-like enough"
            return info
        now = time.monotonic()
        if self._last_transcript and (now - self._last_transcript_at) < self.transcript_cooldown_seconds:
            info["state"] = "speech-attempted"
            info["detail"] = "transcript cooldown active"
            return info
        if (now - getattr(self, '_last_transcribe_attempt_at', 0.0)) < min(0.25, self.transcript_cooldown_seconds):
            info["state"] = "speech-attempted"
            info["detail"] = "attempt throttled"
            return info
        transcript_window_seconds = float(getattr(self, 'transcript_window_seconds', max(self.window_seconds, 1.8)))
        transcript_audio = self.record(transcript_window_seconds)
        transcript_audio = self._prepare_signal(transcript_audio)
        resolved_rate = int(getattr(self, '_resolved_sample_rate', self.sample_rate))
        min_frames = max(8, int(resolved_rate * min(transcript_window_seconds, 0.35)))
        if transcript_audio.size < min_frames:
            info["state"] = "speech-attempted"
            info["detail"] = "transcript window too short"
            return info
        self._last_transcribe_attempt_at = now
        info["state"] = "speech-attempted"
        info["detail"] = "transcriber invoked"
        try:
            result = self.transcriber.transcribe(transcript_audio, sample_rate=int(getattr(self, '_resolved_sample_rate', self.sample_rate)))
        except Exception as exc:
            info["state"] = "transcript-rejected"
            info["detail"] = f"transcriber error: {exc}"
            return info
        if result is None:
            info["state"] = "transcript-rejected"
            info["detail"] = "transcriber returned no result"
            return info
        text = _clean_transcript(result.text)
        if len(text) < self.transcript_min_chars:
            info["state"] = "transcript-rejected"
            info["detail"] = "transcript too short"
            return info
        if _normalize_text(text) == _normalize_text(self._last_transcript):
            info["state"] = "transcript-rejected"
            info["detail"] = "duplicate transcript"
            return info
        self._last_transcript = text
        self._last_transcript_at = now
        info["state"] = "transcript-accepted"
        info["detail"] = "transcript accepted"
        info["transcript"] = text
        return info



def _clean_transcript(text: str) -> str:
    return " ".join(str(text).strip().split())



def _normalize_text(text: str) -> str:
    return _clean_transcript(text).lower()
