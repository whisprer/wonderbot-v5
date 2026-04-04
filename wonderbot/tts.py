from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .config import TTSConfig


class TTSUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class SpeakerStatus:
    enabled: bool
    available: bool
    detail: str
    voice_name: str = ""


class Speaker(Protocol):
    def say(self, text: str) -> None:
        ...

    def status(self) -> SpeakerStatus:
        ...

    def close(self) -> None:
        ...


class NullSpeaker:
    def __init__(self, enabled: bool, detail: str) -> None:
        self._status = SpeakerStatus(enabled=enabled, available=False, detail=detail, voice_name="")

    def say(self, text: str) -> None:
        return None

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        return None


class Pyttsx3Speaker:
    def __init__(self, config: TTSConfig) -> None:
        try:
            import pyttsx3
        except ImportError as exc:
            raise TTSUnavailableError('pyttsx3 is not installed. Install with: pip install -e .[voice]') from exc
        self._engine = pyttsx3.init()
        self._engine.setProperty('rate', int(config.rate))
        self._engine.setProperty('volume', float(config.volume))
        selected_voice = ""
        voice_filter = config.voice_contains.strip().lower()
        if voice_filter:
            for voice in self._engine.getProperty('voices') or []:
                name = str(getattr(voice, 'name', ''))
                if voice_filter in name.lower():
                    self._engine.setProperty('voice', getattr(voice, 'id'))
                    selected_voice = name
                    break
        if not selected_voice:
            current = self._engine.getProperty('voice')
            for voice in self._engine.getProperty('voices') or []:
                if getattr(voice, 'id', None) == current:
                    selected_voice = str(getattr(voice, 'name', ''))
                    break
        self._status = SpeakerStatus(enabled=True, available=True, detail='pyttsx3 voice output active', voice_name=selected_voice)

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        self._engine.say(text)
        self._engine.runAndWait()

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        try:
            self._engine.stop()
        except Exception:
            pass


def build_speaker(config: TTSConfig) -> Speaker:
    if not config.enabled:
        return NullSpeaker(enabled=False, detail='voice output disabled in config')
    try:
        return Pyttsx3Speaker(config)
    except TTSUnavailableError as exc:
        return NullSpeaker(enabled=True, detail=f'voice output unavailable: {exc}')
