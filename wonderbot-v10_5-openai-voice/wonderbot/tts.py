from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, Optional, Protocol
from urllib import error as urlerror
from urllib import request as urlrequest

from .config import TTSConfig


class TTSUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class SpeakerStatus:
    enabled: bool
    available: bool
    detail: str
    voice_name: str = ""
    engine: str = ""


class Speaker(Protocol):
    def say(self, text: str) -> None:
        ...

    def status(self) -> SpeakerStatus:
        ...

    def close(self) -> None:
        ...


class NullSpeaker:
    def __init__(self, enabled: bool, detail: str) -> None:
        self._status = SpeakerStatus(enabled=enabled, available=False, detail=detail, voice_name="", engine="none")

    def say(self, text: str) -> None:
        return None

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        return None


class DelegatingSpeaker:
    def __init__(self, delegate: Speaker, detail: str, engine: str) -> None:
        self._delegate = delegate
        base = delegate.status()
        self._status = SpeakerStatus(
            enabled=base.enabled,
            available=base.available,
            detail=detail,
            voice_name=base.voice_name,
            engine=engine,
        )

    def say(self, text: str) -> None:
        self._delegate.say(text)

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        self._delegate.close()


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
        self._status = SpeakerStatus(
            enabled=True,
            available=True,
            detail='pyttsx3 voice output active',
            voice_name=selected_voice,
            engine='pyttsx3',
        )

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


class OpenAITTSSpeaker:
    def __init__(
        self,
        config: TTSConfig,
        *,
        fallback: Speaker | None = None,
        opener: Callable[..., object] | None = None,
        player: Callable[[Path, str, str], str] | None = None,
    ) -> None:
        api_key = os.getenv(config.openai_api_key_env, '').strip()
        if not api_key:
            raise TTSUnavailableError(f'{config.openai_api_key_env} is not set')
        fmt = config.openai_response_format.strip().lower() or 'wav'
        if fmt not in {'mp3', 'opus', 'aac', 'flac', 'wav', 'pcm'}:
            raise TTSUnavailableError(f'unsupported OpenAI response format: {fmt}')
        self._config = config
        self._api_key = api_key
        self._base_url = (os.getenv('OPENAI_BASE_URL') or config.openai_base_url or 'https://api.openai.com/v1').rstrip('/')
        self._fmt = fmt
        self._fallback = fallback
        self._opener = opener or urlrequest.urlopen
        self._player = player or _play_audio_file
        self._resolved_backend = _resolve_playback_backend(fmt, config.playback_backend)
        self._status = SpeakerStatus(
            enabled=True,
            available=True,
            detail=f'OpenAI TTS active ({config.openai_model}/{config.openai_voice}, {fmt}, playback={self._resolved_backend})',
            voice_name=config.openai_voice,
            engine='openai',
        )

    def _request_audio(self, text: str) -> bytes:
        payload = {
            'model': self._config.openai_model,
            'voice': self._config.openai_voice,
            'input': text,
            'response_format': self._fmt,
            'speed': float(self._config.openai_speed),
        }
        data = json.dumps(payload).encode('utf-8')
        req = urlrequest.Request(
            f'{self._base_url}/audio/speech',
            data=data,
            headers={
                'Authorization': f'Bearer {self._api_key}',
                'Content-Type': 'application/json',
            },
            method='POST',
        )
        timeout = float(self._config.openai_timeout_seconds)
        try:
            with self._opener(req, timeout=timeout) as response:
                return response.read()
        except urlerror.HTTPError as exc:
            try:
                detail = exc.read().decode('utf-8', errors='replace')
            except Exception:
                detail = str(exc)
            raise RuntimeError(f'OpenAI TTS request failed: {detail}') from exc
        except urlerror.URLError as exc:
            raise RuntimeError(f'OpenAI TTS network error: {exc}') from exc

    def say(self, text: str) -> None:
        text = text.strip()
        if not text:
            return
        try:
            audio = self._request_audio(text)
            suffix = f'.{self._fmt}'
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
                handle.write(audio)
                temp_path = Path(handle.name)
            try:
                self._player(temp_path, self._fmt, self._resolved_backend)
            finally:
                try:
                    temp_path.unlink(missing_ok=True)
                except Exception:
                    pass
        except Exception:
            if self._fallback is not None:
                self._fallback.say(text)
                return
            raise

    def status(self) -> SpeakerStatus:
        return self._status

    def close(self) -> None:
        if self._fallback is not None:
            self._fallback.close()



def _resolve_playback_backend(fmt: str, backend: str) -> str:
    choice = (backend or 'auto').strip().lower()
    candidates = [choice] if choice != 'auto' else ['winsound', 'sounddevice', 'afplay', 'aplay', 'pw-play']
    for candidate in candidates:
        if candidate == 'winsound' and sys.platform.startswith('win') and fmt == 'wav':
            return 'winsound'
        if candidate == 'sounddevice':
            try:
                import sounddevice  # noqa: F401
                import soundfile  # noqa: F401
                return 'sounddevice'
            except Exception:
                continue
        if candidate in {'afplay', 'aplay', 'pw-play'} and shutil.which(candidate):
            return candidate
    raise TTSUnavailableError(
        'no audio playback backend is available for OpenAI TTS; use wav on Windows, install sounddevice+soundfile, or rely on pyttsx3 fallback'
    )



def _play_audio_file(path: Path, fmt: str, backend: str) -> str:
    if backend == 'winsound':
        import winsound

        winsound.PlaySound(str(path), winsound.SND_FILENAME)
        return backend
    if backend == 'sounddevice':
        import sounddevice as sd
        import soundfile as sf

        data, sample_rate = sf.read(str(path), dtype='float32')
        sd.play(data, sample_rate)
        sd.wait()
        return backend
    if backend in {'afplay', 'aplay', 'pw-play'}:
        subprocess.run([backend, str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return backend
    raise TTSUnavailableError(f'unsupported playback backend: {backend}')



def _try_build_pyttsx3(config: TTSConfig) -> Speaker | None:
    try:
        return Pyttsx3Speaker(config)
    except TTSUnavailableError:
        return None



def build_speaker(config: TTSConfig) -> Speaker:
    if not config.enabled:
        return NullSpeaker(enabled=False, detail='voice output disabled in config')
    engine = (config.engine or 'openai').strip().lower()
    fallback_engine = (config.fallback_engine or '').strip().lower()

    fallback: Speaker | None = None
    if fallback_engine == 'pyttsx3' and engine != 'pyttsx3':
        fallback = _try_build_pyttsx3(config)

    if engine in {'openai', 'openai_tts'}:
        try:
            return OpenAITTSSpeaker(config, fallback=fallback)
        except TTSUnavailableError as exc:
            if fallback is not None:
                return DelegatingSpeaker(
                    fallback,
                    detail=f'{fallback.status().detail}; preferred OpenAI voice unavailable: {exc}',
                    engine='pyttsx3-fallback',
                )
            return NullSpeaker(enabled=True, detail=f'voice output unavailable: {exc}')
    if engine == 'pyttsx3':
        try:
            return Pyttsx3Speaker(config)
        except TTSUnavailableError as exc:
            return NullSpeaker(enabled=True, detail=f'voice output unavailable: {exc}')
    return NullSpeaker(enabled=True, detail=f'unknown tts engine: {config.engine}')
