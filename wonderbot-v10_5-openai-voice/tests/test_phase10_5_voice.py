from __future__ import annotations

import json
import os
from pathlib import Path

from wonderbot.config import TTSConfig
from wonderbot.tts import OpenAITTSSpeaker, SpeakerStatus, build_speaker


class FakeSpeaker:
    def __init__(self) -> None:
        self.spoken: list[str] = []

    def say(self, text: str) -> None:
        self.spoken.append(text)

    def status(self) -> SpeakerStatus:
        return SpeakerStatus(enabled=True, available=True, detail='fake fallback active', voice_name='fake', engine='fake')

    def close(self) -> None:
        return None


class DummyResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False



def test_build_speaker_falls_back_when_openai_key_missing(monkeypatch) -> None:
    fake = FakeSpeaker()
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setattr('wonderbot.tts._try_build_pyttsx3', lambda config: fake)

    speaker = build_speaker(TTSConfig(enabled=True, engine='openai', fallback_engine='pyttsx3'))
    status = speaker.status()
    assert status.available is True
    assert status.engine == 'pyttsx3-fallback'
    assert 'OpenAI voice unavailable' in status.detail



def test_openai_tts_speaker_posts_expected_payload_and_plays_audio(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    seen: dict[str, object] = {}

    def fake_opener(request, timeout=0):
        seen['url'] = request.full_url
        seen['timeout'] = timeout
        seen['headers'] = dict(request.headers)
        seen['payload'] = json.loads(request.data.decode('utf-8'))
        return DummyResponse(b'RIFFfakewavdata')

    def fake_player(path: Path, fmt: str, backend: str) -> str:
        seen['path_exists_during_play'] = path.exists()
        seen['played_bytes'] = path.read_bytes()
        seen['fmt'] = fmt
        seen['backend'] = backend
        return backend

    monkeypatch.setattr('wonderbot.tts._resolve_playback_backend', lambda fmt, backend: 'fake-player')

    speaker = OpenAITTSSpeaker(
        TTSConfig(
            enabled=True,
            engine='openai',
            openai_model='gpt-4o-mini-tts',
            openai_voice='sage',
            openai_response_format='wav',
            openai_speed=1.1,
            openai_timeout_seconds=12.5,
        ),
        opener=fake_opener,
        player=fake_player,
    )
    speaker.say('henlo wonderbot')

    assert seen['url'] == 'https://api.openai.com/v1/audio/speech'
    assert seen['timeout'] == 12.5
    assert seen['payload'] == {
        'model': 'gpt-4o-mini-tts',
        'voice': 'sage',
        'input': 'henlo wonderbot',
        'response_format': 'wav',
        'speed': 1.1,
    }
    assert seen['fmt'] == 'wav'
    assert seen['backend'] == 'fake-player'
    assert seen['played_bytes'] == b'RIFFfakewavdata'
    assert seen['path_exists_during_play'] is True
