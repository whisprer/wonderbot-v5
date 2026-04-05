# Phase 10.5 — Voice upgrade

Preferred voice path: OpenAI TTS (`gpt-4o-mini-tts`) with `sage` as the default voice and `wav` output for easy playback.

Fallback path: existing `pyttsx3` speaker when the API key, network path, or audio playback backend is unavailable.

Environment:

- `OPENAI_API_KEY` — required for OpenAI voice output
- `OPENAI_BASE_URL` — optional override for compatible gateways

Recommended run:

```powershell
py -3.11 -m pip install -e .[live-full,openai-voice,dev]
$env:OPENAI_API_KEY = "sk-..."
py -3.11 -m wonderbot.cli --live --microphone --stt --tts --diagnostics
```
