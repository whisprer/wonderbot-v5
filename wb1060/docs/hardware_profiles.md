# Hardware/device profiles

WonderBot now has a runtime device layer. The important knobs are:

- `runtime.default_device`
- `runtime.speech_device`
- `runtime.caption_device`
- `runtime.tts_device`
- `runtime.hf_llm_device`
- `runtime.hf_llm_device_map`
- `runtime.hf_llm_torch_dtype`

Supported device specs are:

- `auto`
- `cpu`
- `cuda`
- `cuda:0`, `cuda:1`, ...
- `mps`

## How to use

Run with a specific profile:

```powershell
py -3.11 -m wonderbot.cli --config configs/profiles/current-box-cpu.toml --diagnostics
```

Override per run without editing files:

```powershell
py -3.11 -m wonderbot.cli --device auto --speech-device cuda:0 --tts-device cpu --diagnostics
```

## Strategy

### Current machine

- CPU-only: use `configs/profiles/current-box-cpu.toml`
- Quadro P1000 tactical use: use `configs/profiles/current-box-p1000.toml` and keep only ASR on GPU first

### Future 4070 + A40 box

Start with `configs/profiles/big-machine-dual-gpu-template.toml`, then verify actual GPU numbering in `/diagnostics`.

The safest first routing is:

- ASR -> A40
- Captioning -> A40
- TTS -> CPU
- custom local agent loop -> CPU
- large HF LLM backends -> A40 with `device_map = "auto"`

## Notes

- Device diagnostics now show CUDA availability, device count, and detected GPU names.
- Captioning, ASR, HF TTS, and HF text backends all honor the runtime device layer.
- The current local `lvtc` backend remains CPU-oriented by design; the device routing matters most for HF-backed components.
