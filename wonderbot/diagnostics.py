from __future__ import annotations

from importlib import metadata
import platform
from typing import Any, Dict


def _safe_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_runtime_diagnostics(
    *,
    config: Any,
    backend_name: str,
    sensor_statuses: list[dict[str, Any]],
    speaker_status: dict[str, Any],
    replay_status: dict[str, Any],
    focus_state: dict[str, Any],
    memory_stats: dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> dict[str, Any]:
    packages = {
        'python': platform.python_version(),
        'transformers': _safe_version('transformers'),
        'torch': _safe_version('torch'),
        'opencv-python': _safe_version('opencv-python'),
        'sounddevice': _safe_version('sounddevice'),
        'pillow': _safe_version('pillow'),
        'pyttsx3': _safe_version('pyttsx3'),
        'numpy': _safe_version('numpy'),
    }
    payload = {
        'agent': config.agent.name,
        'backend': backend_name,
        'packages': packages,
        'sensors': sensor_statuses,
        'speaker': speaker_status,
        'replay': replay_status,
        'focus': focus_state,
        'memory': memory_stats,
    }
    if extra:
        payload.update(extra)
    return payload
