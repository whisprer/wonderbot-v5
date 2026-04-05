from __future__ import annotations

from typing import List, Sequence

from .base import SensorAdapter, SensorObservation, SensorStatus
from .camera import CameraUnavailableError, OpenCVCameraAdapter
from .microphone import MicrophoneUnavailableError, SoundDeviceMicrophoneAdapter
from ..config import WonderBotConfig
from ..perception import PerceptionUnavailableError, build_image_captioner, build_speech_transcriber


class SensorHub:
    def __init__(self, adapters: Sequence[SensorAdapter] | None = None, statuses: Sequence[SensorStatus] | None = None) -> None:
        self.adapters = list(adapters or [])
        self._statuses = list(statuses or [])

    def poll(self) -> List[SensorObservation]:
        observations: List[SensorObservation] = []
        updated_statuses: List[SensorStatus] = []
        for adapter in self.adapters:
            try:
                observations.extend(adapter.poll())
                updated_statuses.append(adapter.status())
            except Exception as exc:
                updated_statuses.append(SensorStatus(source=adapter.name, enabled=True, available=False, detail=str(exc)))
        if updated_statuses:
            self._statuses = updated_statuses
        return observations

    def status(self) -> List[SensorStatus]:
        if self._statuses:
            return list(self._statuses)
        return [adapter.status() for adapter in self.adapters]

    def close(self) -> None:
        for adapter in self.adapters:
            try:
                adapter.close()
            except Exception:
                pass



def build_sensor_hub(config: WonderBotConfig) -> SensorHub:
    adapters: List[SensorAdapter] = []
    statuses: List[SensorStatus] = []

    captioner = None
    caption_detail = "captioning disabled"
    if config.caption.enabled:
        try:
            captioner = build_image_captioner(config.caption.model, max_new_tokens=config.caption.max_new_tokens)
            caption_detail = f"captioning active ({config.caption.model})"
        except PerceptionUnavailableError as exc:
            caption_detail = f"captioning unavailable: {exc}"

    speech_transcriber = None
    speech_detail = "speech transcription disabled"
    if config.speech.enabled:
        try:
            speech_transcriber = build_speech_transcriber(config.speech.model, language=config.speech.language)
            speech_detail = f"speech transcription active ({config.speech.model})"
        except PerceptionUnavailableError as exc:
            speech_detail = f"speech transcription unavailable: {exc}"

    if config.camera.enabled:
        try:
            adapters.append(
                OpenCVCameraAdapter(
                    index=config.camera.index,
                    width=config.camera.width,
                    height=config.camera.height,
                    motion_threshold=config.camera.motion_threshold,
                    brightness_threshold=config.camera.brightness_threshold,
                    min_salience=config.camera.min_salience,
                    captioner=captioner,
                    caption_interval_seconds=config.caption.interval_seconds,
                    caption_salience_threshold=config.caption.salience_threshold,
                    caption_min_chars=config.caption.min_chars,
                )
            )
            statuses.append(SensorStatus(source="camera", enabled=True, available=True, detail=f"camera adapter active; {caption_detail}"))
        except CameraUnavailableError as exc:
            statuses.append(SensorStatus(source="camera", enabled=True, available=False, detail=str(exc)))
    else:
        statuses.append(SensorStatus(source="camera", enabled=False, available=False, detail="camera disabled in config"))

    if config.microphone.enabled:
        try:
            adapters.append(
                SoundDeviceMicrophoneAdapter(
                    sample_rate=config.microphone.sample_rate,
                    channels=config.microphone.channels,
                    window_seconds=config.microphone.window_seconds,
                    rms_threshold=config.microphone.rms_threshold,
                    peak_threshold=config.microphone.peak_threshold,
                    min_salience=config.microphone.min_salience,
                    transcriber=speech_transcriber,
                    transcript_salience_threshold=config.speech.salience_threshold,
                    transcript_min_chars=config.speech.min_chars,
                    transcript_cooldown_seconds=config.speech.cooldown_seconds,
                )
            )
            statuses.append(SensorStatus(source="microphone", enabled=True, available=True, detail=f"microphone adapter active; {speech_detail}"))
        except MicrophoneUnavailableError as exc:
            statuses.append(SensorStatus(source="microphone", enabled=True, available=False, detail=str(exc)))
    else:
        statuses.append(SensorStatus(source="microphone", enabled=False, available=False, detail="microphone disabled in config"))

    return SensorHub(adapters=adapters, statuses=statuses)
