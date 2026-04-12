from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Protocol


class PerceptionUnavailableError(RuntimeError):
    pass


@dataclass(slots=True)
class CaptionResult:
    text: str
    model_name: str
    latency_ms: int


@dataclass(slots=True)
class TranscriptResult:
    text: str
    model_name: str
    latency_ms: int


class ImageCaptioner(Protocol):
    model_name: str

    def caption(self, image: Any) -> CaptionResult | None:
        ...


class SpeechTranscriber(Protocol):
    model_name: str

    def transcribe(self, audio: Any, sample_rate: int) -> TranscriptResult | None:
        ...


class HFImageCaptioner:
    def __init__(self, model_name: str, max_new_tokens: int = 24) -> None:
        try:
            from transformers import pipeline
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise PerceptionUnavailableError(
                'Captioning requires transformers, torch, pillow, and numpy. Install with: pip install -e .[multimodal]'
            ) from exc
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self._pipeline = _build_caption_pipeline(pipeline, model_name)
        self._Image = Image
        self._np = np

    def caption(self, image: Any) -> CaptionResult | None:
        pil_image = self._coerce_image(image)
        start = time.perf_counter()
        raw = self._pipeline(pil_image, max_new_tokens=self.max_new_tokens)
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = _extract_generated_text(raw)
        text = _normalize_caption(text)
        if not text:
            return None
        return CaptionResult(text=text, model_name=self.model_name, latency_ms=latency_ms)

    def _coerce_image(self, image: Any):
        if hasattr(image, 'mode') and hasattr(image, 'size'):
            return image
        array = self._np.asarray(image)
        if array.ndim == 2:
            return self._Image.fromarray(array)
        if array.ndim == 3 and array.shape[-1] == 3:
            rgb = array[..., ::-1]
            return self._Image.fromarray(rgb)
        raise ValueError('Unsupported image shape for captioning.')


class HFSpeechTranscriber:
    def __init__(self, model_name: str, language: str = 'en') -> None:
        try:
            from transformers import pipeline
            import numpy as np
        except ImportError as exc:
            raise PerceptionUnavailableError(
                'Speech transcription requires transformers, torch, and numpy. Install with: pip install -e .[multimodal]'
            ) from exc
        self.model_name = model_name
        self.language = language
        self._pipeline = pipeline('automatic-speech-recognition', model=model_name)
        self._np = np

    def transcribe(self, audio: Any, sample_rate: int) -> TranscriptResult | None:
        array = self._np.asarray(audio, dtype='float32').reshape(-1)
        if array.size == 0:
            return None
        start = time.perf_counter()
        kwargs = {}
        if self.language:
            kwargs = {'generate_kwargs': {'language': self.language, 'task': 'transcribe'}}
        raw = self._pipeline({'array': array, 'sampling_rate': sample_rate}, **kwargs)
        latency_ms = int((time.perf_counter() - start) * 1000)
        text = _extract_transcript_text(raw)
        text = _normalize_transcript(text)
        if not text:
            return None
        return TranscriptResult(text=text, model_name=self.model_name, latency_ms=latency_ms)


class NullImageCaptioner:
    model_name = 'none'

    def caption(self, image: Any) -> CaptionResult | None:
        return None


class NullSpeechTranscriber:
    model_name = 'none'

    def transcribe(self, audio: Any, sample_rate: int) -> TranscriptResult | None:
        return None


def build_image_captioner(model_name: str, max_new_tokens: int = 24) -> ImageCaptioner:
    return HFImageCaptioner(model_name=model_name, max_new_tokens=max_new_tokens)


def build_speech_transcriber(model_name: str, language: str = 'en') -> SpeechTranscriber:
    return HFSpeechTranscriber(model_name=model_name, language=language)


def _build_caption_pipeline(pipeline, model_name: str):
    tasks = ['image-text-to-text', 'image-to-text']
    errors: list[str] = []
    for task in tasks:
        try:
            return pipeline(task, model=model_name)
        except KeyError as exc:
            errors.append(f'{task}: {exc}')
    raise PerceptionUnavailableError('No compatible image caption pipeline task was available. Tried: ' + '; '.join(errors))


def _extract_generated_text(raw: Any) -> str:
    if isinstance(raw, list) and raw:
        item = raw[0]
        if isinstance(item, dict):
            return str(item.get('generated_text') or item.get('caption') or item.get('generated_texts') or '')
        return str(item)
    if isinstance(raw, dict):
        return str(raw.get('generated_text') or raw.get('caption') or raw.get('generated_texts') or '')
    return str(raw or '')


def _extract_transcript_text(raw: Any) -> str:
    if isinstance(raw, dict):
        return str(raw.get('text') or '')
    if isinstance(raw, list) and raw:
        item = raw[0]
        if isinstance(item, dict):
            return str(item.get('text') or '')
        return str(item)
    return str(raw or '')


def _normalize_caption(text: str) -> str:
    text = ' '.join(text.strip().split())
    lowered = text.lower()
    prefixes = ['a photo of ', 'an image of ', 'a picture of ']
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix):]
            break
    return text.strip(' .')


def _normalize_transcript(text: str) -> str:
    text = ' '.join(text.strip().split())
    return text.strip()
