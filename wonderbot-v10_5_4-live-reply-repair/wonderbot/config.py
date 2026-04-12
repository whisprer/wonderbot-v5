from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

try:
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback
    import tomli as tomllib  # type: ignore[no-redef]


@dataclass(slots=True)
class CodecConfig:
    dim: int = 192
    ngram: int = 3
    window_chars: int = 16
    min_segment_chars: int = 4
    cosine_drop: float = 0.14
    lowercase: bool = False
    nfkc: bool = True


@dataclass(slots=True)
class MemoryConfig:
    path: str = 'state/memory.json'
    max_active_items: int = 2000
    protect_identity: bool = True
    importance_threshold: float = 0.36
    min_novelty: float = 0.08


@dataclass(slots=True)
class JournalConfig:
    path: str = 'state/journal.json'


@dataclass(slots=True)
class SelfModelConfig:
    path: str = 'state/self_model.json'
    auto_capture_identity: bool = True
    auto_capture_preferences: bool = True
    auto_capture_constraints: bool = True


@dataclass(slots=True)
class GoalsConfig:
    path: str = 'state/goals.json'
    auto_capture_goals: bool = True
    auto_focus_new_goal: bool = True
    default_priority: float = 0.68


@dataclass(slots=True)
class PlanConfig:
    path: str = 'state/plans.json'
    auto_capture_plans: bool = True
    auto_capture_steps: bool = True
    auto_update_goal_progress: bool = True
    default_priority: float = 0.66
    context_limit: int = 3

@dataclass(slots=True)
class ConsolidationConfig:
    auto_enabled: bool = True
    auto_every_explicit_turns: int = 3
    summary_min_items: int = 4
    summary_window_items: int = 12
    max_summary_sentences: int = 3
    task_limit: int = 4
    belief_limit: int = 4
    thread_limit: int = 4


@dataclass(slots=True)
class LongTermMemoryConfig:
    path: str = 'state/long_term_memory.json'


@dataclass(slots=True)
class SleepConfig:
    auto_enabled: bool = True
    auto_sleep_every_explicit_turns: int = 8
    promotion_limit: int = 6
    min_promotion_strength: float = 0.58
    dream_limit: int = 3
    dream_similarity_min: float = 0.32
    dream_similarity_max: float = 0.82
    archive_decay_rate: float = 0.03
    archive_below_strength: float = 0.12


@dataclass(slots=True)
class GanglionConfig:
    height: int = 8
    width: int = 8
    channels: int = 8
    bleed: float = 0.03


@dataclass(slots=True)
class ResonanceConfig:
    sigma: float = 0.5
    tau: float = 14.134725
    alpha: float = 1.2
    prime_count: int = 32


@dataclass(slots=True)
class BackendConfig:
    kind: str = 'lvtc'
    hf_model: str = 'distilgpt2'
    max_new_tokens: int = 120
    temperature: float = 0.8
    delta_scale: float = 0.24
    creative_depth: int = 1
    anchor_pullback: float = 0.72
    novelty_threshold: float = 0.10
    drift_threshold: float = 0.58
    repetition_threshold: float = 0.34
    latency_budget_ms: int = 30


@dataclass(slots=True)
class AgentConfig:
    name: str = 'wonderbot'
    response_style: str = 'warm, concise, reflective, technical when useful'
    reaction_threshold: float = 0.18
    spontaneous_interval: int = 5
    max_context_memories: int = 6
    focus_max_items: int = 6
    focus_decay_seconds: float = 180.0


@dataclass(slots=True)
class LiveConfig:
    enabled: bool = False
    poll_interval_ms: int = 350
    sensor_memory_threshold: float = 0.10
    sensor_reaction_threshold: float = 0.18
    sensor_reaction_gain: float = 1.15


@dataclass(slots=True)
class CameraConfig:
    enabled: bool = False
    index: int = 0
    width: int = 320
    height: int = 240
    motion_threshold: float = 0.08
    brightness_threshold: float = 0.05
    min_salience: float = 0.12


@dataclass(slots=True)
class MicrophoneConfig:
    enabled: bool = False
    sample_rate: int = 16000
    channels: int = 1
    window_seconds: float = 0.35
    rms_threshold: float = 0.03
    peak_threshold: float = 0.12
    min_salience: float = 0.10
    device: str = ''
    latency: str = 'high'
    rolling_seconds: float = 4.0
    transcript_window_seconds: float = 1.8
    preamp_gain: float = 1.0
    agc_target_rms: float = 0.08
    agc_max_gain: float = 8.0


@dataclass(slots=True)
class CaptionConfig:
    enabled: bool = False
    model: str = 'Salesforce/blip-image-captioning-base'
    max_new_tokens: int = 24
    interval_seconds: float = 3.0
    salience_threshold: float = 0.22
    min_chars: int = 12


@dataclass(slots=True)
class SpeechConfig:
    enabled: bool = False
    model: str = 'openai/whisper-tiny.en'
    language: str = 'en'
    salience_threshold: float = 0.22
    min_chars: int = 4
    cooldown_seconds: float = 0.75


@dataclass(slots=True)
class StabilityConfig:
    sensor_response_cooldown_seconds: float = 2.0
    spontaneous_cooldown_seconds: float = 6.0
    repeated_stimulus_cooldown_seconds: float = 3.5
    same_source_cooldown_seconds: float = 1.25
    minimum_response_salience: float = 0.18




@dataclass(slots=True)
class ExecutionConfig:
    path: str = 'state/action_runs.json'
    default_dry_run: bool = True
    auto_mark_step_doing: bool = True
    auto_mark_done_on_success: bool = False

@dataclass(slots=True)
class LoggingConfig:
    enabled: bool = True
    path: str = 'state/replay.jsonl'
    flush_each_write: bool = True


@dataclass(slots=True)
class TTSConfig:
    enabled: bool = False
    engine: str = 'hf'
    fallback_engine: str = 'pyttsx3'
    rate: int = 185
    volume: float = 0.9
    voice_contains: str = ''
    hf_model: str = 'facebook/mms-tts-eng'
    hf_vocoder_model: str = 'microsoft/speecht5_hifigan'
    hf_speaker_embeddings_source: str = 'Matthijs/cmu-arctic-xvectors'
    hf_speaker_id: int = 7306
    hf_device: str = 'cpu'
    hf_sample_rate: int = 16000
    openai_model: str = 'gpt-4o-mini-tts'
    openai_voice: str = 'sage'
    openai_response_format: str = 'wav'
    openai_speed: float = 1.0
    openai_timeout_seconds: float = 30.0
    openai_api_key_env: str = 'OPENAI_API_KEY'
    openai_base_url: str = 'https://api.openai.com/v1'
    playback_backend: str = 'auto'
    speak_user_responses: bool = True
    speak_sensor_responses: bool = False
    speak_spontaneous: bool = True


@dataclass(slots=True)
class WonderBotConfig:
    agent: AgentConfig
    codec: CodecConfig
    memory: MemoryConfig
    journal: JournalConfig
    longterm: LongTermMemoryConfig
    selfmodel: SelfModelConfig
    goals: GoalsConfig
    plans: PlanConfig
    consolidation: ConsolidationConfig
    sleep: SleepConfig
    ganglion: GanglionConfig
    resonance: ResonanceConfig
    backend: BackendConfig
    live: LiveConfig
    camera: CameraConfig
    microphone: MicrophoneConfig
    caption: CaptionConfig
    speech: SpeechConfig
    stability: StabilityConfig
    logging: LoggingConfig
    execution: ExecutionConfig
    tts: TTSConfig

    @classmethod
    def load(cls, path: str | Path) -> 'WonderBotConfig':
        data = _read_toml(path)
        return cls(
            agent=AgentConfig(**data.get('agent', {})),
            codec=CodecConfig(**data.get('codec', {})),
            memory=MemoryConfig(**data.get('memory', {})),
            journal=JournalConfig(**data.get('journal', {})),
            longterm=LongTermMemoryConfig(**data.get('longterm', {})),
            selfmodel=SelfModelConfig(**data.get('selfmodel', {})),
            goals=GoalsConfig(**data.get('goals', {})),
            plans=PlanConfig(**data.get('plans', {})),
            consolidation=ConsolidationConfig(**data.get('consolidation', {})),
            sleep=SleepConfig(**data.get('sleep', {})),
            ganglion=GanglionConfig(**data.get('ganglion', {})),
            resonance=ResonanceConfig(**data.get('resonance', {})),
            backend=BackendConfig(**data.get('backend', {})),
            live=LiveConfig(**data.get('live', {})),
            camera=CameraConfig(**data.get('camera', {})),
            microphone=MicrophoneConfig(**data.get('microphone', {})),
            caption=CaptionConfig(**data.get('caption', {})),
            speech=SpeechConfig(**data.get('speech', {})),
            stability=StabilityConfig(**data.get('stability', {})),
            logging=LoggingConfig(**data.get('logging', {})),
            execution=ExecutionConfig(**data.get('execution', {})),
            tts=TTSConfig(**data.get('tts', {})),
        )


def _read_toml(path: str | Path) -> Dict[str, Any]:
    with open(path, 'rb') as handle:
        return tomllib.load(handle)
