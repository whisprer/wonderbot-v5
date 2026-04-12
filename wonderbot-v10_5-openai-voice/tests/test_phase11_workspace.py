from __future__ import annotations

import sys
from pathlib import Path

from wonderbot.agent import WonderBot
from wonderbot.config import WonderBotConfig


def _config(tmp_path: Path) -> WonderBotConfig:
    cfg = WonderBotConfig.load(Path(__file__).resolve().parents[1] / 'configs' / 'default.toml')
    cfg.memory.path = str(tmp_path / 'memory.json')
    cfg.journal.path = str(tmp_path / 'journal.json')
    cfg.longterm.path = str(tmp_path / 'long_term_memory.json')
    cfg.selfmodel.path = str(tmp_path / 'self_model.json')
    cfg.goals.path = str(tmp_path / 'goals.json')
    cfg.plans.path = str(tmp_path / 'plans.json')
    cfg.execution.path = str(tmp_path / 'action_runs.json')
    cfg.logging.path = str(tmp_path / 'replay.jsonl')
    cfg.workspace.root = str(tmp_path / 'workspace')
    cfg.workspace.allow_writes = True
    cfg.workspace.allow_commands = True
    cfg.workspace.allowed_commands = [Path(sys.executable).name]
    cfg.consolidation.auto_enabled = False
    cfg.sleep.auto_enabled = False
    cfg.tts.enabled = False
    return cfg


def test_phase11_workspace_write_and_read_roundtrip(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        run = bot.run_tool('workspace_write', {'path': 'notes/hello.txt', 'text': 'henlo wonderbot'}, dry_run=False)
        assert run.success
        read = bot.run_tool('workspace_read', {'path': 'notes/hello.txt'}, dry_run=False)
        assert read.success
        assert 'henlo wonderbot' in read.result['text']
    finally:
        bot.close()


def test_phase11_workspace_search_finds_match(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        bot.run_tool('workspace_write', {'path': 'notes/a.txt', 'text': 'coherence over flashy instability'}, dry_run=False)
        result = bot.run_tool('workspace_search', {'query': 'coherence', 'path': 'notes'}, dry_run=False)
        assert result.success
        assert result.result['matches']
        assert result.result['matches'][0]['path'] == 'notes/a.txt'
    finally:
        bot.close()


def test_phase11_workspace_blocks_path_escape(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        run = bot.run_tool('workspace_read', {'path': '../outside.txt'}, dry_run=False)
        assert run.success is False
        assert 'escapes workspace root' in run.summary or 'escapes workspace root' in run.error
    finally:
        bot.close()


def test_phase11_workspace_run_allowlisted_command(tmp_path: Path) -> None:
    bot = WonderBot(_config(tmp_path))
    try:
        command = f'"{sys.executable}" -c "print(12345)"'
        run = bot.run_tool('workspace_run', {'command': command}, dry_run=False)
        assert run.success
        assert '12345' in run.result['stdout']
    finally:
        bot.close()
