from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
import os
import subprocess
from typing import Any, Dict, Iterable, List


@dataclass(slots=True)
class WorkspaceStatus:
    root: str
    writes_allowed: bool
    commands_allowed: bool
    allow_http: bool
    allowed_commands: List[str] = field(default_factory=list)
    max_read_bytes: int = 0
    max_write_bytes: int = 0
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class WorkspaceGuard:
    def __init__(
        self,
        *,
        root: str = ".",
        allow_writes: bool = True,
        allow_commands: bool = False,
        allow_http: bool = False,
        allowed_commands: Iterable[str] | None = None,
        max_read_bytes: int = 131072,
        max_write_bytes: int = 131072,
        command_timeout_seconds: float = 20.0,
    ) -> None:
        self.root = Path(root).resolve()
        self.allow_writes = bool(allow_writes)
        self.allow_commands = bool(allow_commands)
        self.allow_http = bool(allow_http)
        self.allowed_commands = [str(item).strip() for item in (allowed_commands or []) if str(item).strip()]
        self.max_read_bytes = int(max(1024, max_read_bytes))
        self.max_write_bytes = int(max(1024, max_write_bytes))
        self.command_timeout_seconds = float(max(1.0, command_timeout_seconds))
        self.root.mkdir(parents=True, exist_ok=True)

    def status(self) -> WorkspaceStatus:
        detail = f"root={self.root}"
        if self.allow_commands:
            detail += ", commands enabled"
        return WorkspaceStatus(
            root=str(self.root),
            writes_allowed=self.allow_writes,
            commands_allowed=self.allow_commands,
            allow_http=self.allow_http,
            allowed_commands=list(self.allowed_commands),
            max_read_bytes=self.max_read_bytes,
            max_write_bytes=self.max_write_bytes,
            detail=detail,
        )

    def list_dir(self, path: str = ".", *, recursive: bool = False, limit: int = 100) -> List[Dict[str, Any]]:
        target = self.resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"workspace path does not exist: {path}")
        if not target.is_dir():
            raise NotADirectoryError(f"workspace path is not a directory: {path}")
        items: List[Dict[str, Any]] = []
        iterator = target.rglob("*") if recursive else target.iterdir()
        for entry in iterator:
            if len(items) >= max(1, limit):
                break
            rel = entry.relative_to(self.root).as_posix()
            if entry.is_dir():
                items.append({"path": rel, "kind": "dir", "size": 0})
            elif entry.is_file():
                try:
                    size = entry.stat().st_size
                except OSError:
                    size = -1
                items.append({"path": rel, "kind": "file", "size": size})
        return sorted(items, key=lambda item: (item["kind"], item["path"]))

    def read_text(self, path: str, *, max_bytes: int | None = None) -> Dict[str, Any]:
        target = self.resolve_path(path)
        if not target.exists():
            raise FileNotFoundError(f"workspace file does not exist: {path}")
        if not target.is_file():
            raise IsADirectoryError(f"workspace path is not a file: {path}")
        limit = min(self.max_read_bytes, int(max_bytes or self.max_read_bytes))
        raw = target.read_bytes()
        truncated = len(raw) > limit
        raw = raw[:limit]
        text = raw.decode("utf-8", errors="replace")
        return {
            "path": target.relative_to(self.root).as_posix(),
            "text": text,
            "bytes": len(raw),
            "truncated": truncated,
        }

    def write_text(self, path: str, text: str, *, append: bool = False) -> Dict[str, Any]:
        if not self.allow_writes:
            raise PermissionError("workspace writes are disabled")
        data = text.encode("utf-8")
        if len(data) > self.max_write_bytes:
            raise ValueError(f"write exceeds max_write_bytes ({self.max_write_bytes})")
        target = self.resolve_path(path, allow_missing_leaf=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        if append and target.exists():
            existing = target.read_text(encoding="utf-8", errors="replace")
            target.write_text(existing + text, encoding="utf-8")
        else:
            target.write_text(text, encoding="utf-8")
        return {
            "path": target.relative_to(self.root).as_posix(),
            "bytes_written": len(data),
            "append": bool(append),
        }

    def search_text(self, query: str, *, path: str = ".", glob: str = "*", limit: int = 20) -> List[Dict[str, Any]]:
        query = str(query).strip()
        if not query:
            return []
        base = self.resolve_path(path)
        if not base.exists():
            raise FileNotFoundError(f"workspace path does not exist: {path}")
        if base.is_file():
            candidates = [base]
        else:
            candidates = [item for item in base.rglob(glob) if item.is_file()]
        lowered = query.casefold()
        matches: List[Dict[str, Any]] = []
        for candidate in candidates:
            try:
                if candidate.stat().st_size > self.max_read_bytes:
                    continue
                lines = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for idx, line in enumerate(lines, start=1):
                if lowered in line.casefold():
                    matches.append({
                        "path": candidate.relative_to(self.root).as_posix(),
                        "line": idx,
                        "text": line[:280],
                    })
                    if len(matches) >= max(1, limit):
                        return matches
        return matches

    def run_command(self, command: str, *, cwd: str = ".", timeout_seconds: float | None = None) -> Dict[str, Any]:
        if not self.allow_commands:
            raise PermissionError("workspace commands are disabled")
        parts = _split_command(command)
        if not parts:
            raise ValueError("command is empty")
        executable = os.path.basename(parts[0]).lower()
        allowed = {item.lower() for item in self.allowed_commands}
        if allowed and executable not in allowed:
            raise PermissionError(f"command not in allowlist: {executable}")
        workdir = self.resolve_path(cwd)
        if not workdir.exists() or not workdir.is_dir():
            raise NotADirectoryError(f"invalid cwd: {cwd}")
        completed = subprocess.run(
            parts,
            cwd=str(workdir),
            capture_output=True,
            text=True,
            timeout=float(timeout_seconds or self.command_timeout_seconds),
            shell=False,
        )
        return {
            "command": parts,
            "cwd": workdir.relative_to(self.root).as_posix(),
            "returncode": int(completed.returncode),
            "stdout": completed.stdout[: self.max_read_bytes],
            "stderr": completed.stderr[: self.max_read_bytes],
            "success": completed.returncode == 0,
        }

    def resolve_path(self, path: str, *, allow_missing_leaf: bool = False) -> Path:
        candidate = (self.root / path).expanduser()
        candidate_resolved = candidate.resolve(strict=False)
        if allow_missing_leaf:
            parent = candidate_resolved.parent
            parent.mkdir(parents=True, exist_ok=True)
            self._assert_within_root(parent)
        self._assert_within_root(candidate_resolved if candidate_resolved.exists() else candidate_resolved.parent if allow_missing_leaf else candidate_resolved)
        return candidate_resolved

    def _assert_within_root(self, value: Path) -> None:
        try:
            value.relative_to(self.root)
        except ValueError as exc:
            raise PermissionError(f"path escapes workspace root: {value}") from exc


def _split_command(command: str) -> List[str]:
    import shlex

    return shlex.split(command)
