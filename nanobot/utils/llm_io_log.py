"""Append-only logging for raw LLM request/response payloads."""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from loguru import logger

from nanobot.config.paths import get_logs_dir

_LOG_LOCK = threading.Lock()
_DISABLE_VALUES = {"0", "false", "no", "off"}
_SESSION_STARTED_AT = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
_SESSION_LOG_PATH: Path | None = None
_INITIALIZED = False


def _enabled() -> bool:
    configured = os.environ.get("NANOBOT_LLM_IO_LOG")
    if configured is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return False
    raw = (configured or "1").strip().lower()
    return raw not in _DISABLE_VALUES


def _log_path() -> Path:
    global _SESSION_LOG_PATH

    override = os.environ.get("NANOBOT_LLM_IO_LOG_PATH", "").strip()
    if override:
        return Path(override).expanduser()
    if _SESSION_LOG_PATH is None:
        _SESSION_LOG_PATH = (
            get_logs_dir() / f"llm_io_{_SESSION_STARTED_AT}_pid{os.getpid()}.jsonl"
        )
    return _SESSION_LOG_PATH


def initialize_llm_io_log() -> Path | None:
    """Create this process's LLM I/O log file and return its path."""
    global _INITIALIZED

    if not _enabled():
        return None
    try:
        path = _log_path()
        if _INITIALIZED:
            return path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        _INITIALIZED = True
        logger.info("LLM I/O log: {}", path)
        return path
    except Exception as exc:
        logger.debug("Failed to initialize LLM I/O log: {}", exc)
        return None


def _json_default(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, BaseException):
        return {"type": type(value).__name__, "message": str(value)}
    return repr(value)


def _append(record: dict[str, Any]) -> None:
    if not _enabled():
        return
    try:
        path = initialize_llm_io_log()
        if path is None:
            return
        line = json.dumps(record, ensure_ascii=False, default=_json_default)
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
    except Exception as exc:
        logger.debug("Failed to write LLM I/O log: {}", exc)


def _base_record(
    *,
    interaction_id: str,
    phase: str,
    provider: Any,
    attempt: int,
    retry_mode: str,
    stream: bool,
) -> dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "interaction_id": interaction_id,
        "phase": phase,
        "provider": type(provider).__name__,
        "attempt": attempt,
        "retry_mode": retry_mode,
        "stream": stream,
    }


def log_llm_request(
    provider: Any,
    kwargs: dict[str, Any],
    *,
    attempt: int,
    retry_mode: str,
    stream: bool,
    note: str | None = None,
) -> str:
    """Log the complete payload submitted to the model and return its id."""
    interaction_id = uuid.uuid4().hex
    payload = {
        key: value
        for key, value in kwargs.items()
        if key not in {"on_content_delta", "on_retry_wait"}
    }
    record = _base_record(
        interaction_id=interaction_id,
        phase="request",
        provider=provider,
        attempt=attempt,
        retry_mode=retry_mode,
        stream=stream,
    )
    record["payload"] = payload
    if note:
        record["note"] = note
    _append(record)
    return interaction_id


def log_llm_response(
    provider: Any,
    interaction_id: str,
    response: Any,
    *,
    attempt: int,
    retry_mode: str,
    stream: bool,
) -> None:
    """Log the complete normalized model response."""
    record = _base_record(
        interaction_id=interaction_id,
        phase="response",
        provider=provider,
        attempt=attempt,
        retry_mode=retry_mode,
        stream=stream,
    )
    record["response"] = response
    _append(record)


def log_llm_exception(
    provider: Any,
    interaction_id: str,
    exc: BaseException,
    *,
    attempt: int,
    retry_mode: str,
    stream: bool,
) -> None:
    """Log exceptions raised while waiting for the model call."""
    record = _base_record(
        interaction_id=interaction_id,
        phase="exception",
        provider=provider,
        attempt=attempt,
        retry_mode=retry_mode,
        stream=stream,
    )
    record["exception"] = exc
    _append(record)
