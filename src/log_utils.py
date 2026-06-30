from __future__ import annotations

import datetime as _dt
import os
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Optional

_LEVEL = {
    "INFO": "INFO",
    "SUCCESS": "OK",
    "OK": "OK",
    "WARN": "WARN",
    "ERROR": "ERR",
    "ERR": "ERR",
    "DEBUG": "DBG",
    "DBG": "DBG",
}

_LOCK = threading.RLock()
_CONSOLE_ENABLED = True
_FILE_ENABLED = False
_FLUSH = True
_LOG_FILE = None
_LOG_PATH: Optional[Path] = None
_SESSION_ID = ""


def _bool_from_cfg(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def init_logging(cfg: Mapping[str, Any] | None = None, *, force_new_file: bool = True) -> Optional[Path]:
    """Initialize project logging.

    v16 defaults are file-first and console-quiet. Every run gets a timestamped
    .txt file under logs/. The function is intentionally dependency-free and is
    safe to call before detector/controller initialization.
    """
    global _CONSOLE_ENABLED, _FILE_ENABLED, _FLUSH, _LOG_FILE, _LOG_PATH, _SESSION_ID

    cfg = cfg or {}
    logging_cfg = dict(cfg.get("logging", {}) or {})

    _CONSOLE_ENABLED = _bool_from_cfg(logging_cfg.get("console"), False)
    _FILE_ENABLED = _bool_from_cfg(logging_cfg.get("file"), True)
    _FLUSH = _bool_from_cfg(logging_cfg.get("flush"), True)

    if not _FILE_ENABLED:
        return None

    with _LOCK:
        if _LOG_FILE is not None and not force_new_file:
            return _LOG_PATH
        if _LOG_FILE is not None:
            try:
                _LOG_FILE.close()
            except Exception:
                pass
            _LOG_FILE = None

        log_dir = Path(str(logging_cfg.get("log_dir", "logs")))
        log_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(logging_cfg.get("file_prefix", "run")).strip() or "run"
        now = _dt.datetime.now()
        _SESSION_ID = now.strftime("%Y%m%d_%H%M%S")
        file_name = f"{prefix}_{_SESSION_ID}.txt"
        _LOG_PATH = log_dir / file_name
        _LOG_FILE = open(_LOG_PATH, "a", encoding="utf-8", newline="\n")
        return _LOG_PATH


def get_log_path() -> Optional[Path]:
    return _LOG_PATH


def get_session_id() -> str:
    return _SESSION_ID


def close_logging() -> None:
    global _LOG_FILE
    with _LOCK:
        if _LOG_FILE is not None:
            try:
                _LOG_FILE.flush()
                _LOG_FILE.close()
            finally:
                _LOG_FILE = None


def _format_line(msg: str, level: str) -> str:
    tag = _LEVEL.get(level.upper(), level.upper())
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    return f"[{ts}] [{tag}] {msg}"


def log(msg: str, level: str = "INFO", *, console: Optional[bool] = None) -> None:
    line = _format_line(str(msg), level)
    with _LOCK:
        should_console = _CONSOLE_ENABLED if console is None else bool(console)
        if should_console:
            print(line, flush=True)
        if _LOG_FILE is not None:
            try:
                _LOG_FILE.write(line + "\n")
                if _FLUSH:
                    _LOG_FILE.flush()
            except Exception:
                # Last-resort fallback. Do not crash realtime control because logging failed.
                try:
                    print(line, file=sys.stderr, flush=True)
                except Exception:
                    pass


def log_block(title: str, text: str, level: str = "INFO") -> None:
    log(f"{title}_BEGIN", level)
    for raw_line in str(text).splitlines():
        log(raw_line, level)
    log(f"{title}_END", level)


def log_exception(title: str = "UNHANDLED_EXCEPTION") -> None:
    log(f"{title}_BEGIN", "ERROR")
    for line in traceback.format_exc().splitlines():
        log(line, "ERROR")
    log(f"{title}_END", "ERROR")


def log_kv(title: str, mapping: Mapping[str, Any], level: str = "INFO") -> None:
    parts = []
    for k in sorted(mapping.keys()):
        try:
            v = mapping[k]
        except Exception:
            v = "<unreadable>"
        parts.append(f"{k}={v}")
    log(f"{title}: " + ", ".join(parts), level)
