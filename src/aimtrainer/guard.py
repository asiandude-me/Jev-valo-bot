"""Scope enforcement: this bot drives aim trainers, and nothing else.

Two independent checks, both applied before any synthetic input leaves the process:

1. A **block list** of live competitive shooters. Aim assistance in those is cheating
   against real opponents. If one of them owns the foreground window the runner
   refuses to start and exits -- there is no config flag for this.
2. An **allow list** of trainer windows. It fails closed: an unrecognised foreground
   window, or a foreground window we could not read at all, means no input.

The window query is Windows-only. On other platforms it returns ``None``, which the
fail-closed rule turns into "no input" -- so the pipeline can be developed and tested
anywhere, but only ever *acts* where it can verify what it is acting on.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

from .config import GuardConfig

# Live multiplayer shooters. Matched as substrings against both the foreground window
# title and the process image name.
BLOCKED_TITLES: tuple[str, ...] = (
    "valorant",
    "counter-strike",
    "cs2",
    "apex legends",
    "call of duty",
    "warzone",
    "overwatch",
    "rainbow six",
    "battlefield",
    "fortnite",
    "destiny 2",
    "pubg",
    "the finals",
    "escape from tarkov",
    "delta force",
    "marvel rivals",
    "splitgate",
    "team fortress",
    "paladins",
    "xdefiant",
)

BLOCKED_PROCESSES: tuple[str, ...] = (
    "valorant.exe",
    "valorant-win64-shipping.exe",
    "riotclientservices.exe",
    "cs2.exe",
    "csgo.exe",
    "r5apex.exe",
    "modernwarfare.exe",
    "cod.exe",
    "overwatch.exe",
    "rainbowsix.exe",
    "rainbowsixgame.exe",
    "bf2042.exe",
    "fortniteclient-win64-shipping.exe",
    "destiny2.exe",
    "tslgame.exe",
    "discovery.exe",
    "eftclient.exe",
    "escapefromtarkov.exe",
    "marvel-win64-shipping.exe",
)


class BlockedApplicationError(RuntimeError):
    """Raised when a live competitive game is in the foreground."""


@dataclass(frozen=True)
class ForegroundWindow:
    title: str
    process: str

    def haystack(self) -> str:
        return f"{self.title}\n{self.process}".lower()


def foreground_window() -> ForegroundWindow | None:
    """Title and process image name of the focused window, or None if unavailable."""
    if not sys.platform.startswith("win"):
        return None
    return _foreground_window_win32()


def _foreground_window_win32() -> ForegroundWindow | None:  # pragma: no cover - Windows only
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return None

    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    title = buf.value

    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

    process = ""
    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if handle:
        try:
            size = wintypes.DWORD(260)
            name = ctypes.create_unicode_buffer(size.value)
            if kernel32.QueryFullProcessImageNameW(handle, 0, name, ctypes.byref(size)):
                process = name.value.rsplit("\\", 1)[-1]
        finally:
            kernel32.CloseHandle(handle)

    return ForegroundWindow(title=title, process=process)


def is_blocked(window: ForegroundWindow | None) -> bool:
    """True if the foreground window is a live competitive shooter."""
    if window is None:
        return False
    hay = window.haystack()
    proc = window.process.lower()
    return any(b in hay for b in BLOCKED_TITLES) or any(b == proc or b in proc for b in BLOCKED_PROCESSES)


def is_allowed(window: ForegroundWindow | None, cfg: GuardConfig) -> bool:
    """True if input may be sent to this window right now.

    Blocked always wins over allowed, so a window contriving to contain both an
    allow-list and a block-list substring is still refused.
    """
    if is_blocked(window):
        return False
    if not cfg.require_foreground_match:
        return True
    if window is None:
        return False  # fail closed: unknown window, no input
    if not cfg.allow_window_substrings:
        return False
    hay = window.haystack()
    return any(s.lower() in hay for s in cfg.allow_window_substrings)


def assert_not_blocked(window: ForegroundWindow | None) -> None:
    """Hard stop for the runner's startup and per-frame checks."""
    if is_blocked(window):
        name = (window.title or window.process) if window else "unknown"
        raise BlockedApplicationError(
            f"{name!r} is a live competitive game. This tool drives aim trainers only; "
            "using it against real opponents is cheating. Refusing to send input."
        )
