"""Relative mouse output.

Games read *raw* input, so a cursor-teleport (``SetCursorPos``) does nothing to your
view -- only relative motion events register. On Windows that means ``SendInput`` with
``MOUSEEVENTF_MOVE`` and no ``MOUSEEVENTF_ABSOLUTE``.

Windows also applies its pointer-acceleration curve ("Enhance pointer precision") to
these events. The controller's model assumes a linear device, so turn that off; the
calibrator's variance report will tell you if you forgot.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Protocol


class MouseBackend(Protocol):
    def move_relative(self, dx: int, dy: int) -> None:
        ...

    def click(self) -> None:
        ...

    def describe(self) -> str:
        ...


@dataclass
class DryRunMouse:
    """Records commands instead of sending them. Used by ``--dry-run`` and the tests."""

    moves: list[tuple[int, int]] = field(default_factory=list)
    clicks: int = 0

    def move_relative(self, dx: int, dy: int) -> None:
        if dx or dy:
            self.moves.append((dx, dy))

    def click(self) -> None:
        self.clicks += 1

    def describe(self) -> str:
        return "dry-run (no input sent)"

    @property
    def total(self) -> tuple[int, int]:
        return sum(m[0] for m in self.moves), sum(m[1] for m in self.moves)


class Win32Mouse:  # pragma: no cover - Windows only
    """SendInput-based relative movement and left click."""

    MOUSEEVENTF_MOVE = 0x0001
    MOUSEEVENTF_LEFTDOWN = 0x0002
    MOUSEEVENTF_LEFTUP = 0x0004
    INPUT_MOUSE = 0

    def __init__(self) -> None:
        import ctypes
        from ctypes import wintypes

        self._ctypes = ctypes
        ULONG_PTR = ctypes.POINTER(ctypes.c_ulong)

        class MOUSEINPUT(ctypes.Structure):
            _fields_ = [
                ("dx", wintypes.LONG),
                ("dy", wintypes.LONG),
                ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("dwExtraInfo", ULONG_PTR),
            ]

        class _INPUTUNION(ctypes.Union):
            _fields_ = [("mi", MOUSEINPUT)]

        class INPUT(ctypes.Structure):
            _anonymous_ = ("u",)
            _fields_ = [("type", wintypes.DWORD), ("u", _INPUTUNION)]

        self._MOUSEINPUT = MOUSEINPUT
        self._INPUT = INPUT
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._user32.SendInput.argtypes = (wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int)
        self._user32.SendInput.restype = wintypes.UINT

    def _send(self, dx: int, dy: int, flags: int) -> None:
        inp = self._INPUT(type=self.INPUT_MOUSE)
        inp.mi = self._MOUSEINPUT(dx, dy, 0, flags, 0, None)
        sent = self._user32.SendInput(1, self._ctypes.byref(inp), self._ctypes.sizeof(inp))
        if sent != 1:
            raise OSError(
                f"SendInput was rejected (error {self._ctypes.get_last_error()}). "
                "A foreground window running as administrator will block input from a "
                "non-elevated process."
            )

    def move_relative(self, dx: int, dy: int) -> None:
        if dx or dy:
            self._send(int(dx), int(dy), self.MOUSEEVENTF_MOVE)

    def click(self) -> None:
        self._send(0, 0, self.MOUSEEVENTF_LEFTDOWN)
        self._send(0, 0, self.MOUSEEVENTF_LEFTUP)

    def describe(self) -> str:
        return "win32 SendInput (relative)"


class PynputMouse:  # pragma: no cover - needs a live display
    """Generic fallback. Fine for desktop testing; many games ignore it."""

    def __init__(self) -> None:
        from pynput.mouse import Button, Controller

        self._button = Button
        self._mouse = Controller()

    def move_relative(self, dx: int, dy: int) -> None:
        if dx or dy:
            self._mouse.move(int(dx), int(dy))

    def click(self) -> None:
        self._mouse.click(self._button.left)

    def describe(self) -> str:
        return "pynput (relative)"


def build_mouse(dry_run: bool = False) -> MouseBackend:
    if dry_run:
        return DryRunMouse()
    if sys.platform.startswith("win"):
        return Win32Mouse()
    try:
        return PynputMouse()
    except Exception:
        return DryRunMouse()
