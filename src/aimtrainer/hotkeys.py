"""Global hotkey polling, without an extra dependency.

``GetAsyncKeyState`` is polled once per loop iteration rather than hooked, which keeps
it off the hot path -- a keyboard hook would add a callback thread and the associated
lock traffic to a loop whose whole budget is a few milliseconds.
"""

from __future__ import annotations

import sys

VK_CODES = {
    **{f"F{i}": 0x70 + i - 1 for i in range(1, 25)},
    "ESC": 0x1B,
    "SPACE": 0x20,
    "TAB": 0x09,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "ALT": 0x12,
    "CAPSLOCK": 0x14,
    "INSERT": 0x2D,
    "DELETE": 0x2E,
    "HOME": 0x24,
    "END": 0x23,
    "PAGEUP": 0x21,
    "PAGEDOWN": 0x22,
    "MOUSE4": 0x05,
    "MOUSE5": 0x06,
}


def vk_code(name: str) -> int:
    key = name.strip().upper()
    if key in VK_CODES:
        return VK_CODES[key]
    if len(key) == 1 and (key.isalpha() or key.isdigit()):
        return ord(key)
    raise ValueError(f"unknown hotkey {name!r}; try F1-F12, a letter, a digit, or ESC")


class HotkeyWatcher:
    """Edge-triggered polling: :meth:`pressed` fires once per physical press."""

    def __init__(self, names: dict[str, str]) -> None:
        self.codes = {label: vk_code(n) for label, n in names.items()}
        self._down: dict[str, bool] = {label: False for label in self.codes}
        self._user32 = None
        if sys.platform.startswith("win"):  # pragma: no cover - Windows only
            import ctypes

            self._user32 = ctypes.WinDLL("user32", use_last_error=True)

    @property
    def available(self) -> bool:
        return self._user32 is not None

    def pressed(self, label: str) -> bool:
        if self._user32 is None:
            return False
        # The high bit of the return value is "currently down"; the low bit is a
        # sticky "was pressed since last call" that other pollers can steal, so only
        # the high bit is trustworthy.
        down = bool(self._user32.GetAsyncKeyState(self.codes[label]) & 0x8000)
        fired = down and not self._down[label]
        self._down[label] = down
        return fired
