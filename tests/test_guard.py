import pytest

from aimtrainer import guard
from aimtrainer.config import GuardConfig
from aimtrainer.guard import BlockedApplicationError, ForegroundWindow, assert_not_blocked, is_allowed, is_blocked


def win(title="", process=""):
    return ForegroundWindow(title=title, process=process)


ALLOW = GuardConfig()


@pytest.mark.parametrize(
    "window",
    [
        win(title="VALORANT  "),
        win(title="valorant"),
        win(title="Counter-Strike 2"),
        win(title="Apex Legends"),
        win(title="Call of Duty(R)"),
        win(title="Overwatch 2"),
        win(title="Tom Clancy's Rainbow Six Siege"),
        win(title="Fortnite  "),
        win(process="VALORANT-Win64-Shipping.exe"),
        win(process="cs2.exe"),
        win(process="r5apex.exe"),
        win(process="FortniteClient-Win64-Shipping.exe"),
    ],
)
def test_live_competitive_games_are_blocked(window):
    assert is_blocked(window) is True
    assert is_allowed(window, ALLOW) is False
    with pytest.raises(BlockedApplicationError):
        assert_not_blocked(window)


@pytest.mark.parametrize(
    "window",
    [
        win(title="Aim Lab", process="AimLab_tb.exe"),
        win(title="aimlab"),
        win(title="KovaaK's FPS Aim Trainer", process="FPSAimTrainer.exe"),
    ],
)
def test_aim_trainers_are_allowed(window):
    assert is_blocked(window) is False
    assert is_allowed(window, ALLOW) is True
    assert_not_blocked(window)  # does not raise


def test_matching_is_case_insensitive():
    assert is_blocked(win(title="VaLoRaNt"))
    assert is_allowed(win(title="AIM LAB"), ALLOW)


def test_blocked_beats_allowed_for_a_window_claiming_both():
    """A title contrived to satisfy the allow list must not smuggle a blocked game through."""
    window = win(title="Aim Lab - VALORANT practice", process="VALORANT.exe")
    assert is_blocked(window) is True
    assert is_allowed(window, ALLOW) is False


def test_the_block_list_also_wins_when_the_check_is_disabled():
    permissive = GuardConfig(require_foreground_match=False)
    assert is_allowed(win(title="Notepad"), permissive) is True
    assert is_allowed(win(title="VALORANT"), permissive) is False


def test_an_unrelated_window_gets_no_input():
    """Fail closed: not blocked is not the same as allowed."""
    window = win(title="Untitled - Notepad", process="notepad.exe")
    assert is_blocked(window) is False
    assert is_allowed(window, ALLOW) is False


def test_an_unreadable_foreground_window_gets_no_input():
    assert is_allowed(None, ALLOW) is False
    assert_not_blocked(None)  # unknown is not blocked, but it is also not allowed


def test_an_empty_allow_list_permits_nothing():
    empty = GuardConfig(allow_window_substrings=[])
    assert is_allowed(win(title="Aim Lab"), empty) is False


def test_the_block_list_covers_the_major_live_shooters():
    joined = " ".join(guard.BLOCKED_TITLES)
    for name in ["valorant", "counter-strike", "apex legends", "call of duty", "overwatch", "fortnite"]:
        assert name in joined


def test_the_block_list_is_module_level_not_config_driven():
    """It must not be reachable from a config file."""
    assert not any("block" in f for f in GuardConfig.__dataclass_fields__)


def test_the_error_message_explains_why():
    with pytest.raises(BlockedApplicationError) as excinfo:
        assert_not_blocked(win(title="VALORANT", process="VALORANT.exe"))
    message = str(excinfo.value).lower()
    assert "aim trainer" in message and "refusing" in message
