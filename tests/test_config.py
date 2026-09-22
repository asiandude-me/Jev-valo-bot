import json
import textwrap

import pytest

from aimtrainer.config import Config, HSVRange, to_dict


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(textwrap.dedent(text), encoding="utf-8")
    return p


def test_defaults_are_self_consistent():
    cfg = Config()
    assert cfg.detector.backend == "color"
    assert cfg.trigger.enabled is False          # aim assist only, by default
    assert cfg.guard.require_foreground_match is True
    assert cfg.aim.compensation_frames >= 1


def test_loads_yaml_and_overrides_only_what_is_given(tmp_path):
    path = write(tmp_path, "c.yaml", """
        aim:
          kp: 0.8
          deg_per_count: 0.0301
        capture:
          roi_width: 256
    """)
    cfg = Config.load(path)
    assert cfg.aim.kp == 0.8
    assert cfg.aim.deg_per_count == 0.0301
    assert cfg.capture.roi_width == 256
    assert cfg.capture.roi_height == 320          # untouched default
    assert cfg.detector.backend == "color"


def test_loads_json(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps({"aim": {"kp": 0.25}}), encoding="utf-8")
    assert Config.load(path).aim.kp == 0.25


def test_hsv_ranges_become_typed_objects(tmp_path):
    path = write(tmp_path, "c.yaml", """
        detector:
          hsv_ranges:
            - h_min: 0
              h_max: 10
              s_min: 200
            - h_min: 170
              h_max: 179
    """)
    ranges = Config.load(path).detector.hsv_ranges
    assert all(isinstance(r, HSVRange) for r in ranges)
    assert ranges[0].s_min == 200
    assert ranges[0].v_min == 120                 # default fills in
    assert ranges[1].h_min == 170


def test_a_typo_is_an_error_rather_than_a_silent_no_op(tmp_path):
    path = write(tmp_path, "c.yaml", """
        aim:
          kpp: 0.8
    """)
    with pytest.raises(ValueError) as excinfo:
        Config.load(path)
    assert "kpp" in str(excinfo.value)
    assert "kp" in str(excinfo.value)              # suggests the real options


def test_an_unknown_top_level_section_is_an_error(tmp_path):
    path = write(tmp_path, "c.yaml", "aimbot:\n  kp: 1.0\n")
    with pytest.raises(ValueError):
        Config.load(path)


def test_an_empty_file_gives_defaults(tmp_path):
    path = write(tmp_path, "c.yaml", "")
    assert Config.load(path).aim.kp == Config().aim.kp


def test_a_non_mapping_top_level_is_rejected(tmp_path):
    path = write(tmp_path, "c.yaml", "- 1\n- 2\n")
    with pytest.raises(ValueError):
        Config.load(path)


def test_to_dict_round_trips_through_a_file(tmp_path):
    original = Config()
    path = tmp_path / "c.json"
    path.write_text(json.dumps(to_dict(original)), encoding="utf-8")
    assert to_dict(Config.load(path)) == to_dict(original)


def test_the_shipped_config_parses_and_matches_the_defaults():
    """configs/aimlabs_gridshot.yaml is the documented entry point; keep it valid."""
    cfg = Config.load("configs/aimlabs_gridshot.yaml")
    assert cfg.detector.backend == "color"
    assert cfg.trigger.enabled is False
    assert "aimlab" in cfg.guard.allow_window_substrings
    assert cfg.aim.compensation_frames == Config().aim.compensation_frames
    assert cfg.aim.kp == Config().aim.kp
