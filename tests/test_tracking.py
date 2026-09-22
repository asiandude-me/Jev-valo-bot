import pytest

from aimtrainer.config import TrackingConfig
from aimtrainer.detectors.base import Detection
from aimtrainer.tracking import Tracker


def det(x, y, size=30.0, score=1.0):
    return Detection(cx=x, cy=y, width=size, height=size, score=score)


def test_identity_is_stable_for_a_target_moving_smoothly():
    tracker = Tracker(TrackingConfig())
    ids = set()
    for i in range(12):
        tracks = tracker.update([det(100 + 5 * i, 100)], i / 240.0)
        assert len(tracks) == 1
        ids.add(tracks[0].track_id)
    assert len(ids) == 1


def test_velocity_matches_the_true_motion():
    tracker = Tracker(TrackingConfig())
    dt = 1 / 240.0
    speed_px_per_frame = 6.0
    for i in range(30):
        tracker.update([det(50 + speed_px_per_frame * i, 80)], i * dt)
    track = tracker.tracks[0]
    assert track.vx == pytest.approx(speed_px_per_frame / dt, rel=0.05)
    assert track.vy == pytest.approx(0.0, abs=1.0)


def test_prediction_extrapolates_along_the_velocity():
    tracker = Tracker(TrackingConfig())
    dt = 1 / 240.0
    for i in range(30):
        tracker.update([det(50 + 6.0 * i, 80)], i * dt)
    track = tracker.tracks[0]
    px, py = track.predict(0.010)
    assert px > track.cx
    assert px == pytest.approx(track.cx + track.vx * 0.010, rel=1e-6)
    assert py == pytest.approx(track.cy, abs=0.5)


def test_a_jump_beyond_the_gate_starts_a_new_track():
    tracker = Tracker(TrackingConfig(max_assign_px=50.0))
    first = tracker.update([det(100, 100)], 0.0)[0]
    after = tracker.update([det(300, 100)], 1 / 240.0)
    assert {t.track_id for t in after} != {first.track_id}
    assert any(t.track_id != first.track_id for t in after)


def test_two_targets_keep_separate_identities_while_converging():
    tracker = Tracker(TrackingConfig(max_assign_px=40.0))
    a = tracker.update([det(60, 100), det(240, 100)], 0.0)
    ids = sorted(t.track_id for t in a)
    for i in range(1, 8):
        tracks = tracker.update([det(60 + 8 * i, 100), det(240 - 8 * i, 100)], i / 240.0)
    assert sorted(t.track_id for t in tracks) == ids


def test_a_track_survives_a_brief_dropout_then_expires():
    cfg = TrackingConfig(max_missed_frames=3)
    tracker = Tracker(cfg)
    original = tracker.update([det(100, 100)], 0.0)[0].track_id

    for i in range(1, 4):  # unseen, but still within the grace period
        tracker.update([], i / 240.0)
    assert [t.track_id for t in tracker.tracks] == [original]

    tracker.update([], 4 / 240.0)
    assert tracker.tracks == []


def test_reappearing_inside_the_gate_reuses_the_same_id():
    tracker = Tracker(TrackingConfig(max_missed_frames=3, max_assign_px=60.0))
    original = tracker.update([det(100, 100)], 0.0)[0].track_id
    tracker.update([], 1 / 240.0)
    tracks = tracker.update([det(110, 105)], 2 / 240.0)
    assert [t.track_id for t in tracks] == [original]


def test_duplicate_timestamps_do_not_produce_infinite_velocity():
    """A repeated frame or a coarse clock must not fling the crosshair."""
    tracker = Tracker(TrackingConfig())
    tracker.update([det(100, 100)], 0.0)
    tracker.update([det(150, 100)], 0.0)
    track = tracker.tracks[0]
    assert track.vx == 0.0
    assert track.cx == 150.0  # position still updates


def test_established_tracks_win_contested_detections():
    """A long-lived track should claim the nearby detection before a one-frame track."""
    tracker = Tracker(TrackingConfig(max_assign_px=100.0))
    for i in range(10):
        tracker.update([det(100, 100)], i / 240.0)
    established = tracker.tracks[0].track_id

    tracker.update([det(100, 100), det(160, 100)], 10 / 240.0)
    tracker.update([det(105, 100)], 11 / 240.0)

    survivors = [t for t in tracker.tracks if t.missed == 0]
    assert [t.track_id for t in survivors] == [established]


def test_history_is_bounded():
    tracker = Tracker(TrackingConfig())
    for i in range(200):
        tracker.update([det(100 + (i % 3), 100)], i / 240.0)
    assert len(tracker.tracks[0].history) <= 16


def test_reset_clears_everything():
    tracker = Tracker(TrackingConfig())
    tracker.update([det(100, 100)], 0.0)
    tracker.reset()
    assert tracker.tracks == []
