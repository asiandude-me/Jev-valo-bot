import numpy as np
import pytest

from aimtrainer.config import DetectorConfig, HSVRange
from aimtrainer.detectors.color import ColorDetector, build_mask, clean_mask
from aimtrainer.detectors.onnx_yolo import _normalize_predictions, letterbox
from aimtrainer.synthetic import SyntheticTarget, render_frame


@pytest.fixture
def detector():
    return ColorDetector(DetectorConfig())


def test_finds_a_single_target_at_the_right_place(detector):
    frame = render_frame(320, 320, [SyntheticTarget(200.0, 120.0, 18.0)], crosshair=False)
    found = detector.detect(frame)
    assert len(found) == 1
    assert found[0].cx == pytest.approx(200.0, abs=1.5)
    assert found[0].cy == pytest.approx(120.0, abs=1.5)
    assert found[0].width == pytest.approx(36.0, abs=3.0)


def test_finds_every_target_in_a_crowded_frame(detector):
    truth = [
        SyntheticTarget(60.0, 60.0, 16.0),
        SyntheticTarget(160.0, 160.0, 22.0),
        SyntheticTarget(260.0, 90.0, 14.0),
    ]
    found = detector.detect(render_frame(320, 320, truth, crosshair=False))
    assert len(found) == 3
    for gt in truth:
        assert min(abs(d.cx - gt.x) + abs(d.cy - gt.y) for d in found) < 3.0


def test_crosshair_overlay_does_not_split_a_target(detector):
    """A crosshair drawn across the centre target punches a hole; closing fills it."""
    frame = render_frame(320, 320, [SyntheticTarget(160.0, 160.0, 20.0)], crosshair=True)
    found = detector.detect(frame)
    assert len(found) == 1
    assert found[0].cx == pytest.approx(160.0, abs=2.0)


def test_survives_sensor_noise(detector):
    frame = render_frame(320, 320, [SyntheticTarget(100.0, 220.0, 17.0)], noise=8.0, seed=3, crosshair=False)
    found = detector.detect(frame)
    assert len(found) == 1
    assert found[0].cx == pytest.approx(100.0, abs=2.5)


def test_background_only_frame_yields_nothing(detector):
    assert detector.detect(render_frame(320, 320, [], crosshair=True)) == []


def test_off_colour_targets_are_ignored(detector):
    frame = render_frame(320, 320, [SyntheticTarget(160.0, 160.0, 20.0)], target_bgr=(40, 190, 60), crosshair=False)
    assert detector.detect(frame) == []


def test_small_blobs_are_filtered_out():
    cfg = DetectorConfig(min_area_px=2000, open_kernel=0, close_kernel=0)
    frame = render_frame(320, 320, [SyntheticTarget(160.0, 160.0, 8.0)], crosshair=False)
    assert ColorDetector(cfg).detect(frame) == []


def test_thin_streaks_are_rejected_by_aspect_ratio():
    frame = render_frame(320, 320, [], crosshair=False)
    frame[150:156, 20:300] = (200, 20, 220)  # a long magenta bar, e.g. a UI element
    assert ColorDetector(DetectorConfig()).detect(frame) == []


def test_hue_range_wraps_around_for_red():
    """h_min > h_max means the band crosses OpenCV's 0/179 hue seam."""
    frame = render_frame(64, 64, [SyntheticTarget(32.0, 32.0, 14.0)], target_bgr=(30, 30, 220), crosshair=False)
    wrapping = build_mask(frame, [HSVRange(h_min=170, h_max=10)])
    non_wrapping = build_mask(frame, [HSVRange(h_min=140, h_max=170)])
    assert wrapping.max() == 255
    assert non_wrapping.max() == 0


def test_multiple_bands_are_unioned():
    frame = render_frame(96, 48, [], crosshair=False)
    frame[10:38, 6:34] = (30, 30, 220)    # red
    frame[10:38, 60:88] = (220, 40, 30)   # blue
    ranges = [HSVRange(h_min=170, h_max=10), HSVRange(h_min=100, h_max=130)]
    assert build_mask(frame, ranges).sum() > build_mask(frame, ranges[:1]).sum()


def test_cleaning_removes_speckle_but_keeps_the_disc():
    frame = render_frame(160, 160, [SyntheticTarget(80.0, 80.0, 20.0)], crosshair=False)
    frame[10, 10] = frame[20, 140] = (200, 20, 220)  # isolated single pixels
    raw = build_mask(frame, [HSVRange(h_min=140, h_max=170)])
    cleaned = clean_mask(raw, 3, 5)
    assert cleaned[10, 10] == 0 and cleaned[20, 140] == 0
    assert cleaned[80, 80] == 255


def test_empty_hsv_ranges_is_an_error():
    with pytest.raises(ValueError):
        ColorDetector(DetectorConfig(hsv_ranges=[]))


def test_tall_targets_get_an_upper_third_aim_point(detector):
    """Upright targets are aimed at the head, not the centroid."""
    frame = render_frame(200, 240, [], crosshair=False)
    frame[40:128, 90:130] = (200, 20, 220)  # 40x88 upright box, aspect 2.2
    found = detector.detect(frame)
    assert len(found) == 1
    assert found[0].cy == pytest.approx(40 + 88 / 3, abs=3.0)
    assert found[0].cy < (40 + 128) / 2  # strictly above the centroid


def test_round_targets_keep_their_centroid(detector):
    """The head bias must be a no-op on a sphere, which is the common case."""
    frame = render_frame(200, 200, [SyntheticTarget(100.0, 100.0, 24.0)], crosshair=False)
    found = detector.detect(frame)
    assert len(found) == 1
    assert found[0].cy == pytest.approx(100.0, abs=1.5)


# --- ONNX helpers (no model or onnxruntime needed) --------------------------------

def test_letterbox_pads_without_distorting_aspect_ratio():
    image = np.zeros((90, 180, 3), dtype=np.uint8)
    padded, scale, dx, dy = letterbox(image, 320)
    assert padded.shape == (320, 320, 3)
    assert scale == pytest.approx(320 / 180)
    assert dx == 0 and dy > 0
    assert (padded[0, 0] == 114).all()  # the pad colour, not image content


def test_letterbox_offsets_invert_back_to_source_coordinates():
    image = np.zeros((100, 200, 3), dtype=np.uint8)
    _, scale, dx, dy = letterbox(image, 320)
    for sx, sy in [(0.0, 0.0), (200.0, 100.0), (73.0, 41.0)]:
        assert ((sx * scale + dx) - dx) / scale == pytest.approx(sx)
        assert ((sy * scale + dy) - dy) / scale == pytest.approx(sy)


def test_normalize_predictions_handles_both_export_layouts():
    v8 = np.zeros((1, 6, 1000), dtype=np.float32)      # (batch, 4+nc, N)
    v5 = np.zeros((1, 1000, 6), dtype=np.float32)      # (batch, N, 5+nc)
    assert _normalize_predictions(v8).shape == (1000, 6)
    assert _normalize_predictions(v5).shape == (1000, 6)


def test_normalize_predictions_rejects_unexpected_ranks():
    with pytest.raises(ValueError):
        _normalize_predictions(np.zeros((1, 2, 3, 4), dtype=np.float32))
