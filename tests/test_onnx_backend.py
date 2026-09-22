"""The ONNX detector, exercised against real ONNX Runtime sessions.

A tiny hand-built model emits a chosen prediction tensor, so the decode path -- output
layout, letterbox inverse, confidence and class filtering, NMS -- is checked against
known-correct answers without needing a trained checkpoint.
"""

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from aimtrainer.config import DetectorConfig  # noqa: E402
from aimtrainer.detectors.onnx_yolo import OnnxYoloDetector  # noqa: E402


def build_model(path, predictions, layout="v8", input_size=320):
    """Write an ONNX model whose single output is ``predictions``.

    ``predictions`` is (N, 4+nc). 'v8' emits it as (1, 4+nc, N); 'v5' as (1, N, 4+nc).
    """
    array = np.asarray(predictions, dtype=np.float32)
    payload = array.T[None] if layout == "v8" else array[None]

    tensor = onnx.helper.make_tensor(
        name="preds", data_type=onnx.TensorProto.FLOAT,
        dims=list(payload.shape), vals=payload.flatten().tolist(),
    )
    node = onnx.helper.make_node("Constant", inputs=[], outputs=["output0"], value=tensor)
    graph = onnx.helper.make_graph(
        [node], "fake_yolo",
        inputs=[onnx.helper.make_tensor_value_info(
            "images", onnx.TensorProto.FLOAT, [1, 3, input_size, input_size])],
        outputs=[onnx.helper.make_tensor_value_info(
            "output0", onnx.TensorProto.FLOAT, list(payload.shape))],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 12)])
    model.ir_version = 9
    onnx.save(model, str(path))
    return str(path)


def frame(w=320, h=320):
    return np.zeros((h, w, 3), dtype=np.uint8)


def detector_for(tmp_path, predictions, layout="v8", cfg_layout="auto", **cfg_kwargs):
    """`layout` is how the fake model emits its tensor; `cfg_layout` is what we tell
    the detector to expect."""
    path = build_model(tmp_path / "m.onnx", predictions, layout)
    cfg = DetectorConfig(backend="onnx", model_path=path, input_size=320,
                         layout=cfg_layout, providers=["CPUExecutionProvider"], **cfg_kwargs)
    return OnnxYoloDetector(cfg)


# cx, cy, w, h, class-0 score
ONE_BOX = [[200.0, 120.0, 40.0, 40.0, 0.90]]


def test_a_square_frame_maps_straight_through(tmp_path):
    found = detector_for(tmp_path, ONE_BOX).detect(frame())
    assert len(found) == 1
    assert found[0].cx == pytest.approx(200.0, abs=0.5)
    assert found[0].cy == pytest.approx(120.0, abs=0.5)
    assert found[0].width == pytest.approx(40.0, abs=0.5)
    assert found[0].score == pytest.approx(0.90, abs=1e-5)


def test_letterbox_padding_is_undone_for_a_wide_frame(tmp_path):
    """A 320x160 ROI is padded by 80px top and bottom; that must be subtracted back."""
    found = detector_for(tmp_path, ONE_BOX).detect(frame(320, 160))
    assert len(found) == 1
    assert found[0].cx == pytest.approx(200.0, abs=0.5)
    assert found[0].cy == pytest.approx(120.0 - 80.0, abs=0.5)


def test_both_export_layouts_give_the_same_answer(tmp_path):
    """The same detection, exported either way, must decode identically."""
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    v8 = detector_for(a, ONE_BOX, layout="v8", cfg_layout="v8").detect(frame())
    v5_preds = [[200.0, 120.0, 40.0, 40.0, 1.0, 0.90]]  # objectness, then class score
    v5 = detector_for(b, v5_preds, layout="v5", cfg_layout="v5").detect(frame())
    assert len(v8) == len(v5) == 1
    assert v8[0].cx == pytest.approx(v5[0].cx, abs=0.5)
    assert v8[0].score == pytest.approx(v5[0].score, abs=1e-4)


def test_declaring_v5_folds_the_objectness_into_the_score(tmp_path):
    preds = [[200.0, 120.0, 40.0, 40.0, 0.80, 0.90]]
    found = detector_for(tmp_path, preds, layout="v5", cfg_layout="v5").detect(frame())
    assert found[0].score == pytest.approx(0.80 * 0.90, abs=1e-4)


def test_declaring_v8_treats_every_trailing_column_as_a_class(tmp_path):
    preds = [[200.0, 120.0, 40.0, 40.0, 0.80, 0.90]]
    found = detector_for(tmp_path, preds, layout="v5", cfg_layout="v8").detect(frame())
    assert found[0].score == pytest.approx(0.90, abs=1e-4)
    assert found[0].class_id == 1


def test_auto_layout_spots_objectness_across_a_full_candidate_set(tmp_path):
    """The heuristic is meant for a real model's thousands of mostly-background rows."""
    rng = np.random.default_rng(0)
    n = 500
    preds = np.zeros((n, 6), dtype=np.float32)
    preds[:, :4] = rng.uniform(20, 300, size=(n, 4))
    preds[:, 4] = rng.uniform(0.55, 0.95, size=n)   # objectness: consistently high
    preds[:, 5] = rng.uniform(0.0, 0.25, size=n)    # class score: mostly low
    preds[0, 4], preds[0, 5] = 0.95, 0.95           # one real detection
    preds[0, :4] = [200.0, 120.0, 40.0, 40.0]

    found = detector_for(tmp_path, preds, layout="v5").detect(frame())
    assert len(found) == 1
    assert found[0].score == pytest.approx(0.95 * 0.95, abs=1e-3)


def test_an_invalid_layout_is_rejected(tmp_path):
    path = build_model(tmp_path / "m.onnx", ONE_BOX)
    cfg = DetectorConfig(backend="onnx", model_path=path, layout="v9",
                         providers=["CPUExecutionProvider"])
    with pytest.raises(ValueError, match="detector.layout"):
        OnnxYoloDetector(cfg)


def test_low_confidence_boxes_are_dropped(tmp_path):
    preds = [[100.0, 100.0, 30.0, 30.0, 0.90], [200.0, 200.0, 30.0, 30.0, 0.10]]
    found = detector_for(tmp_path, preds, conf_threshold=0.45).detect(frame())
    assert len(found) == 1
    assert found[0].cx == pytest.approx(100.0, abs=0.5)


def test_everything_below_threshold_yields_nothing(tmp_path):
    preds = [[100.0, 100.0, 30.0, 30.0, 0.10]]
    assert detector_for(tmp_path, preds, conf_threshold=0.45).detect(frame()) == []


def test_nms_collapses_duplicate_boxes(tmp_path):
    preds = [
        [150.0, 150.0, 50.0, 50.0, 0.90],
        [152.0, 151.0, 50.0, 50.0, 0.85],  # almost the same box
        [ 40.0,  40.0, 30.0, 30.0, 0.80],  # clearly separate
    ]
    found = detector_for(tmp_path, preds, nms_threshold=0.5).detect(frame())
    assert len(found) == 2


def test_class_filtering_keeps_only_the_requested_ids(tmp_path):
    preds = [
        [100.0, 100.0, 30.0, 30.0, 0.90, 0.10],  # argmax -> class 0
        [200.0, 200.0, 30.0, 30.0, 0.10, 0.95],  # argmax -> class 1
    ]
    found = detector_for(tmp_path, preds, class_ids=[1]).detect(frame())
    assert len(found) == 1
    assert found[0].class_id == 1
    assert found[0].cx == pytest.approx(200.0, abs=0.5)


def test_multiple_classes_report_the_argmax_class(tmp_path):
    preds = [[100.0, 100.0, 30.0, 30.0, 0.20, 0.30, 0.88]]
    found = detector_for(tmp_path, preds).detect(frame())
    assert len(found) == 1 and found[0].class_id == 2


def test_tall_boxes_get_the_upper_third_aim_point(tmp_path):
    preds = [[160.0, 160.0, 30.0, 120.0, 0.90]]  # aspect 4:1, humanoid-ish
    found = detector_for(tmp_path, preds).detect(frame())
    assert found[0].cy == pytest.approx(160.0 - 120.0 / 2 + 120.0 / 3, abs=0.5)


def test_square_boxes_keep_their_centre(tmp_path):
    found = detector_for(tmp_path, ONE_BOX).detect(frame())
    assert found[0].cy == pytest.approx(120.0, abs=0.5)


def test_describe_names_the_execution_provider(tmp_path):
    text = detector_for(tmp_path, ONE_BOX).describe()
    assert "onnx" in text and "CPUExecutionProvider" in text and "layout" in text


def test_a_missing_model_says_how_to_export_one():
    cfg = DetectorConfig(backend="onnx", model_path="definitely/not/here.onnx")
    with pytest.raises(FileNotFoundError, match="export_yolo_onnx"):
        OnnxYoloDetector(cfg)


def test_an_unavailable_provider_falls_back_instead_of_failing(tmp_path):
    """A config naming a GPU provider must still run on a machine without one."""
    path = build_model(tmp_path / "m.onnx", ONE_BOX)
    cfg = DetectorConfig(backend="onnx", model_path=path, providers=["TotallyFakeProvider"])
    detector = OnnxYoloDetector(cfg)
    assert detector.providers == ["CPUExecutionProvider"]
    assert len(detector.detect(frame())) == 1


def test_a_small_detection_count_does_not_confuse_the_layout(tmp_path):
    """Shape alone cannot tell the layouts apart when a frame yields few boxes."""
    preds = [[200.0, 120.0, 40.0, 40.0, 0.9], [60.0, 60.0, 20.0, 20.0, 0.8]]
    for layout in ("v8", "v5"):
        d = tmp_path / layout
        d.mkdir()
        found = detector_for(d, preds, layout=layout).detect(frame())
        assert len(found) == 2, layout
        assert sorted(round(f.cx) for f in found) == [60, 200], layout


def test_a_nonsense_output_shape_is_reported_clearly(tmp_path):
    from aimtrainer.detectors.onnx_yolo import _normalize_predictions

    with pytest.raises(ValueError, match="does not look like YOLO detections"):
        _normalize_predictions(np.zeros((1, 3, 3), dtype=np.float32))
