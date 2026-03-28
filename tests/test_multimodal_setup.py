import cv2
import numpy as np

from tests import TMP


def _write_image(path, shape=(8, 10, 3), value=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full(shape, value, dtype=np.uint8)
    assert cv2.imwrite(str(path), image)


def test_resolve_paired_image_path():
    from ultralytics.data.multimodal import resolve_paired_image_path

    image_path = TMP / "datasets" / "images" / "train" / "sample.jpg"
    paired = resolve_paired_image_path(
        image_path,
        {
            "pair_visible_dir": "images",
            "pair_infrared_dir": "image",
        },
    )

    assert paired == TMP / "datasets" / "image" / "train" / "sample.jpg"


def test_split_modalities_honors_channel_order():
    from ultralytics.data.multimodal import split_modalities

    fused = np.stack([np.full((2, 2), i, dtype=np.uint8) for i in range(6)], axis=-1)

    vi, ir = split_modalities(fused, "vi_ir")
    assert np.all(vi[..., 0] == 0)
    assert np.all(ir[..., 0] == 3)

    vi_alt, ir_alt = split_modalities(fused, "ir_vi")
    assert np.all(vi_alt[..., 0] == 3)
    assert np.all(ir_alt[..., 0] == 0)


def test_collect_pairing_issues_detects_missing_and_shape_mismatch():
    from ultralytics.data.multimodal import collect_pairing_issues

    data_root = TMP / "pair_check"
    visible_a = data_root / "images" / "train" / "a.jpg"
    visible_b = data_root / "images" / "train" / "b.jpg"
    infrared_a = data_root / "image" / "train" / "a.jpg"
    infrared_b = data_root / "image" / "train" / "b.jpg"

    _write_image(visible_a, shape=(8, 10, 3), value=32)
    _write_image(visible_b, shape=(8, 10, 3), value=64)
    _write_image(infrared_a, shape=(7, 10, 3), value=96)

    issues = collect_pairing_issues(
        [visible_a, visible_b],
        {
            "pair_visible_dir": "images",
            "pair_infrared_dir": "image",
        },
        check_shapes=True,
    )

    assert any("shape mismatch" in issue for issue in issues)
    assert any("missing infrared pair" in issue for issue in issues)
    assert not infrared_b.exists()


def test_experiment_registry_exposes_baselines():
    from experiment_config import get_experiment

    baseline = get_experiment("baseline_ir")
    assert baseline.input_modality == "infrared"
    assert str(baseline.weights).endswith("results/baseline_ir/weights/best.pt")

    early_fusion = get_experiment("baseline_early_fusion")
    assert early_fusion.input_modality == "multimodal"
    assert early_fusion.channel_order == "vi_ir"
