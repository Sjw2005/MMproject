from pathlib import Path

import cv2


CHANNEL_ORDER_ALIASES = {
    "vi_ir": "vi_ir",
    "rgb_ir": "vi_ir",
    "visible_infrared": "vi_ir",
    "ir_vi": "ir_vi",
    "ir_rgb": "ir_vi",
    "infrared_visible": "ir_vi",
}


def normalize_channel_order(order):
    key = str(order or "vi_ir").strip().lower()
    if key not in CHANNEL_ORDER_ALIASES:
        raise ValueError(f"Unsupported channel order '{order}'. Expected one of {sorted(CHANNEL_ORDER_ALIASES)}.")
    return CHANNEL_ORDER_ALIASES[key]


def get_multimodal_settings(cfg=None):
    return {
        "input_modality": str(getattr(cfg, "input_modality", "multimodal") or "multimodal").strip().lower(),
        "channel_order": normalize_channel_order(getattr(cfg, "channel_order", "vi_ir")),
        "pair_visible_dir": str(getattr(cfg, "pair_visible_dir", "images") or "images"),
        "pair_infrared_dir": str(getattr(cfg, "pair_infrared_dir", "image") or "image"),
        "verify_pairs": bool(getattr(cfg, "verify_pairs", True)),
        "strict_pairing": bool(getattr(cfg, "strict_pairing", False)),
        "pair_check_samples": int(getattr(cfg, "pair_check_samples", 100) or 0),
    }


def resolve_paired_image_path(visible_path, settings):
    visible_path = Path(visible_path)
    visible_dir = settings["pair_visible_dir"]
    infrared_dir = settings["pair_infrared_dir"]
    parts = list(visible_path.parts)

    for idx, part in enumerate(parts):
        if part == visible_dir:
            parts[idx] = infrared_dir
            return Path(*parts)

    raise ValueError(f"Visible path '{visible_path}' does not contain directory '{visible_dir}'.")


def split_modalities(image, channel_order="vi_ir"):
    order = normalize_channel_order(channel_order)
    if image.ndim != 3 or image.shape[2] != 6:
        raise ValueError(f"Expected fused 6-channel image, got shape {getattr(image, 'shape', None)}.")

    if order == "vi_ir":
        visible = image[..., :3]
        infrared = image[..., 3:6]
    else:
        infrared = image[..., :3]
        visible = image[..., 3:6]
    return visible, infrared


def fuse_modalities(visible_image, infrared_image, channel_order="vi_ir"):
    order = normalize_channel_order(channel_order)
    if order == "vi_ir":
        return cv2.merge((*cv2.split(visible_image), *cv2.split(infrared_image)))
    return cv2.merge((*cv2.split(infrared_image), *cv2.split(visible_image)))


def collect_pairing_issues(visible_files, settings, check_shapes=True):
    issues = []
    for visible_file in visible_files:
        visible_path = Path(visible_file)
        try:
            infrared_path = resolve_paired_image_path(visible_path, settings)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        if not infrared_path.exists():
            issues.append(f"missing infrared pair: {visible_path} -> {infrared_path}")
            continue

        if check_shapes:
            visible_image = cv2.imread(str(visible_path))
            infrared_image = cv2.imread(str(infrared_path))
            if visible_image is None:
                issues.append(f"failed to read visible image: {visible_path}")
                continue
            if infrared_image is None:
                issues.append(f"failed to read infrared image: {infrared_path}")
                continue
            if visible_image.shape[:2] != infrared_image.shape[:2]:
                issues.append(
                    f"shape mismatch: {visible_path} {visible_image.shape[:2]} vs {infrared_path} {infrared_image.shape[:2]}"
                )
    return issues
