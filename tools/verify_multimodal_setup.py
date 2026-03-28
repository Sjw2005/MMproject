from pathlib import Path

import sys

from experiment_config import get_experiment, list_experiments


def print_experiment_summary(name="dmfnet"):
    from ultralytics.data.multimodal import normalize_channel_order

    exp = get_experiment(name)
    print(f"Experiment: {exp.name}")
    print(f"Model: {exp.model_path()}")
    print(f"Data: {exp.data_path()}")
    print(f"Input modality: {exp.input_modality}")
    print(f"Channel order: {normalize_channel_order(exp.channel_order)}")
    print(f"Weights: {exp.weights}")


def verify_pairs(name="dmfnet"):
    from ultralytics.data.multimodal import collect_pairing_issues, get_multimodal_settings

    exp = get_experiment(name)
    settings = get_multimodal_settings(exp)
    data_root = Path(exp.data_path()).resolve().parent / "datasets"
    visible_dir = data_root / settings["pair_visible_dir"] / "train"
    visible_files = sorted(visible_dir.glob("*.*"))
    sample_files = visible_files[: settings["pair_check_samples"]]
    issues = collect_pairing_issues(sample_files, settings, check_shapes=True)
    if issues:
        print("Found pairing issues:")
        for issue in issues[:20]:
            print(f"- {issue}")
    else:
        print(f"Pairing check passed for {len(sample_files)} sample(s)")


def demo_paths(name="dmfnet"):
    from ultralytics.data.multimodal import get_multimodal_settings, resolve_paired_image_path

    exp = get_experiment(name)
    settings = get_multimodal_settings(exp)
    data_root = Path(exp.data_path()).resolve().parent / "datasets"
    example = data_root / settings["pair_visible_dir"] / "train" / "example.jpg"
    print(f"Visible example: {example}")
    print(f"Resolved infrared: {resolve_paired_image_path(example, settings)}")


if __name__ == "__main__":
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "dmfnet"
    if exp_name in {"-l", "--list"}:
        for key, value in list_experiments().items():
            print(f"{key}: {value}")
        raise SystemExit(0)
    print_experiment_summary(exp_name)
    demo_paths(exp_name)
    verify_pairs(exp_name)
