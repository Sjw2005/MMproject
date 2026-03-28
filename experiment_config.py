from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    model: str
    data: str = "data.yaml"
    input_modality: str = "multimodal"
    channel_order: str = "vi_ir"
    pair_visible_dir: str = "images"
    pair_infrared_dir: str = "image"
    verify_pairs: bool = True
    strict_pairing: bool = False
    pair_check_samples: int = 100
    imgsz: int = 640
    epochs: int = 150
    batch: int = 16
    workers: int = 8
    device: str = "0"
    cache: str = "ram"
    optimizer: str = "SGD"
    amp: bool = False
    project: str = "results"
    warmup_epochs: float = 5.0
    mosaic: float = 1.0
    close_mosaic: int = 30
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    lr0: float = 0.01
    scale: float = 0.2
    max_det: int = 500
    conf: float = 0.001
    hsv_h: float = 0.01
    hsv_s: float = 0.01
    hsv_v: float = 0.01
    split: str = "test"
    notes: str = ""

    @property
    def save_dir(self):
        return ROOT / self.project / self.name

    @property
    def weights(self):
        return self.save_dir / "weights" / "best.pt"

    def model_path(self):
        return str((ROOT / self.model).resolve()) if not Path(self.model).is_absolute() else self.model

    def data_path(self):
        return str((ROOT / self.data).resolve()) if not Path(self.data).is_absolute() else self.data

    def to_overrides(self):
        payload = asdict(self)
        for key in ("name", "model", "data", "notes"):
            payload.pop(key, None)
        payload["project"] = str((ROOT / self.project).resolve()) if not Path(self.project).is_absolute() else self.project
        return payload

    def train_args(self):
        args = self.to_overrides()
        args.update({"data": self.data_path()})
        return args

    def val_args(self):
        args = self.to_overrides()
        args.update({"data": self.data_path(), "split": self.split, "batch": max(self.batch, 1) * 2})
        return args

    def predict_args(self):
        args = self.to_overrides()
        args.update({"source": str((ROOT / "datasets" / self.pair_visible_dir / "val").resolve())})
        return args


EXPERIMENTS = {
    "dmfnet": ExperimentConfig(
        name="DMFNet",
        model="improve_multimodal/DMFNet.yaml",
        notes="Current multimodal baseline."
    ),
    "dmfnet_imgsz960": ExperimentConfig(
        name="DMFNet_imgsz960",
        model="improve_multimodal/DMFNet.yaml",
        imgsz=960,
        batch=8,
        notes="High-priority direct improvement by increasing resolution for small targets."
    ),
    "dmfnet_imgsz1280": ExperimentConfig(
        name="DMFNet_imgsz1280",
        model="improve_multimodal/DMFNet.yaml",
        imgsz=1280,
        batch=4,
        notes="More aggressive small-target improvement experiment."
    ),
    "dmfnet_adamw": ExperimentConfig(
        name="DMFNet_adamw",
        model="improve_multimodal/DMFNet.yaml",
        optimizer="AdamW",
        lr0=0.001,
        notes="Optimizer ablation for the current DMFNet."
    ),
}


def get_experiment(name="dmfnet"):
    key = str(name).strip().lower()
    if key not in EXPERIMENTS:
        valid = ", ".join(sorted(EXPERIMENTS))
        raise KeyError(f"Unknown experiment '{name}'. Valid options: {valid}")
    return EXPERIMENTS[key]


def list_experiments():
    return {key: value.name for key, value in EXPERIMENTS.items()}
