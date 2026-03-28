import sys
import warnings
from experiment_config import get_experiment, list_experiments

warnings.filterwarnings("ignore")


def main():
    exp_name = sys.argv[1] if len(sys.argv) > 1 else "dmfnet"
    if exp_name in {"-l", "--list"}:
        for key, value in list_experiments().items():
            print(f"{key}: {value}")
        return

    experiment = get_experiment(exp_name)
    print(f"[train] experiment={exp_name} -> {experiment.name}")
    print(f"[train] model={experiment.model_path()}")
    print(f"[train] save_dir={experiment.save_dir}")
    from ultralytics import YOLO

    model = YOLO(experiment.model_path())
    model.train(**experiment.train_args())
    print(f"Training finished: {experiment.name}")


if __name__ == "__main__":
    main()
