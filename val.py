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
    print(f"[val] experiment={exp_name} -> {experiment.name}")
    print(f"[val] weights={experiment.weights}")
    from ultralytics import YOLO

    model = YOLO(str(experiment.weights))
    model.val(**experiment.val_args())


if __name__ == "__main__":
    main()
