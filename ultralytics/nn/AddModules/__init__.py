from importlib import import_module
from pathlib import Path

__all__ = []


def _register_module(module_name):
    try:
        module = import_module(f"{__name__}.{module_name}")
    except Exception:
        return

    for name, value in vars(module).items():
        if name.startswith("_"):
            continue
        if getattr(value, "__module__", None) != module.__name__:
            continue
        globals()[name] = value
        __all__.append(name)


for _module_path in sorted(Path(__file__).resolve().parent.glob("*.py")):
    if _module_path.name != "__init__.py":
        _register_module(_module_path.stem)


__all__ = tuple(dict.fromkeys(__all__))
