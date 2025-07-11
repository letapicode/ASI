from pathlib import Path
import sys
import importlib.util

# Allow imports without installing the package
_src = Path(__file__).resolve().parent.parent / "src"
if _src.exists() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
    __path__.append(str(_src))


def _import(name: str) -> None:
    path = _src / f"{name}.py"
    if not path.exists():
        return
    spec = importlib.util.spec_from_file_location(f"asi.{name}", path)
    if spec and spec.loader:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"asi.{name}"] = mod
        globals()[name] = mod
        try:
            spec.loader.exec_module(mod)
        except Exception:
            # Skip modules with missing optional dependencies

            # Optional dependencies may be missing in minimal test env

            globals().pop(name, None)
            sys.modules.pop(f"asi.{name}", None)


for _m in [
    "vector_stores",
    "quantum_retrieval",
    "quantum_sampler",
    "quantum_hpo",
    "memory_pb2",
    "memory_pb2_grpc",
    "quantum_memory_server",
    "quantum_memory_client",
    "enclave_runner",
    "zero_trust_memory_server",
]:
    try:  # pragma: no cover - optional deps
        _import(_m)
    except Exception:
        pass


def __getattr__(name: str):
    try:
        _import(name)
    except Exception:
        pass
    if name in globals():
        return globals()[name]
    raise AttributeError(name)
