"""Utilities to execute model inference inside a simulated enclave."""

from __future__ import annotations

import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import Callable, Any, Dict


@dataclass
class EnclaveConfig:
    """Configuration for :class:`EnclaveRunner`."""

    env: Dict[str, str] | None = None
    enabled: bool = True


class EnclaveRunner:
    """Run callables in a separate process to emulate a TEE."""

    def __init__(self, cfg: EnclaveConfig | None = None) -> None:
        self.cfg = cfg or EnclaveConfig()

    def run(self, fn: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
        """Execute ``fn`` inside the enclave and return its result."""

        if not self.cfg.enabled:
            return fn(*args, **kwargs)

        queue: mp.Queue[Any] = mp.Queue()

        def _target() -> None:
            os.environ.update(self.cfg.env or {})
            os.environ["IN_ENCLAVE"] = "1"
            try:
                queue.put(fn(*args, **kwargs))
            except Exception as e:  # pragma: no cover - pass errors back
                queue.put(e)

        proc = mp.Process(target=_target)
        proc.start()
        proc.join()
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        return result


__all__ = ["EnclaveRunner", "EnclaveConfig"]
