from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Callable, ContextManager, Generator, Optional

from .config import LOGGER


@contextmanager
def log_elapsed_time(name: str, enabled: bool = True) -> Generator[None]:
    if not enabled:
        yield
        return

    started = perf_counter()
    LOGGER.info("Starting %s", name)
    try:
        yield
    finally:
        LOGGER.info("Finished %s in %.2fs", name, perf_counter() - started)


def timer_Factory(enabled: bool) -> Callable[[str], ContextManager[None]]:
    return lambda name: log_elapsed_time(name, enabled)


def suffix_path(path: Optional[Path], suffix: str) -> Optional[Path]:
    if path is None:
        return None
    return path.with_name(f"{path.stem}_{suffix}{path.suffix}")


def validation_audit_summary_path(path: Optional[Path]) -> Optional[Path]:
    return suffix_path(path, "multi_split_summary")
