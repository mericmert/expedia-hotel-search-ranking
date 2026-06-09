from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Mapping, Optional


def write_json(
    path: Optional[Path], payload: Mapping[str, object], raise_on_failure: bool = False
) -> None:
    """
    Write payload to a JSON file at the given path.
    """

    if path is None:
        if raise_on_failure:
            raise ValueError("Path cannot be None.")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, sort_keys=True)
        file.write("\n")


def read_json(path: Path) -> object:
    """
    Read a JSON file.
    """
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_pickle(path: Path, obj: object) -> None:
    """
    Save an object to a file using pickle serialization.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fp:
        pickle.dump(obj, fp, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> object:
    """
    Load an object from a pickle file.
    """
    with path.open("rb") as fp:
        return pickle.load(fp)
