from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelResult:
    model: str
    status: str
    mae: float | None = None
    rmse: float | None = None
    smape: float | None = None
    details: dict[str, Any] | None = None


class DataTooShortError(RuntimeError):
    pass
