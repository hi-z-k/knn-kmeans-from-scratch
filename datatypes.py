from dataclasses import dataclass
from typing import Optional


@dataclass
class Data:
    features: list[float]
    label: Optional[str] = None