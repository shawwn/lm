"Base infeed class"
from typing import Dict, Optional

from pydantic.dataclasses import dataclass



@dataclass
class InfeedConfig:
    batch_size: int
    dataset: Dict
    max_sequence_length: int
    file_pattern: Optional[str]


class Infeed:
    batch_size: int
    max_sequence_length: int
    dataset: object
