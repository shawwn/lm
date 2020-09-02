from typing import Optional

from pydantic import BaseModel


class Seq2SeqFormat(BaseModel):
    vocab_size: int
    context_length: int
    has_eos: Optional[bool] = False
    keys = ["content", "target"]
