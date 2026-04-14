import torch
import re
import math
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod



@dataclass
class PhaseSegment:
    """A contiguous segment (phase) of a response."""
    phase_id: int
    start_idx: int      # inclusive
    end_idx: int         # exclusive
    text: str = ""
    reward: float = 0.0
    advantage: float = 0.0

class PhaseSplitter(ABC):
    def __init__(self):
        super(self, PhaseSplitter).__init__()
    @abstractmethod
    def split(self, entropy, response_mask, log_probs=None, token_ids=None, tokenizer=None) -> List[List[int]]:
        """
        Return boundaries of phases in the response.
        """
        pass

    def extract_phase_texts(
        self, 
        boundaries: List[int],
        response_length: int,
        token_ids: Optional[torch.Tensor] = None,
        tokenizer=None,
    ) -> List[PhaseSegment]:
        """Convert boundary indices into PhaseSegment objects."""
        phases = []
        for k in range(len(boundaries)):
            start = boundaries[k]
            end = boundaries[k + 1] if k + 1 < len(boundaries) else response_length

            text = ""
            if token_ids is not None and tokenizer is not None:
                text = tokenizer.decode(token_ids[start:end], skip_special_tokens=True)

            phases.append(PhaseSegment(
                phase_id=k, start_idx=start, end_idx=end, text=text,
            ))
        return phases


