import torch
import numpy as np
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod


class BaseReward(ABC):
    def __init__(self):
        super(self, BaseReward).__init__()
    @abstractmethod
    def compute(boundaries_batch: List[List[int]], response_mask: torch.Tensor, index: torch.Tensor, **context ) \
        -> Tuple[torch.Tensor,  torch.Tensor, dict ]:
        pass

