import torch
import numpy as np
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from .base import BaseReward
from adpo.reward_functions import compute_score
from .trajectory_stitching import TrajectoryStitcher
from ..phase_splitters.base import PhaseSplitter

logger = logging.getLogger(__name__)


class StitchReward(BaseReward):
    def __init__(self, stitcher: TrajectoryStitcher, splitter: PhaseSplitter, config):
        super().__init__()
        self.config = config
        self.algo = config.algorithm
        self.stitcher = stitcher
        self.splitter = splitter
    
    def compute(self, boundaries_batch, response_mask, index,
                outcome_rewards: List[float], phase_texts_batch: List[List[str]], 
                questions: List[str], golden_answers: List[str], 
                data_sources: List[str], entropy: torch.Tensor, log_probs: torch.Tensor,
                 **context
                ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        pass
    
        
        
        