import torch
import numpy as np
import logging
from typing import List, Optional, Tuple
from abc import ABC, abstractmethod

from .base import BaseReward
from adpo.reward_functions import compute_score

logger = logging.getLogger(__name__)

class EntropyReward(BaseReward):
    def __init__(self, config):
        super(self, EntropyReward).__init__()
        algo = config.algorithm
        self.psi = getattr(algo, 'psi', 0.95*0.99)
        self.threshold_pct = getattr(algo, 'default_threshold_percentile', 90.0)
        self.correct_total = getattr(algo,'correct_total', 1.0)
        self.incorrect_total = getattr()