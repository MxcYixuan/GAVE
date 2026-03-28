"""GAVE - Generative Auto-Bidding with Value-Guided Explorations

A Decision Transformer variant for auto-bidding in online advertising auctions.
Accepted by SIGIR'25.

Main package for the GAVE project.
"""

__version__ = "0.1.0"
__author__ = "GAVE Team"

from bidding_train_env.baseline.dt.dt import GAVE
from bidding_train_env.environment.offline_env import OfflineEnv
from bidding_train_env.strategy.dt_bidding_strategy import DtBiddingStrategy

__all__ = ["GAVE", "OfflineEnv", "DtBiddingStrategy", "getScore", "getScore_batch"]