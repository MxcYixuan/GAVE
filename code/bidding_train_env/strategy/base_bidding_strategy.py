"""Base bidding strategy interface."""
from abc import ABC, abstractmethod
import numpy as np
from typing import List, Tuple, Any


class BaseBiddingStrategy(ABC):
    """Abstract base class for bidding strategies.

    Attributes:
        budget: Total budget for the bidding campaign.
        remaining_budget: Remaining budget at current timestep.
        name: Strategy name.
        cpa: Cost per acquisition constraint.
        category: Advertiser category.
    """

    def __init__(
        self,
        budget: float = 100,
        name: str = "BaseStrategy",
        cpa: float = 2,
        category: int = 1
    ) -> None:
        """Initialize the bidding strategy.

        Args:
            budget: Total budget for bidding.
            name: Strategy name.
            cpa: Cost per acquisition constraint.
            category: Advertiser category index.
        """
        self.budget = budget
        self.remaining_budget = budget
        self.name = name
        self.cpa = cpa
        self.category = category

    @abstractmethod
    def reset(self) -> None:
        """Reset the strategy state for a new episode."""
        pass

    @abstractmethod
    def bidding(
        self,
        timeStepIndex: int,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        historyPValueInfo: List[np.ndarray],
        historyBid: List[np.ndarray],
        historyAuctionResult: List[np.ndarray],
        historyImpressionResult: List[np.ndarray],
        historyLeastWinningCost: List[np.ndarray]
    ) -> np.ndarray:
        """Calculate bids for the current timestep.

        Args:
            timeStepIndex: Current timestep index.
            pValues: Expected values for each impression.
            pValueSigmas: Standard deviations of values.
            historyPValueInfo: Historical value information.
            historyBid: Historical bid amounts.
            historyAuctionResult: Historical auction outcomes.
            historyImpressionResult: Historical impression results.
            historyLeastWinningCost: Historical winning costs.

        Returns:
            Array of bid amounts for each impression.
        """
        pass


__all__ = ["BaseBiddingStrategy"]