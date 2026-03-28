"""Offline bidding environment for simulating ad auctions."""
import numpy as np
from typing import Tuple, Optional


class OfflineEnv:
    """Environment for simulating online advertising bidding without live auctions.

    Attributes:
        min_remaining_budget: Minimum budget threshold below which bidding stops.
    """

    def __init__(self, min_remaining_budget: float = 0.1) -> None:
        """Initialize the offline environment.

        Args:
            min_remaining_budget: Minimum remaining budget to continue bidding.
        """
        self.min_remaining_budget = min_remaining_budget

    def simulate_ad_bidding(
        self,
        pValues: np.ndarray,
        pValueSigmas: np.ndarray,
        bids: np.ndarray,
        leastWinningCosts: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Simulate the bidding outcome for a single timestep.

        Args:
            pValues: Expected conversion values for each impression.
            pValueSigmas: Standard deviations of conversion values.
            bids: Bid amounts for each impression.
            leastWinningCosts: Minimum cost to win each impression.

        Returns:
            Tuple of (tick_value, tick_cost, tick_status, tick_conversion).
        """
        tick_status = bids >= leastWinningCosts
        tick_cost = leastWinningCosts * tick_status
        values = np.random.normal(loc=pValues, scale=pValueSigmas)
        values = values * tick_status
        tick_value = np.clip(values, 0, 1)
        tick_conversion = np.random.binomial(n=1, p=tick_value)

        return tick_value, tick_cost, tick_status, tick_conversion


def test() -> None:
    """Run a simple test of the offline environment."""
    pv_values = np.array([10, 20, 30, 40, 50])
    pv_values_sigma = np.array([1, 2, 3, 4, 5])
    bids = np.array([15, 20, 35, 45, 55])
    market_prices = np.array([12, 22, 32, 42, 52])

    env = OfflineEnv()
    tick_value, tick_cost, tick_status, tick_conversion = env.simulate_ad_bidding(
        pv_values, pv_values_sigma, bids, market_prices)

    print(f"Tick Value: {tick_value}")
    print(f"Tick Cost: {tick_cost}")
    print(f"Tick Status: {tick_status}")


if __name__ == '__main__':
    test()
