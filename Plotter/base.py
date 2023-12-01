from abc import ABC, abstractmethod
from typing import Dict, Tuple, Literal
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from Plotter.names import ASSET_FIELD, DATE_FIELD


class BaseReport(ABC):
    """Base class for all reports. Every report should receive all data
    needed to plot in its initialization. No calculations should be done
    in the init or plot methods.
    The plot method should only plot the data receiving possibly a
    matplotlib axes object.
    Every report should have a get_size method that returns the size of the
    report in inches (width, height).

    Args:
        figsize (Tuple[float, float]): size of the report in inches
            (width, height).
    """

    def __init__(self, figsize: Tuple[float, float]) -> None:
        self._figsize = figsize

    @abstractmethod
    def plot(self, ax: plt.Axes = None, **kwargs):
        """The main plotting method for the report. It displays the report and
        return the self object.

        Args:
            ax (plt.Axes, optional): matplotlib axes object. Defaults to None.
        """

    def get_size(self) -> Tuple[int, int]:
        """Returns the size of the report in inches (width, height)."""
        return self._figsize

    def get_padding(self) -> Dict[str, float]:
        """Returns the padding of the report in inches
        (left, right, bottom, top). By default will return the padding from
        matplotlib rcParams. But a report may want to override this return
        populating the variable self._padding."""
        if hasattr(self, "_padding") and self._padding is not None:
            return self._padding

        padding = {
            "left": mpl.rcParams["figure.subplot.left"],
            "right": mpl.rcParams["figure.subplot.right"],
            "bottom": mpl.rcParams["figure.subplot.bottom"],
            "top": mpl.rcParams["figure.subplot.top"],
        }
        return padding


class ReturnsMixin:
    """Useful methods for reports that use returns."""

    @staticmethod
    def _get_daily_returns(returns):
        return returns.groupby(
            returns.index.get_level_values(DATE_FIELD).date
        ).sum()

    @staticmethod
    def _get_daily_returns_per_asset(returns):
        returns_df = returns.rename("return").reset_index().copy()
        returns_df[DATE_FIELD] = pd.to_datetime(returns_df[DATE_FIELD].dt.date)
        daily_returns_per_asset = returns_df.groupby(
            [ASSET_FIELD, DATE_FIELD]
        )["return"].sum()
        return daily_returns_per_asset

    @staticmethod
    def _get_periods_return(
        returns: pd.Series,
        periods: Literal["months", "years"] = "months",
        to_datetime=True,
    ):
        if periods == "months":
            date_str = "%Y-%m"
        elif periods == "years":
            date_str = "%Y"
        else:
            raise ValueError(
                f"period {periods} not in ['months', 'years']. Not supported."
            )
        returns_df = returns.rename("return").reset_index().copy()
        returns_df[DATE_FIELD] = returns_df[DATE_FIELD].dt.strftime(date_str)
        if to_datetime:
            returns_df[DATE_FIELD] = pd.to_datetime(returns_df[DATE_FIELD])
        daily_returns_per_asset = returns_df.groupby(DATE_FIELD)[
            "return"
        ].sum()
        return daily_returns_per_asset


class PositionsMixin:
    """Useful methods for reports that use positions."""

    @staticmethod
    def _get_daily_weights_per_asset(weights):
        weights_df = weights.rename("position").reset_index().copy()
        weights_df[DATE_FIELD] = pd.to_datetime(weights_df[DATE_FIELD].dt.date)
        daily_weights_per_asset = weights_df.groupby(
            [ASSET_FIELD, DATE_FIELD]
        )["position"].last()
        return daily_weights_per_asset
