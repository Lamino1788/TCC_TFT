from typing import Optional, Tuple
import pandas as pd
import numpy as np
import empyrical as ep
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from Plotter.names import DATE_FIELD, ASSET_FIELD
from Plotter.base import BaseReport, ReturnsMixin, PositionsMixin


class CumulativeReturns(BaseReport, ReturnsMixin):
    """Cumulative returns report. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = CumulativeReturns.from_returns(returns)`

    Args:
        cumulative_returns (pd.Series): Cumulative returns.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Cumulative Returns".
        ylabel (str, optional): Defaults to "Return".
    """

    def __init__(
        self,
        cumulative_returns: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Cumulative Returns",
        ylabel: str = "Return",
    ):
        self._cumulative_returns = cumulative_returns
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns = cls._get_daily_returns(returns)
        cumulative_returns = daily_returns.cumsum()
        return cls(cumulative_returns, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._cumulative_returns.plot(kind="line", ax=ax, legend=False)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.grid()

        return self


class CumulativeReturnsWithBenchmarks(BaseReport, ReturnsMixin):
    """Cumulative returns with benchmarks or general comparison report. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = CumulativeReturnsWithBenchmarks.from_returns(returns)`

    Args:
        cumulative_returns (pd.Series): Cumulative returns.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Cumulative Returns".
        ylabel (str, optional): Defaults to "Return".

    from_returns Args:
        returns (pd.Series): Returns of the strategy, the benchmark and other comparisons.
    """

    def __init__(
        self,
        cumulative_returns: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Cumulative Returns",
        ylabel: str = "Returns",
    ):
        self._cumulative_returns = cumulative_returns
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns = cls._get_daily_returns_per_asset(returns)

        cumulative_returns = pd.DataFrame(daily_returns)
        cumulative_returns.index.rename(["", "Date"], inplace=True)
        cumulative_returns = (
            cumulative_returns.pivot_table(
                index="Date", columns="", values="return"
            )
            .fillna(0)
            .add(1).cumprod().sub(1)
        )

        return cls(cumulative_returns, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._cumulative_returns.mul(100).plot(
            kind="line", ax=ax, legend=False
        )
        figure = ax.get_figure()

        ax.set_ylabel(self._ylabel)
        ax.set_xlabel("", visible=False)
        ax.set_title(self._title)
        ax.grid(visible=True)
        ax.legend()

        return self


class UnderwaterPlot(BaseReport, ReturnsMixin):
    """Underwater plot report. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = UnderwaterPlot.from_returns(returns)`

    Args:
        drawdown (pd.Series): Drawdown series.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Underwater Plot".
        ylabel (str, optional): Defaults to "Drawdown".
        alpha (float, optional): Defaults to 0.7.
    """

    def __init__(
        self,
        drawdown: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Underwater Plot",
        ylabel: str = "Drawdown",
        alpha: float = 0.7,
    ):
        self._drawdown = drawdown
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._alpha = alpha

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns = cls._get_daily_returns(returns)
        cum_returns = daily_returns.add(1).cumprod()
        running_max = cum_returns.cummax()
        drawdown = (cum_returns.sub(running_max)).div(running_max).mul(100)
        return cls(drawdown, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._drawdown.plot(kind="area", ax=ax, alpha=self._alpha)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.grid(visible=True)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter())

        return self


class RollingSharpe(BaseReport, ReturnsMixin):
    """Rolling Sharpe report. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = RollingSharpe.from_returns(returns)`

    Args:
        rolling_sharpe (pd.Series): Rolling Sharpe series.
        rolling_window (int, optional): Defaults to 252.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Rolling Sharpe - {rolling_window} days".
        ylabel (str, optional): Defaults to "Sharpe Ratio".
        alpha (float, optional): Defaults to 0.7.
    """

    def __init__(
        self,
        rolling_sharpe: pd.Series,
        rolling_window: int = 252,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Rolling Sharpe - {rolling_window} days",
        ylabel: str = "Sharpe Ratio",
        alpha: float = 0.7,
    ):
        self._rolling_sharpe = rolling_sharpe
        self._figsize = figsize
        self._width = width
        self._title = title.format(rolling_window=rolling_window)
        self._ylabel = ylabel
        self._alpha = alpha

    @classmethod
    def from_returns(cls, returns: pd.Series, rolling_window=252, **kwargs):
        daily_returns = cls._get_daily_returns(returns)
        mean = daily_returns.rolling(rolling_window).mean()
        std = daily_returns.rolling(rolling_window).std()
        rolling_sharpe = mean.div(std).mul(np.sqrt(252))
        return cls(rolling_sharpe, rolling_window=rolling_window, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._rolling_sharpe.plot(kind="line", ax=ax)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.grid(visible=True)

        return self


class ExpandingSharpe(BaseReport, ReturnsMixin):
    """Expanding Sharpe report. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = ExpandingSharpe.from_returns(returns)`

    Args:
        expanding_sharpe (pd.Series): Expanding Sharpe Ratio series.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Expanding Sharpe Ratio".
        ylabel (str, optional): Defaults to "Sharpe Ratio".
        alpha (float, optional): Defaults to 0.7.
    """

    def __init__(
        self,
        expanding_sharpe: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Expanding Sharpe Ratio",
        ylabel: str = "Sharpe Ratio",
        alpha: float = 0.7,
    ):
        self._expanding_sharpe = expanding_sharpe
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._alpha = alpha

    @classmethod
    def from_returns(cls, returns: pd.Series, min_periods=150, **kwargs):
        daily_returns = cls._get_daily_returns(returns)
        mean = daily_returns.expanding(min_periods=min_periods).mean()
        std = daily_returns.expanding(min_periods=min_periods).std()
        expanding_sharpe = mean.div(std).mul(np.sqrt(252))
        return cls(expanding_sharpe, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._expanding_sharpe.plot(kind="line", ax=ax)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.grid()

        return self


class RollingVolatility(BaseReport, ReturnsMixin):
    """Rolling Volatility report. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = RollingVolatility.from_returns(returns)`

    Args:
        rolling_volatility (pd.Series): Rolling Volatility series.
        rolling_window (int, optional): Defaults to 252.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Rolling Volatility - {rolling_window} days".
        ylabel (str, optional): Defaults to "Volatility (annualized))".
        alpha (float, optional): Defaults to 0.7.
    """

    def __init__(
        self,
        rolling_volatility: pd.Series,
        rolling_window: int = 252,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Rolling Volatility - {rolling_window} days",
        ylabel: str = "Volatility (annualized)",
        alpha: float = 0.7,
    ):
        self._rolling_volatility = rolling_volatility
        self._figsize = figsize
        self._width = width
        self._title = title.format(rolling_window=rolling_window)
        self._ylabel = ylabel
        self._alpha = alpha

    @classmethod
    def from_returns(cls, returns: pd.Series, rolling_window=252, **kwargs):
        daily_returns = cls._get_daily_returns(returns)
        std = daily_returns.rolling(rolling_window).std()
        rolling_volatity = std.mul(np.sqrt(252))
        return cls(rolling_volatity, rolling_window=rolling_window, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._rolling_volatility.plot(kind="line", ax=ax)
        ax.set_ylabel(self._ylabel)
        ax.set_xlabel("Date")
        ax.set_title(self._title)
        ax.grid(visible=True)

        return self


class RollingAverageTurnover(BaseReport, PositionsMixin):
    """Rolling Average Turnover report. It is recommended to use the class
    constructor from_weights to create an instance of this class.
    `report = RollingAverageTurnover.from_weights(weights)`

    Args:
        rolling_average_turnover (pd.Series): Rolling Average Turnover series.
        rolling_window (int, optional): Defaults to 252.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Rolling Average Daily Turnover - {rolling_window} days".
        ylabel (str, optional): Defaults to "Daily Turnover (weights)".
    """

    def __init__(
        self,
        rolling_average_turnover: pd.Series,
        rolling_window: int = 252,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Rolling Average Daily Turnover - {rolling_window} days",
        ylabel: str = "Daily Turnover (weights)",
    ):
        self._rolling_average_turnover = rolling_average_turnover
        self._figsize = figsize
        self._width = width
        self._title = title.format(rolling_window=rolling_window)
        self._ylabel = ylabel

    @classmethod
    def from_weights(cls, weights: pd.DataFrame, rolling_window=252, **kwargs):
        daily_weights = cls._get_daily_weights_per_asset(weights)
        daily_turnover = (
            daily_weights.groupby(ASSET_FIELD)
            .diff()
            .abs()
            .groupby(DATE_FIELD)
            .sum()
        )
        rolling_average_turnover = daily_turnover.rolling(
            rolling_window
        ).mean()
        return cls(
            rolling_average_turnover, rolling_window=rolling_window, **kwargs
        )

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._rolling_average_turnover.plot(kind="line", ax=ax)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.grid()

        return self


class GenericLine(BaseReport, ReturnsMixin):
    """Generic line report. It is recommended to use the class
    constructor from_series to create an instance of this class.
    `report = GenericLine.from_series(series)`

    Args:
        series (pd.Series): Series.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Series".
        ylabel (str, optional): Defaults to "Value".
    """

    def __init__(
        self,
        series: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Series",
        ylabel: str = "Value",
    ):
        self._series = series
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel

    @classmethod
    def from_series(cls, series: pd.Series, **kwargs):
        new_series = pd.DataFrame(series.copy())
        new_series.index.rename(["Date", ""], inplace=True)
        new_series = new_series.pivot_table(
            index="Date", columns="", values="value"
        )
        return cls(new_series, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._series.plot(kind="line", ax=ax, legend=True)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.grid(visible=True)

        return self
