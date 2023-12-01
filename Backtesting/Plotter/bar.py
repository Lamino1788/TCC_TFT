from typing import Literal, Optional, Tuple
import pandas as pd
import numpy as np
import empyrical as ep
import matplotlib.pyplot as plt
import seaborn as sns


from Plotter.names import DATE_FIELD, ASSET_FIELD
from Plotter.base import BaseReport, ReturnsMixin, PositionsMixin
from Plotter.colors import *


class SharpePerAssetBarReport(BaseReport, ReturnsMixin):
    """Sharpe Ratio per Asset Bar Report. It is recommended to use the
    class constructor from_returns to create an instance of this class.
    `report = SharpePerAssetBarReport.from_returns(returns)`

    Args:
        asset_sharpes (pd.Series): Sharpe of each asset
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 4).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Sharpe Ratio per Asset".
        ylabel (str, optional): Defaults to "Sharpe Ratio".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
    """

    def __init__(
        self,
        asset_sharpes: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 4),
        width: float = 0.8,
        title: str = "Sharpe Ratio per Asset",
        ylabel: str = "Sharpe Ratio",
        xlabel: str = "asset",
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
    ):
        self._asset_sharpes = asset_sharpes
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation
        self._xlabel = xlabel

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns_per_asset = cls._get_daily_returns_per_asset(returns)
        asset_sharpes = daily_returns_per_asset.groupby(ASSET_FIELD).apply(
            lambda x: ep.sharpe_ratio(x)
        )
        fig_width = asset_sharpes.shape[0]
        figsize = (fig_width, 4)
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        return cls(asset_sharpes, **kwargs)

    def plot(self, ax=None):
        sharpes = self._asset_sharpes.sort_values(ascending=False)
        color = np.where(sharpes > 0, GREEN, RED)

        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        sharpes.plot(kind="bar", ax=ax, color=DARK_BLUE, width=self._width)
        ax.set_ylabel(self._ylabel)
        ax.set_xlabel(self._xlabel)
        ax.set_title(self._title)
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)

        return self


class ReturnsPerAsset(BaseReport, ReturnsMixin):
    """Sharpe Ratio per Asset Bar Report. It is recommended to use the
    class constructor from_returns to create an instance of this class.
    `report = SharpePerAssetBarReport.from_returns(returns)`

    Args:
        asset_sharpes (pd.Series): Sharpe of each asset
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 4).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Sharpe Ratio per Asset".
        ylabel (str, optional): Defaults to "Sharpe Ratio".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
    """

    def __init__(
        self,
        asset_sharpes: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 4),
        width: float = 0.8,
        title: str = "Sharpe Ratio per Asset",
        ylabel: str = "Sharpe Ratio",
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
    ):
        self._asset_sharpes = asset_sharpes
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns_per_asset = cls._get_daily_returns_per_asset(returns)
        asset_sharpes = daily_returns_per_asset.groupby(ASSET_FIELD).apply(
            lambda x: ep.sharpe_ratio(x)
        )
        fig_width = asset_sharpes.shape[0]
        figsize = (fig_width, 4)
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        return cls(asset_sharpes, **kwargs)

    def plot(self, ax=None):
        sharpes = self._asset_sharpes.sort_values(ascending=False)
        color = np.where(sharpes > 0, GREEN, RED)

        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        sharpes.plot(kind="bar", ax=ax, color=color, width=self._width)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)
        ax.grid(True)

        return self


class AccumulatedReturns(BaseReport, ReturnsMixin):
    """AccumulatedReturns report. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = AccumulatedReturns.from_returns(returns)`

    Args:
        cumulative_returns (pd.Series): Cumulative returns.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Accumulated Returns".
        ylabel (str, optional): Defaults to "Returns".
        add_fund (bool, optional): Defaults to False.
        fund_name (str, optional): Defaults to "Fund".

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
        add_fund: bool = False,
        fund_name: str = "Fund",
        horizontal: bool = False,
        rotate_x_ticks: bool = False,
    ):
        self._cumulative_returns = cumulative_returns
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._add_fund = add_fund
        self._fund_name = fund_name
        self._horizontal = horizontal
        self._rotate_x_ticks = rotate_x_ticks

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns = cls._get_daily_returns_per_asset(returns)

        cumulative_returns = pd.DataFrame(daily_returns)
        cumulative_returns.index.rename(["", "Date"], inplace=True)
        cumulative_returns = ep.cum_returns(
            cumulative_returns.pivot_table(
                index="Date", columns="", values="return"
            )
            .fillna(0)
        )
        # if "Offshore" in cumulative_returns.columns:
        #     cumulative_returns.iloc[-1]["BRL"] = cumulative_returns.iloc[-1]["BRL"] + abs(cumulative_returns.iloc[-1]['BRL'] + cumulative_returns.iloc[-1]['Offshore'] - 0.06887054673907267)
        cumulative_returns = pd.DataFrame(cumulative_returns.iloc[-1])
        cumulative_returns.columns = ["return"]

        return cls(cumulative_returns, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)
        self._cumulative_returns["colors"] = np.where(
            self._cumulative_returns["return"] > 0, DARK_BLUE, SALMON
        )
        if self._add_fund:
            self._cumulative_returns.loc[
                self._fund_name, "colors"
            ] = LIGHT_BLUE
            self._cumulative_returns.loc[
                self._fund_name, "return"
            ] = self._cumulative_returns["return"].sum()

        if not self._horizontal:
            (self._cumulative_returns["return"]*100).plot(
                kind="bar", ax=ax, color=self._cumulative_returns["colors"]
            )
            if self._rotate_x_ticks:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        else:
            (self._cumulative_returns["return"]*100).plot(
                kind="barh", ax=ax, color=self._cumulative_returns["colors"]
            )
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.grid(True)

        return self


class VolatilityByAsset(BaseReport, ReturnsMixin):
    """Volatility By Asset report. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = VolatilityByAsset.from_returns(returns)`

    Args:
        vol (pd.Series): Volatility of each asset's series.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Volatility By Asset".
        ylabel (str, optional): Defaults to "Volatility(%)".
        add_fund (bool, optional): Defaults to False.
        fund_name (str, optional): Defaults to "Fund".

    from_returns Args:
        returns (pd.Series): Return series.
    """

    def __init__(
        self,
        vol: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Volatility By Asset",
        ylabel: str = "Volatility(%)",
        add_fund: bool = False,
        fund_name: str = "Fund",
        rotate_x_ticks: bool = False,
    ):
        self._vol = vol
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._add_fund = add_fund
        self._fund_name = fund_name
        self._rotate_x_ticks = rotate_x_ticks

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns = cls._get_daily_returns_per_asset(returns)

        returns = pd.DataFrame(daily_returns)
        returns.index.rename(["", "Date"], inplace=True)
        returns = returns.pivot_table(
            index="Date", columns="", values="return"
        ).fillna(0)
        if kwargs.get("add_fund"):
            returns[kwargs.get("fund_name")] = returns.sum(axis=1)

        vol = pd.DataFrame(returns.std() ** 0.5).rename(columns={0: "vol"})
        return cls(vol, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)
        self._vol["colors"] = DARK_BLUE
        if self._add_fund:
            self._vol.loc[self._fund_name, "colors"] = LIGHT_BLUE
        (self._vol["vol"].mul(100)).plot(
            kind="bar", ax=ax, color=self._vol["colors"], legend=False
        )
        if self._rotate_x_ticks:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.grid(True)

        return self


class DailyReturnsDistribution(BaseReport, ReturnsMixin):
    """Daily Returns Distribution Report. It is recommended to use the
    class constructor from_returns to create an instance of this class.
    `report = DailyReturnsDistribution.from_returns(returns)`

    Args:
        daily_returns (pd.Series): Daily returns
        figsize (Tuple[int, int], optional): Defaults to (6, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Distribution of Daily Returns".
        ylabel (str, optional): Defaults to "Density".
        xlabel (str, optional): Defaults to "Daily Return".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
        xmin (Tuple[float, float], optional): Defaults to None.
        xmax (Tuple[float, float], optional): Defaults to None.
    """

    def __init__(
        self,
        daily_returns: pd.Series,
        figsize: Tuple[int, int] = (6, 5),
        width: float = 0.8,
        title: str = "Distribution of Daily Returns",
        ylabel: str = "Density",
        xlabel: str = "Daily Return",
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
        xmin: Tuple[float, float] = None,
        xmax: Tuple[float, float] = None,
    ):
        self._daily_returns = daily_returns
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._xlabel = xlabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation
        self._xmin = xmin
        self._xmax = xmax

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns = cls._get_daily_returns(returns)
        std = daily_returns.std()
        mean = daily_returns.mean()
        xmin = kwargs.get("xmin", mean - 6 * std)
        kwargs["xmin"] = xmin
        xmax = kwargs.get("xmax", mean + 6 * std)
        kwargs["xmax"] = xmax
        return cls(daily_returns, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        sns.histplot(self._daily_returns, ax=ax, stat="density", kde=True)
        ax.set_xlabel(self._xlabel)
        ax.set_title(self._title)
        ax.set_xlim(self._xmin, self._xmax)
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)

        return self


class PeriodsReturnBarReport(BaseReport, ReturnsMixin):
    """Periods Returns Bar Report. It is recommended to use the
    class constructor from_returns to create an instance of this class.
    `report = PeriodsReturnsDistribution.from_returns(returns)`

    Args:
        monthly_returns (pd.Series): Monthly returns.
        average_monthly_return (float): Average monthly return.
        figsize (Tuple[int, int], optional): Defaults to (6, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Distribution of Monthly Returns".
        xlabel (str, optional): Defaults to "Monthly Return".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
        xmin (Tuple[float, float], optional): Defaults to None.
        xmax (Tuple[float, float], optional): Defaults to None.
        stat (str, optional): Defaults to "proportion".
        kde (bool, optional): Defaults to True.
    """

    def __init__(
        self,
        periods_returns: pd.Series,
        average_period_return: float,
        orientation: Literal["horizontal", "vertical"] = "vertical",
        figsize: Tuple[int, int] = (6, 6),
        width: float = 0.8,
        title: str = "Yearly Returns",
        xlabel: str = "Yearly Return",
        ylabel: Optional[str] = None,
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
        stat: str = "proportion",
        kde: bool = True,
    ):
        self._periods_return = periods_returns
        self._average_period_return = average_period_return
        self._orientation = orientation
        self._figsize = figsize
        self._width = width
        self._title = title
        self._xlabel = xlabel
        self._ylabel = ylabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation
        self._stat = stat
        self._kde = kde

    @classmethod
    def from_returns(
        cls,
        returns: pd.Series,
        periods="years",
        orientation="vertical",
        **kwargs,
    ):
        periods_returns = cls._get_periods_return(
            returns, periods=periods, to_datetime=False
        )
        average_period_return = periods_returns.mean()

        bar_width = 0.5
        length = bar_width * (periods_returns.shape[0] + 1)
        if orientation == "vertical":
            figsize = (6, length)
        elif orientation == "horizontal":
            figsize = (length, 6)
        else:
            raise ValueError(
                f'Orientation {orientation} not in ["vertical", "horizontal"]'
            )
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        return cls(
            periods_returns,
            average_period_return,
            orientation=orientation,
            **kwargs,
        )

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        if self._orientation == "vertical":
            sns.barplot(
                y=self._periods_return.index.values,
                x=self._periods_return.values,
                orient="h",
                color=BLUE,
                ax=ax,
            )
            ax.axvline(
                self._average_period_return,
                color=RED,
                ls="--",
                label="Average",
            )

        ax.set_xlabel(self._xlabel)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.legend()
        return self


class PeriodsReturnsDistribution(BaseReport, ReturnsMixin):
    """Monthly Returns Distribution Report. It is recommended to use the
    class constructor from_returns to create an instance of this class.
    `report = PeriodsReturnsDistribution.from_returns(returns)`

    Args:
        monthly_returns (pd.Series): Monthly returns.
        average_monthly_return (float): Average monthly return.
        figsize (Tuple[int, int], optional): Defaults to (6, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Distribution of Monthly Returns".
        xlabel (str, optional): Defaults to "Monthly Return".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
        xmin (Tuple[float, float], optional): Defaults to None.
        xmax (Tuple[float, float], optional): Defaults to None.
        stat (str, optional): Defaults to "proportion".
        kde (bool, optional): Defaults to True.
    """

    def __init__(
        self,
        periods_returns: pd.Series,
        average_period_return: float,
        figsize: Tuple[int, int] = (6, 5),
        width: float = 0.8,
        title: str = "Distribution of Monthly Returns",
        xlabel: str = "Monthly Return",
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
        xmin: Tuple[float, float] = None,
        xmax: Tuple[float, float] = None,
        stat: str = "proportion",
        kde: bool = True,
    ):
        self._periods_return = periods_returns
        self._average_period_return = average_period_return
        self._figsize = figsize
        self._width = width
        self._title = title
        self._xlabel = xlabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation
        self._xmin = xmin
        self._xmax = xmax
        self._stat = stat
        self._kde = kde

    @classmethod
    def from_returns(cls, returns: pd.Series, periods="months", **kwargs):
        periods_returns = cls._get_periods_return(returns, periods=periods)
        average_period_return = periods_returns.mean()
        return cls(periods_returns, average_period_return, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        sns.histplot(
            self._periods_return,
            ax=ax,
            stat=self._stat,
            kde=self._kde,
        )
        ax.set_xlabel(self._xlabel)
        ax.set_title(self._title)
        ax.set_xlim(self._xmin, self._xmax)
        ymin, ymax = ax.get_ylim()
        ax.vlines(
            self._average_period_return,
            ymin,
            ymax,
            color=RED,
            ls="--",
            label="Average",
        )
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)
        ax.legend()
        return self


class AverageAssetsWeight(BaseReport, PositionsMixin):
    """Average Assets Weight Bar Report. It is recommended to use the
    class constructor from_weights to create an instance of this class.
    `report = AverageAssetsWeight.from_weights(weights)`

    Args:
        asset_average_weights (pd.Series): Average weight of each asset.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 3).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Average Assets Weight".
        ylabel (str, optional): Defaults to "Weight".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
    """

    def __init__(
        self,
        asset_average_weights: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 3),
        width: float = 0.8,
        title: str = "Average Assets Weight",
        ylabel: str = "Weight",
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
    ):
        self._asset_average_weights = asset_average_weights
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation

    @classmethod
    def from_weights(cls, weights: pd.Series, **kwargs):
        daily_weights_by_asset = cls._get_daily_weights_per_asset(weights)
        asset_average_weights = daily_weights_by_asset.groupby(
            ASSET_FIELD
        ).mean()
        fig_width = min(asset_average_weights.shape[0], 8)
        figsize = (fig_width, 3)
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        return cls(asset_average_weights, **kwargs)

    def plot(self, ax=None):
        weights = self._asset_average_weights.sort_values(ascending=False)
        color = np.where(weights > 0, GREEN, RED)

        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        weights.plot(kind="bar", ax=ax, color=color, width=self._width)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)

        return self


class AssetsWeightChart(BaseReport, PositionsMixin):
    """Assets Weight Chart Report. It is recommended to use the
    class constructor from_weights to create an instance of this class.
    `report = AssetsWeightChart.from_weights(weights)`

    Args:
        asset_long_average_weights (pd.Series): Assets long average
            weights.
        asset_short_average_weights (pd.Series): Assets short average
            weights.
        asset_average_weights (pd.Series): Assets average weights.
        positioned_asset_average_weights (pd.Series): Assets average
            weights when positioned.
        asset_long_max_weights (pd.Series): Max long weights.
        asset_short_max_weights (pd.Series): Max short weights.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Assets Weight Report".
        ylabel (str, optional): Defaults to "Weight".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
        plot_avg_positioned (bool, optional): Defaults to True.
        plot_avg_weights (bool, optional): Defaults to True.
        plot_max_long (bool, optional): Defaults to True.
        plot_max_short (bool, optional): Defaults to True.
        plot_avg_long (bool, optional): Defaults to True.
        plot_avg_short (bool, optional): Defaults to True.
    """

    def __init__(
        self,
        asset_long_average_weights: pd.Series,
        asset_short_average_weights: pd.Series,
        asset_average_weights: pd.Series,
        positioned_asset_average_weights: pd.Series,
        asset_long_max_weights: pd.Series,
        asset_short_max_weights: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Assets Weight Report",
        ylabel: str = "Weight",
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
        plot_avg_positioned: bool = True,
        plot_avg_weights: bool = True,
        plot_max_long: bool = True,
        plot_max_short: bool = True,
        plot_avg_long: bool = True,
        plot_avg_short: bool = True,
    ):
        self._asset_long_average_weights = asset_long_average_weights
        self._asset_short_average_weights = asset_short_average_weights
        self._asset_average_weights = asset_average_weights
        self._positioned_asset_average_weights = (
            positioned_asset_average_weights
        )
        self._asset_long_max_weights = asset_long_max_weights
        self._asset_short_max_weights = asset_short_max_weights
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation
        self._plot_avg_positioned = plot_avg_positioned
        self._plot_avg_weights = plot_avg_weights
        self._plot_max_long = plot_max_long
        self._plot_max_short = plot_max_short
        self._plot_avg_long = plot_avg_long
        self._plot_avg_short = plot_avg_short

    @classmethod
    def from_weights(cls, weights: pd.Series, **kwargs):
        daily_weights_by_asset = cls._get_daily_weights_per_asset(weights)
        long_weights = daily_weights_by_asset[daily_weights_by_asset > 0]
        short_weights = daily_weights_by_asset[daily_weights_by_asset < 0]
        positioned_weights = daily_weights_by_asset[
            daily_weights_by_asset != 0
        ]

        asset_long_average_weights = long_weights.groupby(ASSET_FIELD).mean()
        asset_short_average_weights = short_weights.groupby(ASSET_FIELD).mean()
        asset_long_max_weights = long_weights.groupby(ASSET_FIELD).max()
        asset_short_max_weights = short_weights.groupby(ASSET_FIELD).min()
        asset_average_weights = daily_weights_by_asset.groupby(
            ASSET_FIELD
        ).mean()
        positioned_asset_average_weights = positioned_weights.groupby(
            ASSET_FIELD
        ).mean()

        return cls(
            asset_long_average_weights,
            asset_short_average_weights,
            asset_average_weights,
            positioned_asset_average_weights,
            asset_long_max_weights,
            asset_short_max_weights,
            **kwargs,
        )

    def plot(self, ax=None):
        long_avg = self._asset_long_average_weights
        short_avg = self._asset_short_average_weights
        long_max = self._asset_long_max_weights
        short_max = self._asset_short_max_weights
        avg_weights = self._asset_average_weights
        positioned_weights = self._positioned_asset_average_weights
        index_order = (
            (long_avg.sub(short_avg)).sort_values(ascending=False).index
        )
        long_avg = long_avg.reindex(index_order)
        short_avg = short_avg.reindex(index_order)
        avg_weights = avg_weights.reindex(index_order)
        positioned_weights = positioned_weights.reindex(index_order)
        long_max = long_max.reindex(index_order)
        short_max = short_max.reindex(index_order)

        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        if self._plot_avg_long:
            long_avg.plot(
                kind="bar",
                ax=ax,
                color=GREEN,
                width=self._width,
                label="Avg Long",
            )
        if self._plot_avg_short:
            short_avg.plot(
                kind="bar",
                ax=ax,
                color=RED,
                width=self._width,
                label="Avg Short",
            )
        if self._plot_avg_weights:
            avg_weights.rename("weight").reset_index().plot(
                kind="scatter",
                x=ASSET_FIELD,
                y="weight",
                marker="^",
                ax=ax,
                color=BLUE,
                label="Avg Weights",
            )
        if self._plot_avg_positioned:
            positioned_weights.rename("weight").reset_index().plot(
                kind="scatter",
                x=ASSET_FIELD,
                y="weight",
                marker="^",
                ax=ax,
                color=ORANGE,
                label="Avg Weights (Positioned)",
            )
        if self._plot_max_long:
            long_max.rename("weight").reset_index().plot(
                kind="scatter",
                x=ASSET_FIELD,
                y="weight",
                marker="x",
                ax=ax,
                color=GREEN,
                label="Max Long",
            )
        if self._plot_max_short:
            short_max.rename("weight").reset_index().plot(
                kind="scatter",
                x=ASSET_FIELD,
                y="weight",
                marker="x",
                ax=ax,
                color=RED,
                label="Max Short",
            )
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)
        ax.legend()
        ax.grid(axis="y")

        return self


class AverageTurnoverPerAsset(BaseReport, PositionsMixin):
    """Average Turnover per Asset Bar Report. It is recommended to use the
    class constructor from_weights to create an instance of this class.
    `report = AverageTurnoverPerAsset.from_weights(weights)`

    Args:
        avg_turnover_per_asset (pd.Series): Average turnover of each asset.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 4).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Average Daily Turnover".
        ylabel (str, optional): Defaults to "Daily Turnover (weights)".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
    """

    def __init__(
        self,
        avg_turnover_per_asset: pd.Series,
        figsize: Optional[Tuple[int, int]] = (8, 4),
        width: float = 0.8,
        title: str = "Average Daily Turnover",
        ylabel: str = "Daily Turnover (weights)",
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
    ):
        self._avg_turnover_per_asset = avg_turnover_per_asset
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation

    @classmethod
    def from_weights(cls, weights: pd.Series, **kwargs):
        daily_weights = cls._get_daily_weights_per_asset(weights)
        daily_turnover_per_asset = (
            daily_weights.groupby(ASSET_FIELD).diff().abs()
        )
        avg_turnover_per_asset = daily_turnover_per_asset.groupby(
            ASSET_FIELD
        ).mean()

        fig_width = avg_turnover_per_asset.shape[0]
        figsize = (fig_width, 4)
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        return cls(avg_turnover_per_asset, **kwargs)

    def plot(self, ax=None):
        weights = self._avg_turnover_per_asset.sort_values(ascending=False)

        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        weights.plot(kind="bar", ax=ax, color=BLUE, width=self._width)
        ax.set_ylabel(self._ylabel)
        ax.set_title(self._title)
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)

        return self


class GenericBar(BaseReport, ReturnsMixin):
    """Generic bar report. It is recommended to use the class
    constructor from_series to create an instance of this class.
    `report = GenericBar.from_series(series)`

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
        xlabel: str = "Date",
        rotate_x_ticks: bool = True,
    ):
        self._series = series
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._xlabel = xlabel
        self._rotate_x_ticks = rotate_x_ticks

    @classmethod
    def from_series(cls, series: pd.Series, sort=True, **kwargs):
        new_series = series.copy()
        if sort:
            new_series = new_series.sort_values(ascending=False)
        return cls(new_series, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        color = np.where(self._series > 0, DARK_BLUE, SALMON)
        self._series.plot(kind="bar", ax=ax, color=color)
        ax.set_ylabel(self._ylabel)
        ax.set_xlabel(self._xlabel, visible=False)
        ax.set_title(self._title)
        if self._rotate_x_ticks:
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.grid(True)

        return self


class HorizontalSnapshotPositions(BaseReport, PositionsMixin):
    """Horizontal Snapshot Position Report. It is recommended to use the
    class constructor from_weights to create an instance of this class.
    `report = HorizontalSnapshotPosition.from_weights(weights)`

    Args:
        weights (pd.Series): weight of each asset.
        fund (str, optional): name of the fund.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 3).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Average Assets Weight".
        ylabel (str, optional): Defaults to "Weight".
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
    """

    def __init__(
        self,
        asset_weights: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = (8, 3),
        width: float = 0.8,
        title: str = "Position Snapshot",
        fund: str = "Fund",
        ylabel: str = "Country",
        xlabel: str = "Allocation",
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
    ):
        self.asset_weights = asset_weights
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._xlabel = xlabel
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation
        self._fund = fund

    @classmethod
    def from_weights(cls, weights: pd.Series, **kwargs):
        daily_weights_by_asset = pd.DataFrame(
            cls._get_daily_weights_per_asset(weights)
        )
        daily_weights_by_asset.index.rename(["", "Date"], inplace=True)
        asset_weights = daily_weights_by_asset.pivot_table(
            index="Date", columns="", values="position"
        )
        asset_weights = asset_weights.fillna(0).tail(1).T
        asset_weights.columns = ["position"]

        return cls(asset_weights, **kwargs)

    def plot(self, ax=None):
        weights = self.asset_weights.sort_values(by="position", ascending=True)
        zara_df = pd.DataFrame(
            {"position": [weights["position"].sum()]}, index=[f"{self._fund}"]
        )
        weights = pd.concat([weights, zara_df], axis=0) * 100
        weights["color"] = np.where(weights["position"] > 0, DARK_BLUE, SALMON)
        weights.loc[weights.index == self._fund, "color"] = LIGHT_BLUE

        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        weights.position.plot(
            kind="barh", ax=ax, color=weights.color, width=self._width
        )
        ax.set_ylabel(self._ylabel)
        ax.set_xlabel(self._xlabel)
        ax.set_title(self._title)
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)
        ax.grid(True)

        return self


class BarComparison(BaseReport, ReturnsMixin):
    """Bar Comparison report. It is recommended to use the class
    constructor from_series to create an instance of this class.
    `report = BarComparison.from_series(series)`

    Args:
        series (pd.Series): Series.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (8, 5).
        width (float, optional): Defaults to 0.8.
        title (str, optional): Defaults to "Series".
        ylabel (str, optional): Defaults to "Value".
    """

    def __init__(
        self,
        series: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = (8, 5),
        width: float = 0.8,
        title: str = "Series",
        ylabel: str = "Value",
        xlabel: str = "Date",
    ):
        self._series = series
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._xlabel = xlabel

    @classmethod
    def from_series(cls, series: pd.Series, **kwargs):
        new_series = pd.DataFrame(series.copy())
        new_series.columns = ["return"]
        new_series.index.rename(["Date", ""], inplace=True)
        new_series = new_series.pivot_table(
            index="Date", columns="", values="return"
        ).fillna(0)

        return cls(new_series, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        self._series.mul(100).plot.bar(ax=ax)
        ax.set_ylabel(self._ylabel)
        ax.set_xlabel(self._xlabel, visible=False)
        ax.set_title(self._title)
        ax.grid(True)

        return self

class New_Bar(BaseReport, ReturnsMixin):
    """Generic bar report. It is recommended to use the class
    constructor from_series to create an instance of this class.
    `report = GenericBar.from_series(series)`

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
        xlabel: str = "Date",
        rotate_x_ticks: bool = True,
    ):
        self._series = series
        self._figsize = figsize
        self._width = width
        self._title = title
        self._ylabel = ylabel
        self._xlabel = xlabel
        self._rotate_x_ticks = rotate_x_ticks

    @classmethod
    def from_series(cls, series: pd.Series, sort=True, **kwargs):
        new_series = series.copy()
        if sort:
            new_series = new_series.sort_values(ascending=False)
        return cls(new_series, **kwargs)

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        df = self._series.copy()
        positive = pd.Series(np.where(df>=0, df, np.nan), index=df.index)
        negative = pd.Series(np.where(df<0, df, np.nan), index=df.index)
        ax2 = ax.twinx()
        pd.Series(np.where(positive.index!='Eq Ano', positive, np.nan), index=df.index).plot(kind='bar', ax=ax, alpha=0.7, width=1, color = DARK_BLUE, edgecolor="white", linewidth=3, fontsize=20)
        pd.Series(np.where(positive.index=='Eq Ano', positive, np.nan), index=df.index).plot(kind='bar', ax=ax2, alpha=0.7, width=1, color = DARK_BLUE, edgecolor="white", linewidth=3, fontsize=20)
        pd.Series(np.where(negative.index!='Eq Ano', negative, np.nan), index=df.index).plot(kind='bar', ax=ax, alpha=0.7, width=1, color = SALMON,edgecolor="white", linewidth=3, fontsize=20)
        pd.Series(np.where(negative.index=='Eq Ano', negative, np.nan), index=df.index).plot(kind='bar', ax=ax2, alpha=0.7, width=1, color = SALMON, edgecolor="white", linewidth=3, fontsize=20)
        ax2.xaxis.set_tick_params(labelsize=18, rotation=0)
        ax2.yaxis.set_tick_params(labelsize=18)
        
        ax.set_ylabel('Exposure (%)', fontsize=25)
        ax.set_xlabel('Class', fontsize=25)
        ax.set_title(self._title, fontsize=30)

        ax.grid(False,linestyle='--', linewidth=2, alpha=0.2)
        ax.xaxis.set_tick_params(labelsize=18, rotation=0)
        ax.yaxis.set_tick_params(labelsize=18)

        return self