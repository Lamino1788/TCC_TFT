from typing import Dict, Optional, Tuple
import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import empyrical as ep

from Plotter.names import DATE_FIELD, ASSET_FIELD
from Plotter.base import BaseReport, ReturnsMixin

import pandas as pd


def render_mpl_table(
    data,
    ax,
    font_size=10,
    num_font_size=12,
    header_color="#40466e",
    row_colors=["#f1f1f2", "w"],
    edge_color="#cacdcf",
    bbox=[0, 0, 1, 1],
    header_columns=0,
    use_index=True,
    **kwargs,
):
    mpl_table = ax.table(
        cellText=[
            ["" for _ in range(len(data.columns))] for _ in range(len(data))
        ],
        bbox=bbox,
        colLabels=data.columns,
        rowLabels=data.index if use_index else None,
        **kwargs,
    )
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for i in range(len(data.values)):
        for j in range(len(data.values[i])):
            val = data.values[i][j]
            cell = mpl_table[i + 1, j]
            cell.get_text().set_text(str(val))

    for k, cell in mpl_table._cells.items():
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(weight="bold", color="w")
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0] % len(row_colors)])
            cell.set_fontsize(num_font_size)

    ax.axis(False)
    ax.grid(False)


...


class ReturnMetricsTable(BaseReport, ReturnsMixin):
    """Table with return metrics. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = ReturnMetricsTable.from_returns(returns)`

    Args:
        metrics_table (pd.DataFrame): Metrics table.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (18, 0.8).
        use_index (bool, optional): Defaults to False.
    """

    def __init__(
        self,
        metrics_table: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = (18, 0.8),
        use_index: bool = False,
        font_size=10,
        num_font_size=16,
        padding: Optional[Dict[str, float]] = None,
        vertical: bool = False,
        name: str = "ReturnMetricsTable",
    ):
        self._metrics_table = metrics_table
        self._figsize = figsize
        self._use_index = use_index
        self.font_size = font_size
        self.num_font_size = num_font_size
        self._padding = padding
        self._vertical = vertical
        self._name = name

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        daily_returns = cls._get_daily_returns(returns)
        metrics = (
            pd.Series(
                data={
                    "Annual Return": ep.annual_return(daily_returns),
                    "Annual Volatility": ep.annual_volatility(daily_returns),
                    "Sharpe Ratio": ep.sharpe_ratio(daily_returns),
                    "Max Drawdown": ep.max_drawdown(daily_returns),
                    "Skewness": daily_returns.skew(),
                    "Kurtosis": daily_returns.kurt(),
                    "Sortino Ratio": ep.sortino_ratio(daily_returns),
                    "Calmar Ratio": ep.calmar_ratio(daily_returns),
                    "Omega Ratio": ep.omega_ratio(daily_returns),
                    "CVAR": ep.conditional_value_at_risk(daily_returns),
                    "Downside Risk": ep.downside_risk(daily_returns),
                    "Max Daily Return": daily_returns.max(),
                    "Min Daily Return": daily_returns.min(),
                    "CAGR": ep.cagr(daily_returns),
                }
            )
            .round(3)
            .to_frame()
            .T
        )

        col_width = 2.0
        row_height = 0.4
        figsize = (
            col_width * metrics.shape[1],
            row_height * (metrics.shape[0] + 1),
        )
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        top_padding = mpl.rcParams["figure.subplot.top"] - (
            (
                mpl.rcParams["figure.subplot.top"]
                - mpl.rcParams["figure.subplot.bottom"]
            )
            / ((metrics.shape[0] + 1))
        )
        top_padding = mpl.rcParams["figure.subplot.top"]
        padding = {
            "left": mpl.rcParams["figure.subplot.left"],
            "right": mpl.rcParams["figure.subplot.right"],
            "top": top_padding,
            "bottom": mpl.rcParams["figure.subplot.bottom"],
        }
        padding = kwargs.get("padding", padding)
        kwargs["padding"] = padding

        use_index = False
        use_index = kwargs.get("use_index", use_index)
        kwargs["use_index"] = use_index
        kwargs["font_size"] = kwargs.get("font_size", 10)

        name = kwargs.get("name", "ReturnMetricsTable")
        kwargs["name"] = name

        return cls(metrics, **kwargs)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            padding = self.get_padding()
            _, ax = plt.subplots(figsize=self._figsize, gridspec_kw=padding)
        
        if self._vertical:
            self._metrics_table = self._metrics_table.T
            self._metrics_table.columns=[self._name]

        render_mpl_table(
            self._metrics_table,
            ax=ax,
            use_index=self._use_index,
            font_size=self.font_size,
            num_font_size=self.num_font_size,
            **kwargs,
        )
        return self


class MonthlyReturnsMatrix(BaseReport, ReturnsMixin):
    """Monthly returns matrix. It is recommended to use the class
    constructor from_returns to create an instance of this class.
    `report = MonthlyReturnsMatrix.from_returns(returns)`

    Args:
        monthly_returns_matrix (pd.DataFrame): Monthly returns matrix.
        figsize (Optional[Tuple[int, int]], optional): Defaults to (12, 10).
        title (str, optional): Defaults to "Monthly Returns Matrix".
        cmap (str, optional): Defaults to "RdYlGn".
        annot (bool, optional): Defaults to True.
        fmt (str, optional): Defaults to ".2%".
        linewidths (float, optional): Defaults to 0.5.
        cbar (bool, optional): Defaults to False.
        center (float, optional): Defaults to 0.0.
        ytick_rotation (float, optional): Defaults to 0.0.
        xtick_rotation (float, optional): Defaults to 0.0.
    """

    def __init__(
        self,
        monthly_returns_matrix: pd.DataFrame,
        figsize: Optional[Tuple[int, int]] = (12, 10),
        title: str = "Monthly Returns Matrix",
        cmap: str = "RdYlGn",
        annot: bool = True,
        fmt: str = ".2%",
        linewidths: float = 0.5,
        cbar: bool = False,
        center: float = 0.0,
        ytick_rotation: float = 0.0,
        xtick_rotation: float = 0.0,
    ):
        self._monthly_returns_matrix = monthly_returns_matrix
        self._figsize = figsize

        self._title = title
        self._cmap = cmap
        self._annot = annot
        self._fmt = fmt
        self._linewidths = linewidths
        self._cbar = cbar
        self._center = center
        self._ytick_rotation = ytick_rotation
        self._xtick_rotation = xtick_rotation

    @classmethod
    def from_returns(cls, returns: pd.Series, **kwargs):
        monthly_returns = cls._get_periods_return(returns, periods="months")
        returns_df = monthly_returns.rename("return").to_frame()
        returns_df["year"] = returns_df.index.get_level_values(DATE_FIELD).year
        returns_df["month"] = returns_df.index.get_level_values(
            DATE_FIELD
        ).month
        monthly_returns_matrix = pd.pivot(
            returns_df,
            index="year",
            columns="month",
            values="return",
        )
        col_width = 0.8
        row_height = 0.5
        figsize = (
            col_width * (monthly_returns_matrix.shape[1] + 1),
            row_height * (monthly_returns_matrix.shape[0] + 1),
        )
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        return cls(monthly_returns_matrix, **kwargs)

    def plot(self, ax=None, **kwargs):
        if ax is None:
            _, ax = plt.subplots(figsize=self._figsize)

        sns.heatmap(
            data=self._monthly_returns_matrix,
            ax=ax,
            annot=self._annot,
            fmt=self._fmt,
            cmap=self._cmap,
            linewidths=self._linewidths,
            cbar=self._cbar,
            center=self._center,
        )
        ax.yaxis.set_tick_params(rotation=self._ytick_rotation)
        ax.xaxis.set_tick_params(rotation=self._xtick_rotation)
        ax.set_title(self._title)
        return self
