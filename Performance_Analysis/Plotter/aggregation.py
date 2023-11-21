from typing import List, Tuple, Type
import matplotlib as mpl
import matplotlib.pyplot as plt

from Plotter.base import BaseReport


class RowFigureConcatenationReport(BaseReport):
    """Aggregates reports vertically. It is recommended to use the class
    constructor from_report_list to create an instance of this class.
    `report = RowFigureConcatenationReport.from_report_list(report_list)`

    Args:
        report_list (List[Type[BaseReport]]): List of reports.
        figsize (Tuple[float, float]): Figsize of the report.
        heights (List[float]): Height of each report.
        widths (List[float], optional): Width of each report.
            Defaults to None.
        center_figures (bool, optional): Center figures if they are of
            different sizes. Defaults to True.
    """

    def __init__(
        self,
        report_list: List[Type[BaseReport]],
        figsize: Tuple[float, float],
        heights: List[float],
        widths: List[float] = None,
        center_figures: bool = True,
        padding: dict = {
            "left": mpl.rcParams["figure.subplot.left"],
            "right": mpl.rcParams["figure.subplot.right"],
            "top": mpl.rcParams["figure.subplot.top"],
            "bottom": mpl.rcParams["figure.subplot.bottom"],
        }
    ):
        self._report_list = report_list
        self._figsize = figsize
        self._heights = heights
        self._center_figures = center_figures
        self._widths = widths
        self._padding = padding

    @classmethod
    def from_report_list(cls, report_list: List[Type[BaseReport]], **kwargs):
        heights = [report.get_size()[1] for report in report_list]
        widths = [report.get_size()[0] for report in report_list]

        figsize = (max(widths), sum(heights))
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        heights = kwargs.get("heights", heights)
        kwargs["heights"] = heights

        widths = kwargs.get("widths", widths)
        kwargs["widths"] = widths

        padding = {
            "left": mpl.rcParams["figure.subplot.left"],
            "right": mpl.rcParams["figure.subplot.right"],
            "top": mpl.rcParams["figure.subplot.top"],
            "bottom": mpl.rcParams["figure.subplot.bottom"],
        }
        padding = kwargs.get("padding", padding)
        kwargs["padding"] = padding

        return cls(
            report_list,
            **kwargs,
        )

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self.get_size(), gridspec_kw=self._padding)

        fig = ax.get_figure()

        subfigs = fig.subfigures(
            nrows=len(self._report_list),
            ncols=1,
            height_ratios=self._heights,
        )

        for idx, report in enumerate(self._report_list):
            if self._center_figures:
                report_padding = report.get_padding()
                add_abs_pad = self.get_size()[0] - self._widths[idx]
                left_abs_pad = report_padding["left"] * self._widths[
                    idx
                ] + add_abs_pad / 2
                left = left_abs_pad / self.get_size()[0]
                right_abs_pad = (1 - report_padding["right"]) * self._widths[
                    idx
                ] + add_abs_pad / 2
                right = 1 - (right_abs_pad / self.get_size()[0])
                sub_ax = subfigs[idx].subplots(
                    gridspec_kw={
                        "left": left,
                        "right": right,
                        "top": report_padding["top"],
                        "bottom": report_padding["bottom"],
                    }
                )
            else:
                sub_ax = subfigs[idx].subplots()
            report.plot(ax=sub_ax)
        ax.grid(False)
        ax.axis(False)

        return self

class ColumnFigureConcatenationReport(BaseReport):
    """Aggregates reports horizontally. It is recommended to use the class
    constructor from_report_list to create an instance of this class.
    `report = ColumnFigureConcatenationReport.from_report_list(report_list)`

    Args:
        report_list (List[Type[BaseReport]]): List of reports.
        figsize (Tuple[float, float]): Figsize of the report.
        widths (List[float]): Width of each report.
        heights (List[float], optional): Height of each report.
            Defaults to None.
        center_figures (bool, optional): Center figures if they are of
            different sizes. Defaults to True.
    """

    def __init__(
        self,
        report_list: List[Type[BaseReport]],
        figsize: Tuple[float, float],
        widths: List[float],
        heights: List[float] = None,
        center_figures: bool = True,
        padding: float = 0.0,
    ):
        self._report_list = report_list
        self._figsize = figsize
        self._heights = heights
        self._center_figures = center_figures
        self._widths = widths
        self._padding2 = padding

    @classmethod
    def from_report_list(cls, report_list: List[Type[BaseReport]], **kwargs):
        heights = [report.get_size()[1] for report in report_list]
        widths = [report.get_size()[0] for report in report_list]

        figsize = (sum(widths), max(heights))
        figsize = kwargs.get("figsize", figsize)
        kwargs["figsize"] = figsize

        heights = kwargs.get("heights", heights)
        kwargs["heights"] = heights

        widths = kwargs.get("widths", widths)
        kwargs["widths"] = widths

        padding = kwargs.get("padding", 0.0) 
        kwargs["padding"] = padding 

        return cls(
            report_list,
            **kwargs,
        )

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self.get_size())

        fig = ax.get_figure()

        subfigs = fig.subfigures(
            nrows=1,
            ncols=len(self._report_list),
            width_ratios=self._widths,
        )

        for idx, report in enumerate(self._report_list):
            if self._center_figures:
                pad = self.get_size()[1] - self._heights[idx]
                top_pad = (
                    1 - mpl.rcParams["figure.subplot.top"]
                ) * self._heights[idx] + pad / 2
                top = 1 - (top_pad / self.get_size()[1])
                bottom_pad = (
                    mpl.rcParams["figure.subplot.bottom"] * self._heights[idx]
                    + pad / 2)
                bottom = bottom_pad / self.get_size()[1]

                sub_ax = subfigs[idx].subplots(
                    gridspec_kw={
                        "bottom": bottom,
                        "top": top,
                        "left": self._padding2,
                        "right": 1 - self._padding2,
                    }
                )
            else:
                sub_ax = subfigs[idx].subplots()
            report.plot(ax=sub_ax)
        
        ax.grid(False)
        ax.axis(False)
        return self
