from typing import Tuple
import matplotlib.pyplot as plt

from Plotter.base import BaseReport
from Plotter.colors import GREY



class TitleReport(BaseReport):
    """Title report.

    Args:
        title (str): Title to be displayed.
        figsize (Tuple[float, float], optional): Defaults to (8, 0.4).
        fontsize (int, optional): Defaults to 16.
        fontweight (str, optional): Defaults to "bold".
        color (str, optional): Defaults to "black".
        background_color (str, optional): Defaults to "white".
        fontfamily (str, optional): Defaults to "sans-serif".
    """

    def __init__(
        self,
        title: str,
        figsize: Tuple[float, float] = (8, 0.4),
        fontsize: int = 16,
        fontweight: str = "bold",
        color: str = "black",
        background_color: str = "white",
        fontfamily: str = "sans-serif",
    ):
        self._title = title
        self._figsize = figsize
        self._fontsize = fontsize
        self._fontweight = fontweight
        self._color = color
        self._background_color = background_color
        self._fontfamily = fontfamily

    def plot(self, ax=None):
        if ax is None:
            _, ax = plt.subplots(figsize=self.get_size())

        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            self._title,
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=self._fontsize,
            fontweight=self._fontweight,
            color=self._color,
            backgroundcolor=self._background_color,
            fontfamily=self._fontfamily,
        )

        return self
