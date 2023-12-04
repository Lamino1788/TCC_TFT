import pandas as pd
import numpy as np
from functools import cache
from typing import List
from DataProvider import projectData
from MODEL_DATA import ASSET_SECTOR

class Moskowitz:

    def __init__(self, prices: pd.DataFrame, vol_target: float = 0.15, vol_lookback: int = 60,
                 fi_vol_target: int = 0.05, fi_tickers : List[str] = []):
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.prices = prices
        self.daily_returns = self._rolling_returns(1)
        self.daily_vol = self._realized_vol()
        self.fi_vol_target = fi_vol_target
        self.fi_tickers = fi_tickers

    def _rolling_returns(self, days: int = 1) -> pd.DataFrame:
        return self.prices / self.prices.shift(days) - 1

    def _realized_vol(self) -> pd.DataFrame:
        daily_returns = self.daily_returns.copy()
        return daily_returns.ewm(span=self.vol_lookback,
                                 min_periods=self.vol_lookback) \
            .std().fillna(method="bfill")

    def _vol_scaled_returns(self) -> pd.DataFrame:

        daily_returns = self.daily_returns.copy()
        annual_vol = self.daily_vol.copy() * np.sqrt(252)

        scaled_returns = daily_returns / annual_vol.shift(1)

        scaled_returns[self.fi_tickers] *= self.fi_vol_target
        not_fi_tickers = set(scaled_returns.columns) - set(self.fi_tickers)
        scaled_returns[list(not_fi_tickers)] *= self.vol_target

        return scaled_returns

    def strategy_return(self) -> pd.DataFrame:
         """
            Computes the return of the Moskowitz' Time Series Momentum Strategy
            Returns:
                pd.Series : series of TSMOM returns
         """
         annual_returns = self._rolling_returns(252)
         next_day_returns = self._vol_scaled_returns().shift(-1)
         return np.sign(annual_returns) * next_day_returns

class MACDTimeScale:

    def __init__(self, short : int, long: int):
        self.S = short
        self.L = long

class MACD:

    def __init__(self, prices: pd.DataFrame,combinations : List[MACDTimeScale]):
        self.prices = prices
        self.combinations = combinations

    @staticmethod
    @cache
    def halflife(timescale):
        return np.log(0.5)/ np.log(1 - 1/timescale)

    def _rolling_returns(self, days: int = 1) -> pd.DataFrame:
        return self.prices / self.prices.shift(days) - 1

    def _signal(self, srs: pd.Series, short_timescale : int, long_timescale: int) -> pd.Series:

        prices = srs[srs.index >= srs.first_valid_index()].fillna(method="bfill")
        m_s = prices.ewm(halflife= MACD.halflife(short_timescale)).mean()
        m_l = prices.ewm(halflife= MACD.halflife(long_timescale)).mean()

        macd = m_s - m_l

        q: pd.Series = macd / prices.rolling(63).std()

        q.replace(-np.inf, 0, inplace= True)

        return q / q.rolling(252).std()



    def _position_size(self, signal : pd.Series) -> pd.Series:

        return (signal * np.exp(-(np.square(signal)) / 4)) / 0.89

    def _average_signal(self, srs: pd.Series) -> pd.Series:
        signals = [self._signal(srs, c.S, c.L) for c in self.combinations]
        return pd.concat(signals, axis=1).mean(axis=1)

    def strategy_return(self) -> pd.DataFrame:
        daily_returns = self._rolling_returns()
        tickers = self.prices.columns.tolist()
        returns = daily_returns.copy()
        for ticker in tickers:
            signals = self._average_signal(self.prices[ticker])
            positions = self._position_size(signals)
            positions.fillna(0, inplace=True)
            returns[ticker] = positions * daily_returns[ticker]
        return returns


def moskowitz_returns():

    df = projectData()
    mk = Moskowitz(df, fi_tickers= ASSET_SECTOR["Fixed Income"])
    return mk.strategy_return()

def macd_returns():

    df = projectData()
    macd = MACD(df, [MACDTimeScale(s,l) for s, l in [(8, 24), (16, 48), (32, 96)]])

    return macd.strategy_return()


if __name__ == "__main__":
    import matplotlib
    import datetime

    # matplotlib.use("TkAgg")
    mk_returns = moskowitz_returns()
    macd_returns = macd_returns()

    # c = np.sign(mk._rolling_returns(252)).value_counts("ZT=F")
    # print(c)
    # mk_returns.loc[mk_returns.index > datetime.date(2017,1,1), "ZT=F"].add(1).cumprod().plot()
    # matplotlib.pyplot.show()

    # print(mk_returns.head())
    mk_returns.to_csv("moskowitz.csv", index_label="Date")
    macd_returns.to_csv("macd.csv", index_label="Date")




