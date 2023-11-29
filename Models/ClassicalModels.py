import pandas as pd
import numpy as np
from functools import cache
from typing import List
from DataProvider import projectData

class Moskowitz:

    def __init__(self, prices: pd.DataFrame, vol_target: float = 0.15, vol_lookback: int = 60):
        self.vol_target = vol_target
        self.vol_lookback = vol_lookback
        self.prices = prices
        self.daily_returns = self._rolling_returns(1)
        self.daily_vol = self._realized_vol()

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

        return daily_returns * self.vol_target / annual_vol.shift(1)

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

        m_s = srs.ewm(halflife= MACD.halflife(short_timescale)).mean()
        m_l = srs.ewm(halflife= MACD.halflife(long_timescale)).mean()

        macd = m_s - m_l

        q = macd / srs.rolling(63).std().fillna(method="bfill")

        return q / q.rolling(252).std().fillna(method="bfill")

    def _position_size(self, signal : pd.Series) -> pd.Series:

        return (signal * np.exp(-(np.square(signal)) / 4)) / 0.89

    def _average_signal(self, srs: pd.Series) -> pd.Series:
        signals = [self._signal(srs, c.S, c.L) for c in self.combinations]
        return np.sum(signals, axis=0) / len(self.combinations)

    def strategy_return(self) -> pd.DataFrame:
        daily_returns = self._rolling_returns()
        tickers = self.prices.columns.tolist()
        returns = daily_returns.copy()
        for ticker in tickers:
            signals = self._average_signal(daily_returns[ticker])
            positions = self._position_size(signals)
            returns[ticker] = positions * daily_returns[ticker].shift(1)
        return returns


def moskowitz_returns():

    df = projectData()
    mk = Moskowitz(df)
    return mk.strategy_return()

def macd_returns():

    df = projectData()
    macd = MACD(df, [MACDTimeScale(s,l) for s, l in [(8, 24), (16, 48), (32, 96)]])

    return macd.strategy_return()


if __name__ == "__main__":

    mk_returns = moskowitz_returns()
    macd_returns = macd_returns()

    mk_returns.to_csv("moskowitz.csv", index_label="Date")
    macd_returns.to_csv("macd.csv", index_label="Date")




