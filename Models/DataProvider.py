import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt
import threading
from typing import List
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import operator
import functools

from MODEL_DATA import ASSET_SECTOR

class DataProvider:

    def __init__(self, start: dt.datetime, end: dt.datetime):
        self.start = start
        self.end = end
        self.tickers = []

    def _getHistory(self, ticker: str) -> None:
        hist = yf.Ticker(ticker).history(start = self.start, end = self.end)
        hist.index = list(map(lambda x : dt.date(x.year, x.month, x.day), hist.index))
        hist.index.name = "Date"
        hist["ticker"] = ticker
        self.tickers.append(hist)


    def getTickersHistory(self, tickers: List[str]) -> pd.DataFrame:
        self.historic = {}
        threads = []
        for ticker in tickers:
            thread = threading.Thread(target= self._getHistory, args= (ticker,))
            threads.append(thread)
            thread.start()
        for thread in threads:
            thread.join()
        df = pd.concat(self.tickers)
        df = df.pivot( columns="ticker", values="Close")
        df = df.reset_index(drop = False).rename_axis("Date")
        return df

def cleanData(df: pd.DataFrame) -> pd.DataFrame:

    tickers = df.columns.tolist()
    for t in tickers:
        first_index = df[t].first_valid_index()
        df.loc[first_index:, t] = df.loc[first_index:, t].fillna(method="bfill")
        mask = abs(df[t]/df[t].shift(1) -1 ) > 0.5
        df[mask] = np.NAN
        df.loc[first_index:, t] = df.loc[first_index:, t].fillna(method="bfill")
    return df

def projectData():
    dp = DataProvider(dt.datetime(2000, 1, 1), dt.datetime(2023, 11, 13))

    all_tickers = functools.reduce(operator.iconcat, ASSET_SECTOR.values(), [])
    df = dp.getTickersHistory(all_tickers)

    df.set_index("Date", inplace=True)

    df = cleanData(df)
    return df

if __name__ == "__main__":

    matplotlib.use("TkAgg")
    sns.set_style("darkgrid")
    df = projectData()

    df.reset_index(inplace=True, names= "Date")
    df_melt = pd.melt(df, id_vars=["Date"], var_name=["Ticker"], value_name="Close")

    df_melt.dropna(inplace=True)

    f, axes = plt.subplots(2,2)
    f.tight_layout()
    i = 0
    for sector in ASSET_SECTOR:
        subdf = df_melt[df_melt["Ticker"].isin(ASSET_SECTOR[sector])]
        row = i//2
        col = i %2
        sns.lineplot(subdf, x = "Date", y = "Close", hue="Ticker", ax= axes[row,col]).set_title(sector)
        i+=1

    plt.show()




