import yfinance as yf
import pandas as pd
import datetime as dt
import threading
from typing import List

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


def projectData():
    dp = DataProvider(dt.datetime(1990, 1, 1), dt.datetime(2023, 10, 1))

    df = dp.getTickersHistory([
        #Equities
        "ES=F", # E-Mini S&P 500
        "YM=F", # Mini Dow Jones
        "NQ=F", # Nasdaq 100
        "MME=F", #MSCI Emerging Markets Index Fut
        "RTY=F", #E-mini Russell 2000
        #Commodities
        "GC=F", #Gold
        "SI=F", #Silver
        "ZC=F", #Corn Futures
        "CL=F", #Crude Oil
        "SB=F", #Sugar
        "CT=F", #Cotton
        #Fixed Income
        "ZB=F", #U.S. Treasury Bond Futures
        "ZN=F", #10-Year Treasury Note
        "ZF=F", # 5-year T-Note
        "ZT=F", #2-year T-Note
        #Currencies
        "EUR=X",
        "JPY=X",
        "GBP=X",
        "BRL=X",
        "MXN=X",
        "CAD=X"
    ])

    df.set_index("Date", inplace=True)

    return df

if __name__ == "__main__":

    df = projectData()

    df.to_csv("futurePrices.csv")

