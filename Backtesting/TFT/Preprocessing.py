import numpy as np
import pandas as pd
from typing import List
from Models.DataProvider import projectData
from Models.MODEL_DATA import ASSET_SECTOR


def add_macd(df, short_period=12, long_period=26, signal_period=9): #Moving Average Convergence Divergence
    exp1 = df['Close'].ewm(span=short_period, adjust=False).mean()
    exp2 = df['Close'].ewm(span=long_period, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    df['MACD'] = macd
    df['MACD_Signal'] = signal
    return df

def add_rsi(df, period=30): # Relative Strength Index
    delta = df['Close'].diff()
    delta = delta[1:]
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0

    for period in [7, 14, 22, 30, 60]:
      average_gain = up.rolling(window=period).mean()
      average_loss = abs(down.rolling(window=period).mean())
      rs = average_gain / average_loss
      rsi = 100.0 - (100.0 / (1.0 + rs))

      df[f'RSI_{period}'] = rsi
    return df

def encode_cyclical(df, col, max_val):
    df[col] = np.cos(2 * np.pi * df[col]/max_val)
    return df


def make_Complete_Data(prices: pd.DataFrame, main_ticker: str, other_tickers: List[str],
                       max_encoder_length: int = 126, max_prediction_length:int = 10):

    rename_dict = {t: f"Close_{t}" for t in other_tickers}
    rename_dict[main_ticker] = "Close"
    Complete_Data = prices.rename(columns=rename_dict)

    Complete_Data[f"Close_{main_ticker}"] = Complete_Data["Close"]

    Complete_Data = np.log(Complete_Data / Complete_Data.shift(1))
    Complete_Data['Close_prediction'] = Complete_Data['Close'].shift(max_prediction_length).fillna(0)
    Complete_Data['Close_encoder'] = Complete_Data['Close'].shift(max_encoder_length).fillna(0)
    Complete_Data = Complete_Data.fillna(0).drop(list(Complete_Data.index)[0])

    Complete_Data.reset_index(inplace=True)
    Complete_Data.rename(columns={'date': 'Date'},inplace=True)

    for i in Complete_Data.index:
        Complete_Data.loc[i,'Rolling_Avg_22d'] = Complete_Data.loc[i-30:i-1, 'Close'].mean()
        Complete_Data.loc[i,'Rolling_Vol_22d'] = Complete_Data.loc[i-30:i-1, 'Close'].std()

    Complete_Data['Exponential_Avg'] = Complete_Data.Close.ewm(span=max_encoder_length).mean()
    Complete_Data = add_macd(Complete_Data)
    Complete_Data = add_rsi(Complete_Data)

    Complete_Data.fillna(0)
    Complete_Data.loc[Complete_Data.Rolling_Avg_22d == 0, 'Rolling_Avg_22d'] = Complete_Data.loc[Complete_Data.Rolling_Avg_22d == 0, 'Close']

    unknown_reals = list(Complete_Data.columns)[1:]

    Complete_Data['Ticker'] = main_ticker

    Complete_Data.index = Complete_Data.index.astype(int)
    Complete_Data['Time_Fix'] = Complete_Data.index

    Complete_Data['Year'] = Complete_Data.Date.apply(lambda x: x.year)
    Complete_Data['Month'] = Complete_Data.Date.apply(lambda x: x.month)
    Complete_Data['Day'] = Complete_Data.Date.apply(lambda x: x.day)

    Complete_Data = encode_cyclical(Complete_Data, 'Month', 12)
    Complete_Data = encode_cyclical(Complete_Data, 'Day', 31)

    return Complete_Data, unknown_reals