import yfinance as yf
import datetime
import pandas as pd
import numpy as np
import bnlearn
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('fivethirtyeight')



import h2o
from h2o.automl import *

from sklearn.linear_model import SGDClassifier

from sklearn import metrics # for the evaluation
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

import datetime as dt

import warnings
warnings.filterwarnings("ignore")

######## GLOBAL VARS ########

_DATA = "50y"
_WINDOW_SIZE = 252 * 1
_WINDOW_SIZE_TRADE = 15
_MAX_DATE = dt.date(2023,10,27)
_MIN_DATE = dt.date(2000,1,1)
now = dt.datetime.now() # current date and time
_DATE_TIME = now.strftime("%Y%m%d_%Hh")
_PATH_OUTPUT = './bayes_project/' + _DATE_TIME + '/'

#python program to check if a directory exists
import os
# Check whether the specified path exists or not
isExist = os.path.exists(_PATH_OUTPUT)
if not isExist:
   # Create a new directory because it does not exist
   os.makedirs(_PATH_OUTPUT)
   print("The new directory is created!")

def f_get_data_yahoo(_ASSET_TARGET):

  spot_obj_tmp = yf.Ticker(_ASSET_TARGET)
  df_hist_tmp = spot_obj_tmp.history(period=_DATA)


  print(df_hist_tmp.head())
  df_hist_tmp.index = list(map(lambda x : dt.date(x.year, x.month, x.day), df_hist_tmp.index))

  df_hist_tmp.index.name = "Date"
  min_date = max(df_hist_tmp.first_valid_index(), _MIN_DATE)
  df_hist_tmp = df_hist_tmp[df_hist_tmp.index<_MAX_DATE]
  df_hist_tmp = df_hist_tmp[df_hist_tmp.index>min_date]
  return spot_obj_tmp, df_hist_tmp

def f_plot_charts(df_in):
    plt.figure(figsize = (12,6))
    plt.plot(df_in['signal_strenght_acc'], label="Signal Strength")
    plt.xlabel("Date")
    plt.ylabel("Signal Strength")
    plt.legend()
    # plt.show()

    plt.figure(figsize = (12,6))
    plt.plot(df_in['Close'], label="Price")
    plt.plot(df_in['close_ewma'], label="Price EWMA")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    # plt.show()

    plt.figure(figsize = (12,6))
    plt.plot(df_in['Close'], label="Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    # plt.show()

    plt.figure(figsize = (12,6))
    plt.plot(df_in['signal_sign'], 'k-',label="Signal",  linewidth=2)
    plt.xlabel("Date")
    plt.ylabel("signal")
    plt.legend()
    # plt.show()

    plt.figure(figsize = (12,6))
    plt.scatter(x=df_in.index, y=df_in['signal_sign'], cmap='gray')
    plt.xlabel("Date")
    plt.ylabel("signal")
    plt.legend()
    # plt.show()

def f_run_model(spot_obj, _WINDOW_SIZE, df_in):
    df_hist = df_in.copy()
    df_hist['var_abs'] = np.abs(df_hist['Close'] - df_hist['Close'].shift(1))
    df_hist['var'] = df_hist['Close'] - df_hist['Close'].shift(1)
    df_hist['close_ewma'] = df_hist['Close'].ewm(halflife = _WINDOW_SIZE, adjust=False).mean()
    df_hist['vol_ewma'] = df_hist['var_abs'].ewm(halflife = _WINDOW_SIZE, adjust=False).mean()
    df_hist['signal'] = (df_hist['Close'].shift(1) - df_hist['close_ewma'].shift(1)) / (df_hist['vol_ewma'].shift(1))
    df_hist['signal_sign'] = np.sign(df_hist['signal'])
    df_hist['signal_strenght'] = df_hist['signal_sign'] * (df_hist['Close'].shift(-1) - df_hist['Close']) / df_hist['vol_ewma'].shift(1)
    df_hist['signal_strenght_acc'] = df_hist['signal_strenght'].cumsum()
    sharpe_model = (df_hist['signal_strenght'].mean() / df_hist['signal_strenght'].std()) * (252 ** 0.5)
    sharpe_asset = (df_hist['var'].mean() / df_hist['var'].std()) * (252 ** 0.5)
    sharpe_model = round(sharpe_model, 3)
    sharpe_asset = round(sharpe_asset, 3)
    df_out = df_hist.copy()
    return [sharpe_asset, sharpe_model], df_out

def f_run_model_mod(spot_obj, _WINDOW_SIZE, df_in, df_signal_in):
    df_hist = df_in.copy()
    df_hist['var_abs'] = np.abs(df_hist['Close'] - df_hist['Close'].shift(1))
    df_hist['var'] = df_hist['Close'] - df_hist['Close'].shift(1)
    df_hist['close_ewma'] = df_hist['Close'].ewm(halflife = _WINDOW_SIZE, adjust=False).mean()
    df_hist['vol_ewma'] = df_hist['var_abs'].ewm(halflife = _WINDOW_SIZE, adjust=False).mean()
    # df_hist['signal'] = (df_hist['Close'].shift(1) - df_hist['close_ewma'].shift(1)) / (df_hist['vol_ewma'].shift(1))


    ##################################################
    df_hist = df_hist.merge(df_signal_in, how='left', on='Date')
    df_hist.fillna(method='ffill', inplace = True)
    df_hist.fillna(method='bfill', inplace = True)
    print(df_hist.head())
    ##################################################
    df_hist['signal_sign'] = np.sign(df_hist['signal'])
    df_hist['signal_strenght'] = df_hist['signal_sign'] * (df_hist['Close'].shift(-1) - df_hist['Close']) / df_hist['vol_ewma'].shift(1)
    df_hist['signal_strenght_acc'] = df_hist['signal_strenght'].cumsum()
    sharpe_model = (df_hist['signal_strenght'].mean() / df_hist['signal_strenght'].std()) * (252 ** 0.5)
    sharpe_asset = (df_hist['var'].mean() / df_hist['var'].std()) * (252 ** 0.5)
    sharpe_model = round(sharpe_model, 3)
    sharpe_asset = round(sharpe_asset, 3)
    df_out = df_hist.copy()
    return [sharpe_asset, sharpe_model], df_out

def f_get_optmized(spot_obj,df_hist):

  # lst_agg = []
  # for i in range(1, 10):
  #   _out, df_ = f_run_model(spot_obj, 252 * i, df_hist)
  #   _out_str = ' '.join(str(e) for e in _out)
  #   print(str(i) + ' ' + _out_str)
  #   lst_tmp = [i, _out[0], _out[1]]
  #   lst_agg.append(lst_tmp)
  # out_np = np.array(lst_agg)
  # max_sharpe = max(out_np[:,2])
  # best_years = out_np[out_np[:,2] == max_sharpe,0]
  # _best = best_years[0]
  ###################################################################################################
  _best = 1
  ###################################################################################################
  _out_best, df_best = f_run_model(spot_obj, 252 * _best, df_hist)
  return _out_best, df_best, _best

def f_get_data_all_tickers(_ASSET_TARGET):

  lst_tickers = [_ASSET_TARGET, "^VIX", "^TNX"]
  df_agg_out = pd.DataFrame()
  min_date = None
  for ticker_tmp in lst_tickers:
    spot_obj_lst = yf.Ticker(ticker_tmp)
    df_hist_lst = spot_obj_lst.history(period=_DATA)

  ###############################
    df_hist_lst.index = list(map(lambda x : dt.date(x.year, x.month, x.day), df_hist_lst.index))
    df_hist_lst.index.name = "Date"
  ###############################

    if ticker_tmp == _ASSET_TARGET:
      min_date = max(df_hist_lst.first_valid_index(), _MIN_DATE)
    df_hist_lst = df_hist_lst[df_hist_lst.index<_MAX_DATE]
    df_hist_lst = df_hist_lst[df_hist_lst.index>min_date]
    df_tmp = df_hist_lst[['Close']]
    df_tmp['ticker'] = ticker_tmp
    df_agg_out = df_agg_out.append(df_tmp)
  print(df_agg_out.head())
  df_agg_out.reset_index(drop=False, inplace = True)
  print(df_agg_out.head())

  df_agg_pivot_out = df_agg_out.pivot(index="Date", columns="ticker", values="Close")
  #df_agg_pivot.reset_index(drop=False, inplace = True)
  df_agg_pivot_out = df_agg_pivot_out.reset_index(drop=False).rename_axis(None, axis=1)
  print(df_agg_pivot_out.head())
  df_agg_pivot_out.fillna(method='ffill', inplace = True)
  df_agg_pivot_out.fillna(method='bfill', inplace = True)
  df_agg_pivot_out.set_index('Date',drop=True, inplace = True)
  print(df_agg_pivot_out.head())

  df_agg_pivot_out['variance'] = (df_agg_pivot_out[_ASSET_TARGET] / df_agg_pivot_out[_ASSET_TARGET].shift(1) - 1) ** 2

  df_agg_pivot_out['vol_realized'] = (df_agg_pivot_out['variance'].rolling(_WINDOW_SIZE_TRADE).mean() * 2520000) ** 0.5
  print(df_agg_pivot_out.head())
  df_agg_pivot_out.drop(columns = ['variance'], inplace = True)
  df_agg_pivot_out.fillna(method='ffill', inplace = True)
  df_agg_pivot_out.fillna(method='bfill', inplace = True)
  print(df_agg_pivot_out.head())
  return df_agg_pivot_out

def f_get_discrete_data(df_in, df_signal_best_in, _ASSET_TARGET):
  df_tmp = df_in.copy()
  for ticker_tmp in list(df_tmp.columns):
      df_tmp[ticker_tmp] = df_tmp[ticker_tmp] - df_tmp[ticker_tmp].shift(1)
      df_tmp.loc[df_tmp[ticker_tmp]>=0, ticker_tmp] = 1
      df_tmp.loc[df_tmp[ticker_tmp]<0, ticker_tmp] = 0

  df_tmp = df_tmp.merge(df_signal_best_in, how='left', on='Date')
  df_tmp.dropna(how='all', inplace = True)

  df_tmp[_ASSET_TARGET + '_shift'] = df_tmp[_ASSET_TARGET].shift(-1)
  print(df_tmp)
  #df_agg_pivot = df_agg_pivot[:-2]
  df_tmp.fillna(method='ffill', inplace = True)
  df_tmp.fillna(method='bfill', inplace = True)
  return df_tmp


def custom_ts_multi_data_prep(dataset, target, start, end, window, horizon):
    X = []
    y = []
    start = start + window
    if end is None:
        end = len(dataset) - horizon
    for i in range(start, end):
        indices = range(i - window, i)
        X.append(dataset[indices])
        indicey = range(i + 1, i + 1 + horizon)
        y.append(target[indicey])
    return np.array(X), np.array(y)

def f_run_rolling_prediction(df_data_in, _ASSET_TARGET):
  # Define the network structure


  edges = [(_ASSET_TARGET, _ASSET_TARGET + '_shift'),
          (_ASSET_TARGET, '^VIX'),
          ('signal_sign', '^VIX'),
          ('signal_sign', '^TNX'),
          ('vol_realized', '^TNX'),
          ('^VIX', _ASSET_TARGET + '_shift'),
          ('signal_sign', _ASSET_TARGET + '_shift'),
          ('^TNX', _ASSET_TARGET + '_shift')]

  # Make the actual Bayesian DAG
  DAG = bnlearn.make_DAG(edges)
  # Print the CPDs
  bnlearn.print_CPD(DAG)
  # [BNLEARN.print_CPD] No CPDs to print. Use bnlearn.plot(DAG) to make a plot.
  # Plot the DAG
  # bnlearn.plot(DAG)


  df_mod_tmp = df_data_in.copy()
  df_mod_tmp.reset_index(drop = False, inplace=True)
  df_mod_tmp['trigger'] = 0.00


  # DAG = bnlearn.parameter_learning.fit(DAG, df_agg_pivot)

  len_df = df_data_in.shape[0]

  # for i in range(100, df_agg_pivot.shape[0]):
  for i in tqdm(range(100, df_data_in.shape[0])):

    try:
        # Parameter learning on the user-defined DAG and input data
        DAG = bnlearn.parameter_learning.fit(DAG, df_data_in[:i-1], verbose = 0)

        vix_val = df_data_in.iloc[i]['^VIX']
        tsy_val = df_data_in.iloc[i]['^TNX']
        spx_val = df_data_in.iloc[i][_ASSET_TARGET]
        vol_realized_val = df_data_in.iloc[i]['vol_realized']
        signal_sign_val = df_data_in.iloc[i]['signal_sign']

        date_val = df_mod_tmp.iloc[i]['Date']

        q1 = bnlearn.inference.fit(DAG, variables=[_ASSET_TARGET + '_shift'],
                                   evidence={'^VIX':vix_val, '^TNX':tsy_val, _ASSET_TARGET:spx_val, 'vol_realized':vol_realized_val,
                                             'signal_sign':signal_sign_val}, verbose = 0)
        ret = q1.values[1]
    except Exception as e:
        print(e)
        ret = 0

    df_mod_tmp.loc[i,'trigger'] = ret
    # print(df_mod.tail())
    # # print(q1.values)
    # if i%100 ==0:
    #   print(str(i) + ' / ' + str(len_df))
  df_mod_tmp.set_index('Date',drop=True, inplace = True)
  return df_mod_tmp

def f_run_rolling_prediction_lstm(df_data_in, _ASSET_TARGET):

  df_mod_tmp = df_data_in.copy()
  df_mod_tmp.reset_index(drop = False, inplace=True)
  df_mod_tmp['trigger'] = 0.00

  len_df = df_data_in.shape[0]

  data = df_data_in.copy()

  for i in data.select_dtypes('object').columns:
    le = LabelEncoder().fit(data[i])
    data[i] = le.transform(data[i])

  X_scaler = MinMaxScaler()
  Y_scaler = MinMaxScaler()
  X_data = X_scaler.fit_transform(data[[_ASSET_TARGET,'^TNX','vol_realized','signal_sign']])
  Y_data = Y_scaler.fit_transform(data[[_ASSET_TARGET + '_shift']])

  hist_window = 90
  horizon = 1
  TRAIN_SPLIT = int(2 * len_df / 3)
  TRAIN_SPLIT = 100
  print(TRAIN_SPLIT)

  x_train, y_train = custom_ts_multi_data_prep(X_data, Y_data, 0, TRAIN_SPLIT, hist_window, horizon)
  x_vali, y_vali = custom_ts_multi_data_prep(X_data, Y_data, TRAIN_SPLIT, None, hist_window, horizon)

  x_all, y_all = custom_ts_multi_data_prep(X_data, Y_data, 0, None, hist_window, horizon)

  model = Sequential()
  model.add(LSTM(10, activation='relu', return_sequences = True, input_shape = x_train.shape[-2:]))
  model.add(LSTM(10, activation='relu'))
  model.add(Dense(horizon))
  model.compile(optimizer='adam', loss='mse')
  # fit model
  model.fit(x_train, y_train, epochs = 1, verbose = 1)

  yhat_all = model.predict(x_all, verbose = 1)

  # print('x_all.shape')
  # print(x_all.shape)

  # print('yhat_all.shape')
  # print(yhat_all.shape)

  for i in tqdm(range(100, df_data_in.shape[0])):

    X_ = df_data_in[:i-1][[_ASSET_TARGET,'^TNX','vol_realized','signal_sign']]
    Y_ = df_data_in[:i-1][[_ASSET_TARGET + '_shift']]

    X_all = df_data_in[:i][[_ASSET_TARGET,'^TNX','vol_realized','signal_sign']]
    Y_all = df_data_in[:i][[_ASSET_TARGET + '_shift']]

    X_data_tmp_ = X_scaler.fit_transform(X_[[_ASSET_TARGET,'^TNX','vol_realized','signal_sign']])
    Y_data_tmp_ = Y_scaler.fit_transform(Y_[[_ASSET_TARGET + '_shift']])

    X_data_tmp = X_scaler.fit_transform(X_all[[_ASSET_TARGET,'^TNX','vol_realized','signal_sign']])
    Y_data_tmp = Y_scaler.fit_transform(Y_all[[_ASSET_TARGET + '_shift']])

    x_vali, y_vali = custom_ts_multi_data_prep(X_data_tmp, Y_data_tmp, 0, None, hist_window, horizon)
    x_train_, y_train_ = custom_ts_multi_data_prep(X_data_tmp_, Y_data_tmp_, 0, None, hist_window, horizon)
    yhat = model.predict(x_vali, verbose = 0)

    # ret = yhat[-1][0]
    ret = yhat_all[i - hist_window - 1][0]

    df_mod_tmp.loc[i,'trigger'] = ret
    date_val = df_mod_tmp.iloc[i]['Date']

    if i%360 ==0:
      print(str(i) + ' / ' + str(len_df))
      model.fit(x_train_[-100:], y_train_[-100:], epochs = 1, verbose = 1)
  df_mod_tmp.set_index('Date', drop=True, inplace = True)
  return df_mod_tmp

def f_agg_output(df_agg_prices, df_mod, _ASSET_TARGET):
  df_agg_merged = df_agg_prices.merge(df_mod, how='left', on='Date')
  df_agg_merged['trigger_discr'] = df_agg_merged['trigger']
  df_agg_merged.loc[df_agg_merged['trigger']>0.01,'trigger_discr'] = -1
  df_agg_merged.loc[df_agg_merged['trigger']>=0.50,'trigger_discr'] = 1
  df_agg_merged.loc[df_agg_merged['trigger']==0.00,'trigger_discr'] = 0
  df_agg_merged['signal'] = df_agg_merged['trigger_discr']
  df_agg_merged[_ASSET_TARGET + '_var'] = df_agg_merged[_ASSET_TARGET + '_x'] - df_agg_merged[_ASSET_TARGET + '_x'].shift(1)
  df_agg_merged[_ASSET_TARGET + '_pnl_model'] = df_agg_merged[_ASSET_TARGET + '_var'] * df_agg_merged['trigger_discr'].shift(1)
  df_agg_merged[_ASSET_TARGET + '_pnl_model_acc'] = df_agg_merged[_ASSET_TARGET + '_pnl_model'].cumsum()
  df_agg_merged[_ASSET_TARGET + '_pnl_asset_acc'] = df_agg_merged[_ASSET_TARGET + '_var'].cumsum()
  return df_agg_merged

def f_compute_sharpe(df_agg_merged, _ASSET_TARGET):
  sharpe_model = (df_agg_merged[_ASSET_TARGET + '_pnl_model'].mean() / df_agg_merged[_ASSET_TARGET + '_pnl_model'].std()) * (252 ** 0.5)
  sharpe_asset = (df_agg_merged[_ASSET_TARGET + '_var'].mean() / df_agg_merged[_ASSET_TARGET + '_var'].std()) * (252 ** 0.5)
  return sharpe_model, sharpe_asset


def f_plot_comparison(df_best, df_best_mod, df_best_bh, _ASSET_TARGET, title_in = ''):
  file_name_out = _PATH_OUTPUT + _ASSET_TARGET + '.png'
  plt.figure(figsize = (12,6))
  plt.plot(df_best['signal_strenght_acc'], label="Trend Following")
  plt.plot(df_best_mod['signal_strenght_acc'], label="Proposed Model")
  plt.plot(df_best_bh['signal_strenght_acc'], label="Buy and Hold")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.title(title_in)
  plt.legend()
  plt.savefig(file_name_out)
  # plt.show()

def f_save_outputs(df_in, _ASSET_TARGET):
  file_name_out = _PATH_OUTPUT + _ASSET_TARGET + '.csv'
  df_in.to_csv(file_name_out, sep =';')

def f_read_outputs(_ASSET_TARGET):
  file_name_out = _PATH_OUTPUT + _ASSET_TARGET + '.csv'
  df_out = pd.read_csv(file_name_out, sep =';')
  return df_out

def f_plot_signal_tail(df_best, df_best_mod, _ASSET_TARGET):
  file_name_out = _PATH_OUTPUT + _ASSET_TARGET
  df_best_tail = df_best.tail(504)
  df_best_mod_tail = df_best_mod.tail(504)
  plt.figure(figsize = (12,6))
  fig, (ax1, ax2) = plt.subplots(2)
  fig.suptitle('Signal Plots')
  ax1.plot(df_best_tail.index,df_best_tail['signal_sign'], 'k-',label="Signal Trend Following",  linewidth=2)
  ax2.plot(df_best_mod_tail.index,df_best_mod_tail['signal_sign'], 'k-',label="Signal Proposed Model",  linewidth=2)

  plt.xlabel("Date")
  plt.ylabel("Signal")
  plt.legend()
  plt.savefig(file_name_out + 'plot_1.png')
  plt.close()
  # plt.show()

  fig = plt.figure(figsize = (12,6))
  gs = fig.add_gridspec(2, hspace=0.2)
  axs = gs.subplots(sharex=True, sharey=True)
  #fig.suptitle('Signal Plots')
  axs[0].plot(df_best_tail.index,df_best_tail['signal_sign'], 'k-', label="Signal Trend Following",  linewidth=2)
  axs[1].plot(df_best_mod_tail.index,df_best_mod_tail['signal_sign'], 'k-', label="Signal Proposed Model",  linewidth=2)

  axs[0].set_title("Signal")
  axs[1].set_title("Signal_mod")

  # Hide x labels and tick labels for all but bottom plot.
  for ax in axs:
      ax.label_outer()
  plt.savefig(file_name_out + 'plot_2.png')
  plt.close()
  # plt.show()

  t = np.array(df_best_tail.index)
  s = np.array(df_best_tail['signal_sign'])
  t_mod = np.array(df_best_mod_tail.index)
  s_mod = np.array(df_best_mod_tail['signal_sign'])
  spx = np.array(df_best_mod_tail['Close'])

  upper = 0.5
  lower = -0.5

  supper = np.ma.masked_where(s < upper, s)
  slower = np.ma.masked_where(s > lower, s)
  smiddle = np.ma.masked_where((s < lower) | (s > upper), s)

  supper_mod = np.ma.masked_where(s_mod < upper, s_mod)
  slower_mod = np.ma.masked_where(s_mod > lower, s_mod)
  smiddle_mod = np.ma.masked_where((s_mod < lower) | (s_mod > upper), s_mod)

  #fig, ax = plt.subplots()
  #ax.plot(t, smiddle, t, slower,'r-', t, supper,'g-')#
  #plt.show()


  fig = plt.figure(figsize = (12,6))
  gs = fig.add_gridspec(3, hspace=0.2)
  axs = gs.subplots(sharex=True)
  #fig.suptitle('Signal Plots')
  axs[0].plot(t, smiddle, t, slower,'rx', t, supper,'g.',label="Signal Trend Following",  linewidth=5)
  axs[1].plot(t, smiddle_mod, t, slower_mod,'rx', t, supper_mod,'g.',label="Signal Proposed Model",  linewidth=5)
  axs[2].plot(t, spx, 'k-', label="SPX500",  linewidth=2)

  axs[0].set_title("Signal Trend Following")
  axs[1].set_title("Signal Proposed Model")
  axs[2].set_title("Financial Asset")

  # Hide x labels and tick labels for all but bottom plot.
  for ax in axs:
      ax.label_outer()
  plt.savefig(file_name_out + 'plot_2.png')
  # plt.show()
  plt.close()

  t = np.array(df_best_tail.index)
  s = np.array(df_best_tail['signal_sign'])
  t_mod = np.array(df_best_mod_tail.index)
  s_mod = np.array(df_best_mod_tail['signal_sign'])
  spx = np.array(df_best_mod_tail['Close'])

  upper = 0.5
  lower = -0.5

  supper = np.ma.masked_where(s < upper, s)
  slower = np.ma.masked_where(s > lower, s)
  smiddle = np.ma.masked_where((s < lower) | (s > upper), s)

  supper_mod = np.ma.masked_where(s_mod < upper, s_mod)
  slower_mod = np.ma.masked_where(s_mod > lower, s_mod)
  smiddle_mod = np.ma.masked_where((s_mod < lower) | (s_mod > upper), s_mod)

  #fig, ax = plt.subplots()
  #ax.plot(t, smiddle, t, slower,'r-', t, supper,'g-')#
  #plt.show()


  fig = plt.figure(figsize = (12,6))
  gs = fig.add_gridspec(2, hspace=0.2)
  axs = gs.subplots(sharex=True)
  #fig.suptitle('Signal Plots')
  axs[0].plot(t, smiddle, t, slower,'rx', t, supper,'g.',label="Signal",  linewidth=5)
  axs[1].plot(t, spx, 'k-', label="SPX500",  linewidth=2)

  axs[0].set_title("Signal Trend Following")
  axs[1].set_title("Financial Asset")

  # Hide x labels and tick labels for all but bottom plot.
  for ax in axs:
      ax.label_outer()
  plt.savefig(file_name_out + 'plot_3.png')
  # plt.show()
  plt.close()

  t = np.array(df_best_tail.index)
  s = np.array(df_best_tail['signal_sign'])
  t_mod = np.array(df_best_mod_tail.index)
  s_mod = np.array(df_best_mod_tail['signal_sign'])
  spx = np.array(df_best_mod_tail['Close'])

  upper = 0.5
  lower = -0.5

  supper = np.ma.masked_where(s < upper, s)
  slower = np.ma.masked_where(s > lower, s)
  smiddle = np.ma.masked_where((s < lower) | (s > upper), s)

  supper_mod = np.ma.masked_where(s_mod < upper, s_mod)
  slower_mod = np.ma.masked_where(s_mod > lower, s_mod)
  smiddle_mod = np.ma.masked_where((s_mod < lower) | (s_mod > upper), s_mod)

  #fig, ax = plt.subplots()
  #ax.plot(t, smiddle, t, slower,'r-', t, supper,'g-')#
  #plt.show()


  fig = plt.figure(figsize = (12,6))
  gs = fig.add_gridspec(2, hspace=0.2)
  axs = gs.subplots(sharex=True)
  #fig.suptitle('Signal Plots')

  axs[0].plot(t, smiddle_mod, t, slower_mod,'rx', t, supper_mod,'g.',label="Signal_mod",  linewidth=5)
  axs[1].plot(t, spx, 'k-', label="SPX500",  linewidth=2)

  axs[0].set_title("Signal Proposed Model")
  axs[1].set_title("Financial Asset")

  # Hide x labels and tick labels for all but bottom plot.
  for ax in axs:
      ax.label_outer()
  plt.savefig(file_name_out + 'plot_4.png')
  # plt.show()
  plt.close()

def f_plot_assets(df_agg_prices_tmp, _ASSET_TARGET):
  df_agg_prices_tmp = df_agg_prices_tmp.tail(5000)
  t_val = np.array(df_agg_prices_tmp.index)
  spx_val = np.array(df_agg_prices_tmp[_ASSET_TARGET])
  t10y_val = np.array(df_agg_prices_tmp['^TNX'])
  vix_val = np.array(df_agg_prices_tmp['^VIX'])
  vol_real_val = np.array(df_agg_prices_tmp['vol_realized'])
  fig = plt.figure(figsize = (12,6))
  gs = fig.add_gridspec(4, hspace=0.2)
  axs = gs.subplots(sharex=True)
  axs[0].plot(t_val, spx_val, 'k-', label=_ASSET_TARGET,  linewidth=2)
  axs[1].plot(t_val, t10y_val, 'k-', label="T10y",  linewidth=2)
  axs[2].plot(t_val, vix_val, 'k-', label="VIX",  linewidth=2)
  axs[3].plot(t_val, vol_real_val, 'k-', label="Vol",  linewidth=2)

  axs[0].set_title(_ASSET_TARGET)
  axs[1].set_title("T10y")
  axs[2].set_title("VIX")
  axs[3].set_title("Vol")

  # Hide x labels and tick labels for all but bottom plot.
  for ax in axs:
      ax.label_outer()
  # plt.show()

def f_plot_eq_curves(df_best, df_best_mod, df_best_mod_k):
  plt.figure(figsize = (12,6))
  plt.plot(df_best['signal_strenght_acc'], label="Trading")
  plt.plot(df_best_mod['signal_strenght_acc'], label="Trading_mod")
  plt.plot(df_best_mod_k['signal_strenght_acc'], label="Trading_k")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  # plt.show()

  plt.figure(figsize = (12,6))
  plt.plot(df_best['signal_strenght_acc'], label="Trading")
  plt.plot(df_best_mod['signal_strenght_acc'], label="Trading_mod")
  plt.xlabel("Date")
  plt.ylabel("Price")
  plt.legend()
  # plt.show()

def f_compute_metrics_models(df_best, df_best_mod, df_best_mod_k):
  _df_best = df_best.copy()
  _df_best_mod = df_best_mod.copy()
  _df_best_mod_k = df_best_mod_k.copy()
  _df_spx = df_best.copy()

  _df_best['ret'] = _df_best['signal_strenght'] / _df_best['Close']
  _df_best_mod['ret'] = _df_best_mod['signal_strenght'] / _df_best_mod['Close']
  _df_best_mod_k['ret'] = _df_best_mod_k['signal_strenght'] / _df_best_mod_k['Close']
  _df_spx['ret'] = _df_spx['var'] / _df_spx['Close']

  sharpe_best = (_df_best['signal_strenght'].mean() / _df_best['signal_strenght'].std()) * (252 ** 0.5)
  sharpe_best_mod = (_df_best_mod['signal_strenght'].mean() / _df_best_mod['signal_strenght'].std()) * (252 ** 0.5)
  sharpe_best_mod_k = (_df_best_mod_k['signal_strenght'].mean() / _df_best_mod_k['signal_strenght'].std()) * (252 ** 0.5)
  sharpe_spx = (_df_spx['ret'].mean() / _df_spx['ret'].std()) * (252 ** 0.5)


  vol_spx = (_df_spx['ret'].std() ) * (252 ** 0.5)

  ret_best = vol_spx * sharpe_best
  ret_best_mod = vol_spx * sharpe_best_mod
  ret_best_mod_k = vol_spx * sharpe_best_mod_k
  ret_spx = (_df_spx['ret'].mean() ) * (252 )


  len(_df_spx[_df_spx['var']>=0])/(len(_df_spx[_df_spx['var']>=0])+len(_df_spx[_df_spx['var']<0]))
  len(_df_best[_df_best['signal_strenght']>=0])/(len(_df_best[_df_best['signal_strenght']>=0])+len(_df_best[_df_best['signal_strenght']<0]))
  len(_df_best_mod[_df_best_mod['signal_strenght']>=0])/(len(_df_best_mod[_df_best_mod['signal_strenght']>=0])+len(_df_best_mod[_df_best_mod['signal_strenght']<0]))

  _df_best["signal_sign_shift"] = _df_best["signal_sign"].shift(1)
  _df_best["signal_trades"] = 0
  _df_best.loc[ _df_best["signal_sign_shift"]!= _df_best["signal_sign"] ,"signal_trades"] = 1

  _df_best_mod["signal_sign_shift"] = _df_best_mod["signal_sign"].shift(1)
  _df_best_mod["signal_trades"] = 0
  _df_best_mod.loc[ _df_best_mod["signal_sign_shift"]!= _df_best_mod["signal_sign"] ,"signal_trades"] = 1

  _n_trades_model = _df_best["signal_trades"].sum()
  _n_da_model = _df_best["signal_trades"].sum()
  _df_best_mod["signal_trades"].sum()
  _df_best["signal_trades"].count()

  _df_best["signal_sign_shift"] = _df_best["signal_sign"].shift(1)
  _df_best["signal_trades"] = 0
  _df_best.loc[ _df_best["signal_sign_shift"]!= _df_best["signal_sign"] ,"signal_trades"] = 1

  _df_best_mod["signal_sign_shift"] = _df_best_mod["signal_sign"].shift(1)
  _df_best_mod["signal_trades"] = 0
  _df_best_mod.loc[ _df_best_mod["signal_sign_shift"]!= _df_best_mod["signal_sign"] ,"signal_trades"] = 1

  _n_trades_model = _df_best["signal_trades"].sum()
  _n_days_model = _df_best["signal_trades"].count()

  _n_trades_model_mod = _df_best_mod["signal_trades"].sum()
  _n_days_model_mod = _df_best_mod["signal_trades"].count()

  _n_trades_per_year_model = _n_trades_model / _n_days_model * 365
  _n_trades_per_year_model_mod = _n_trades_model_mod / _n_days_model_mod * 365

  _df_spx = _df_best["Close"]
  _df_spx = _df_spx[_df_spx.index<_MAX_DATE]
  _df_spx = _df_spx[_df_spx.index>_MIN_DATE]

  _df_best_conf = _df_best[["var","signal_strenght"]]
  _df_best_mod_conf = _df_best_mod[["var","signal_strenght"]]

  _pos_pos = _df_best_conf[(_df_best_conf["var"] >= 0 ) & (_df_best_conf["signal_strenght"] >= 0 ) ].count()
  _pos_neg = _df_best_conf[(_df_best_conf["var"] >= 0 ) & (_df_best_conf["signal_strenght"] < 0 ) ].count()
  _neg_pos = _df_best_conf[(_df_best_conf["var"] < 0 ) & (_df_best_conf["signal_strenght"] >= 0 ) ].count()
  _neg_neg = _df_best_conf[(_df_best_conf["var"] < 0 ) & (_df_best_conf["signal_strenght"] < 0 ) ].count()
  _total = _df_best_conf.count()

  print(_pos_pos / _total)
  print(_pos_neg / _total)
  print(_neg_pos / _total)
  print(_neg_neg / _total)

  _pos_pos = _df_best_mod_conf[(_df_best_mod_conf["var"] >= 0 ) & (_df_best_mod_conf["signal_strenght"] >= 0 ) ].count()
  _pos_neg = _df_best_mod_conf[(_df_best_mod_conf["var"] >= 0 ) & (_df_best_mod_conf["signal_strenght"] < 0 ) ].count()
  _neg_pos = _df_best_mod_conf[(_df_best_mod_conf["var"] < 0 ) & (_df_best_mod_conf["signal_strenght"] >= 0 ) ].count()
  _neg_neg = _df_best_mod_conf[(_df_best_mod_conf["var"] < 0 ) & (_df_best_mod_conf["signal_strenght"] < 0 ) ].count()
  _total = _df_best_mod_conf.count()

  print(_pos_pos / _total)
  print(_pos_neg / _total)
  print(_neg_pos / _total)
  print(_neg_neg / _total)
  _pos_pos = _df_best_mod_conf[(_df_best_mod_conf["var"] >= 0 ) & (_df_best_mod_conf["signal_strenght"] >= 0 ) ].count()
  _pos_neg = _df_best_mod_conf[(_df_best_mod_conf["var"] >= 0 ) & (_df_best_mod_conf["signal_strenght"] < 0 ) ].count()
  _neg_pos = _df_best_mod_conf[(_df_best_mod_conf["var"] < 0 ) & (_df_best_mod_conf["signal_strenght"] >= 0 ) ].count()
  _neg_neg = _df_best_mod_conf[(_df_best_mod_conf["var"] < 0 ) & (_df_best_mod_conf["signal_strenght"] < 0 ) ].count()
  _total = _df_best_mod_conf.count()

  print(_pos_pos / _total)
  print(_pos_neg / _total)
  print(_neg_pos / _total)
  print(_neg_neg / _total)

  return sharpe_best, sharpe_best_mod, sharpe_best_mod_k, sharpe_spx

def f_run_prediction_generic(method_in = 'bayes', asset_ticker_in = '^GSPC'):



  spot_obj_in, df_hist_in = f_get_data_yahoo(asset_ticker_in)
  df_agg_pivot = f_get_data_all_tickers(asset_ticker_in)
  df_agg_prices_in = df_agg_pivot.copy()

  _out_best, df_best, _best_in = f_get_optmized(spot_obj_in, df_hist_in)
  df_signal_best = df_best[['signal_sign']]
  # f_plot_charts(df_best)

  df_data_model_in = f_get_discrete_data(df_agg_pivot, df_signal_best, asset_ticker_in)
  print('df_data_model')
  print(df_data_model_in.head(10))
  print(df_agg_pivot.tail())


  if method_in == 'bayes':
    df_prediction = f_run_rolling_prediction(df_data_model_in, asset_ticker_in)
  elif method_in == 'lstm':
    df_prediction = f_run_rolling_prediction_lstm(df_data_model_in, asset_ticker_in)
  else:
    df_prediction = f_run_rolling_prediction(df_data_model_in, asset_ticker_in)


  df_agg_pred_merged = f_agg_output(df_agg_prices_in, df_prediction, asset_ticker_in)
  df_signal_pred_model = df_agg_pred_merged[['signal']]
  df_signal_bh = df_agg_pred_merged[['signal']].copy()
  df_signal_bh['signal'] = 1
  _out_best, df_best = f_run_model(spot_obj_in, 252 * _best_in, df_hist_in)
  _out_best_pred_mod, df_best_pred_mod = f_run_model_mod(spot_obj_in, 252 * _best_in, df_hist_in, df_signal_pred_model)

  _out_best_bh, df_best_bh = f_run_model_mod(spot_obj_in, 252 * _best_in, df_hist_in, df_signal_bh)
  sharpe_model_pred, sharpe_asset = f_compute_sharpe(df_agg_pred_merged, asset_ticker_in)

  #'[sharpe_asset, sharpe_model], df_out

  print('_out_best')
  print(_out_best)
  print('_out_best_pred_mod')
  print(_out_best_pred_mod)
  print('_out_best_bh')
  print(_out_best_bh)


  print('sharpe_model_pred')
  print(sharpe_model_pred)
  print('sharpe_asset')
  print(sharpe_asset)
  title_out = asset_ticker_in + '_sharpe_asset_' + '{:.2f}'.format(_out_best_pred_mod[0]) + '_trend_' + '{:.2f}'.format(_out_best[1])+ '_model_pred_' + '{:.2f}'.format(_out_best_pred_mod[1])
  f_plot_comparison(df_best, df_best_pred_mod, df_best_bh, asset_ticker_in, title_out)
  f_plot_signal_tail(df_best, df_best_pred_mod, asset_ticker_in)

  #################################### SAVE OUTPUTS ####################################
  f_save_outputs(df_best, asset_ticker_in + '_df_best')
  f_save_outputs(df_best_pred_mod, asset_ticker_in + '_df_best_pred_mod')
  f_save_outputs(df_best_bh, asset_ticker_in + '_df_best_bh')
  #################################### SAVE OUTPUTS ####################################
  plt.close("all")
  vars = "df_prediction, df_signal_pred_model, sharpe_model_pred, sharpe_asset".split(", ")
  for v in vars:
    if v in locals() or v in globals():
      exec(f"del {v}")


if __name__ == "__main__":
    tickers = ["ES=F" # E-Mini S&P 500
               "YM=F", # Mini Dow Jones
               "NQ=F", # Nasdaq 100
               "RTY=F", #E-mini Russell 2000"
               "GC=F"] # Gold
    errors = []
    for t in tickers:
        try:
            f_run_prediction_generic(method_in='bayes', asset_ticker_in=t)
        except Exception as e:
            print(e)
            errors.append((t,e))

    with open(os.path.join(_PATH_OUTPUT, "errors.txt"), "w") as f:
        for t, e in errors:
            f.write(f"{t}: {e}\n\n")
