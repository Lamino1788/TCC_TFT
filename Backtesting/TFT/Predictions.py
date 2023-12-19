import pandas as pd


def make_predictions(test_data: pd.DataFrame, best_tft, max_encoder_length, max_prediction_length):
    j = 0
    test_data = test_data.sort_values(["Date", "Ticker"])
    output = {}
    for ticker_out in test_data.Ticker.unique():
      aux_df = test_data.loc[test_data.Ticker == ticker_out]
      indices = aux_df.iloc[max_encoder_length+max_prediction_length:].Time_Fix
      for i in indices:
        rolling_df = aux_df.loc[aux_df.Time_Fix.isin(list(range(i-max_encoder_length-max_prediction_length, i)))]
        if j == max_prediction_length:
            j = 0
        if j == 0:
            test_predictions = best_tft.predict(rolling_df, mode="raw", return_x=True)
            counter = 0
            for ticker in rolling_df.Ticker.unique():
                if ticker == ticker_out:
                    pred = test_predictions.output.prediction[counter].cpu().numpy().cumsum()[-1]
                    if ticker not in output.keys():
                        output[ticker] = [pred]
                        output[ticker + "_Real"] = [rolling_df["Close_" + ticker].iloc[-max_prediction_length]]
                        output[ticker + "_Date"] = [rolling_df.loc[rolling_df.Ticker == ticker].Date.iloc[-max_prediction_length]]
                    else:
                        output[ticker] = output[ticker] + [pred]
                        output[ticker + "_Real"] = output[ticker + "_Real"] + [rolling_df["Close_" + ticker].iloc[-max_prediction_length]]
                        output[ticker + "_Date"] = output[ticker + "_Date"] + [rolling_df.loc[rolling_df.Ticker == ticker].Date.iloc[-max_prediction_length]]
                    counter = counter + 1
            j += 1
        else:
            rolling_df = test_data.loc[test_data.Time_Fix.isin(list(range(i-max_encoder_length-max_prediction_length, i)))]
            counter = 0
            for ticker in rolling_df.Ticker.unique():
                if ticker == ticker_out:
                    if ticker in output.keys():
                        output[ticker + "_Real"] = output[ticker + "_Real"] + [rolling_df["Close_" + ticker].iloc[-max_prediction_length]]
                        output[ticker + "_Date"] = output[ticker + "_Date"] + [rolling_df.loc[rolling_df.Ticker == ticker].Date.iloc[-max_prediction_length]]
                        output[ticker] = output[ticker] + [output[ticker][-1]]
                counter = counter + 1
                j += 1
    return output