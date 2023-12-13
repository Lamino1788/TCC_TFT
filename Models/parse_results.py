import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("TkAgg")

if __name__ == "__main__":

    dfs = []
    path = r"C:\Users\rhena\Desktop\usp\TCC\Fuma\TCC_TFT\Models\Bayes_Results"
    for p, d, files in  os.walk(path):
        for file in files:
            df = pd.read_csv(os.path.join(p, file), sep= ";")
            ticker = file.replace("_df_best_pred_mod.csv", "")
            returns = df['Close']/ df['Close'].shift(1) -1
            # vol_ewma = returns.ewm(halflife= 252, adjust = False).std().fillna(method= "bfill")
            # df["return"] = df['signal_sign'] * returns / vol_ewma.shift(1)
            df["return"] = returns * np.sign(df["signal_strenght"])
            df["ticker"] = ticker
            df = df[["Date", "ticker", "return"]]
            dfs.append(df)

    final_df = pd.concat(dfs)

    final_df = final_df.pivot(index = "Date", columns="ticker", values="return")
    final_df.add(1).cumprod().plot()
    plt.show()
    final_df.reset_index(inplace=True)


    # final_df.to_csv(os.path.join(path,"bayes_results.csv"), index = False)