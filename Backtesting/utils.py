import pandas as pd
import empyrical as ep
from mergedeep import merge
import numpy as np
import streamlit as st
import datetime

def return_to_nested(dictionary, l=[]):
    for key, value in dictionary.items():
        if '__' in key:
            for k in reversed(key.split('__')):
                value = {k: value}
            l.append(value)
        else:
            l.append({key: value})
    return merge(*l)

def time_underwater(strategy):
    cum_returns = ep.cum_returns(strategy, starting_value=1)
    aux_ = (cum_returns["Strategy"]/cum_returns["Benchmark"])
    aux_.name = "Strategy"
    map_drawdown = pd.DataFrame(
        np.where(cum_returns["Strategy"] < aux_.expanding(0).max(), True, False),
        index = cum_returns.index,
        columns=["Strategy"]
    )
    tuw = cum_returns[["Strategy"]][map_drawdown].expanding(0).count() - cum_returns[["Strategy"]][map_drawdown].expanding(0).count()[~map_drawdown].ffill()
    return tuw

def get_metrics(session_returns,benchmark_returns, omega):
    strategy = pd.concat(
        [
            session_returns,
            benchmark_returns
        ],
        axis=1,
        keys=[
            "Strategy", "Benchmark"
        ]
    )
    
    bench_metrics = {
        'total_return': ep.cum_returns_final(strategy["Benchmark"]),
        'sharpe_ratio': ep.sharpe_ratio(strategy["Benchmark"]),
        'sortino_ratio': ep.sortino_ratio(strategy["Benchmark"]),
        'omega_ratio': ep.omega_ratio(strategy["Benchmark"]),
        'calmar_ratio': ep.calmar_ratio(strategy["Benchmark"]),
        'annual_volatility': ep.annual_volatility(strategy["Benchmark"]),
        'cvar': ep.conditional_value_at_risk(strategy["Benchmark"]),
        'downside_risk': ep.downside_risk(strategy["Benchmark"]),
        'max_dd': ep.max_drawdown(strategy["Benchmark"]),
        'cagr': ep.cagr(strategy["Benchmark"], annualization=1)
    }
    strategy_metrics = {
        'total_return': ep.cum_returns_final(strategy["Strategy"]),
        'sharpe_ratio': ep.sharpe_ratio(strategy["Strategy"], risk_free=0,period='daily'),
        'sortino_ratio': ep.sortino_ratio(strategy["Strategy"]),
        'omega_ratio': ep.omega_ratio(strategy["Strategy"], risk_free=strategy["Benchmark"].mean(), required_return=omega, annualization=1),
        'calmar_ratio': ep.calmar_ratio(strategy["Strategy"]),
        'annual_volatility': ep.annual_volatility(strategy["Strategy"]),
        'cvar': ep.conditional_value_at_risk(strategy["Strategy"]),
        'downside_risk': ep.downside_risk(strategy["Strategy"], required_return=0),
        'max_dd': ep.max_drawdown(strategy["Strategy"]),
        'skewness': strategy["Strategy"].skew(),
        'kurtosis': strategy["Strategy"].kurtosis(),
        'max_time_underwater': time_underwater(strategy).max().values[0],
        'mean_time_underwater': time_underwater(strategy)[time_underwater(strategy)>5].mean().values[0],
        'max_daily_return': strategy["Strategy"].max()*100,
        'min_daily_return': strategy["Strategy"].min()*100,
        'beta': ep.beta(strategy["Strategy"], strategy["Benchmark"]),
        'cagr': ep.cagr(strategy["Strategy"], annualization=1)
    }
    # strategy_metrics['cagr'] = (1+strategy_metrics['cagr'])/(1+bench_metrics['cagr']) -1
    strategy_metrics['cagr'] = ((session_returns.add(1).cumprod().iloc[-1])**(1/(len(session_returns)/252))) - 1
    bench_metrics['cagr'] = ((benchmark_returns.add(1).cumprod().iloc[-1])**(1/(len(benchmark_returns)/252))) - 1
    
    strategy_metrics['cagr'] = (1+strategy_metrics['cagr'])/(1+bench_metrics['cagr']) -1

    strategy_metrics['above'] = (session_returns.add(1).cumprod().iloc[-1]) ** (252/len(session_returns)) - (benchmark_returns.add(1).cumprod().iloc[-1]) ** (252/len(benchmark_returns))
    
    strategy_metrics['more'] = (session_returns.add(1).cumprod().iloc[-1])/(benchmark_returns.add(1).cumprod().iloc[-1])
    
    return strategy_metrics, bench_metrics

def display_metrics(fig2, metrics, bench):
    tmp, metrics1, metrics2, metrics3 = st.columns([3, 1,1,1], gap = "small") 
    tmp.plotly_chart(fig2)
    metrics1.metric("Retorno Acumulado", f"{round(metrics['total_return']*100,2)} %")
    metrics1.write("")
    if round(100*metrics['above'],2) > 0:
        metrics3.metric(f"Retorno Anual Comparado a Long Only", f"{bench} + {round(100*metrics['above'],2)}%")
    else:
        metrics3.metric(f"Retorno Anual Comparado a Long Only", f"{bench} {round(100*metrics['above'],2)}%")
    metrics3.write("")
    metrics2.metric(f"Porcentagem do Retorno Long Only", f"{round(metrics['more']*100,2)} %")
    metrics2.write("")
    metrics3.metric("Sharpe Ratio", round(metrics['sharpe_ratio'],2))
    metrics3.write("")
    metrics2.metric("Volatilidade", f"{round(100*metrics['annual_volatility'],2)}%")
    metrics2.write("")

    metrics1.metric("Retorno Diário Min/Max", f"{round(metrics['min_daily_return'],2)} / {round(metrics['max_daily_return'],2)} %")
    metrics1.write("")
    
    metrics2.metric("Downside Risk", f"{round(100*metrics['downside_risk'],2)}%")
    metrics2.write("")
    metrics3.metric("Drawdown Máximo", f"{round(100*metrics['max_dd'],2)}%")
    metrics3.write("")

    metrics1.metric(f"CAGR Sobre Long Only", f"{round(100*metrics['cagr'],2)}%")
    metrics1.write("")
    metrics2.metric("Omega", round(metrics['omega_ratio'],2))
    # metrics3.metric("Calmar", round(metrics['calmar_ratio'],3)) 

    metrics3.metric("CVaR", f"{round(100*metrics['cvar'],2)}%")

    metrics1.metric("Sortino", round(metrics['sortino_ratio'],2))
    

    # metrics1.metric("Skewness", round(metrics['skewness'],3))
    # metrics2.metric("Kurtosis", round(metrics['kurtosis'],3))
    # metrics3.metric("Mean/Max TuW (Days)", f"{round(metrics['mean_time_underwater'], 0)} / {round(metrics['max_time_underwater'], 0)}")

    
    

def input_tax(base_df):
    start_date = list(base_df.index)[0]
    end_date = start_date + datetime.timedelta(days=int(365/2) + 1)
    last_line = 0
    test = pd.Series()
    dates = []
    while start_date < list(base_df.index)[-1]:
        try:
            df = base_df.loc[start_date:end_date].reset_index()
        except:
            df = base_df.loc[start_date:].reset_index()

        try:
            df = pd.concat([pd.DataFrame(last_line).T, df]).reset_index(drop=True)
        except:
            pass

        df['Nacumul'] = df['tax'].add(1).cumprod()
        df['Ncap'] = 0
        df.loc[df['Nacumul'] >= 1, 'Ncap'] = df['Nacumul'] - 1
        df['Nefeito'] = 0

        for i in df.index:
            if i == 0:
                continue
            df.loc[i, 'Nefeito'] = df.loc[i, 'Ncap'] - df.loc[i - 1, 'Ncap'] 
        
        df['Rcotista'] = df['Fundo Resultado'] - df['Nefeito']
        base_df.loc[start_date:end_date, 'Rcotista'] = df.set_index('date')['Rcotista'].astype(float)
        df['Racumul'] = df['Rcotista'].add(1).cumprod()
        
        
        if start_date != list(base_df.index)[0]:
            df['Racumul'] = df['Racumul'] * last_line['Racumul']
        test = pd.concat([test, df['Racumul']], axis=0, ignore_index=True)

        last_line = df.iloc[-1]
        dates.append(end_date)
        start_date = end_date + datetime.timedelta(days=1)
        end_date = start_date + datetime.timedelta(days=int(365/2))
    
    base_df = base_df.reset_index()
    base_df['Rfinal'] = test

    return base_df, dates