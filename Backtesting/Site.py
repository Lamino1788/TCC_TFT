import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
from utils import *

pd.options.plotting.backend = "plotly"

# from pathlib import Path
AVAILABLE_TICKERS = {
      "Renda Fixa": ['ZF=F', 'ZT=F', 'ZB=F', "ZN=F"],
      "Equity": ["ES=F", "YM=F", "NQ=F"],
      "Commodity": ["GC=F"],
      "Moedas": ["EUR=X", "JPY=X", "GBP=X", 'BRL=X', "MXN=X", "CAD=X"]
}

AVAILABLE_CLASS = [
    "Equity",
    "Renda Fixa",
    "Commodity",
    "Moedas"
]

# Configuracoes do Streamlit
def config():

    side_bar = st.sidebar

    with side_bar.container():

        classe = side_bar.selectbox(
        label = "Classe de Ativo",
        key = 'classe',
        options = AVAILABLE_CLASS
        )

        ticker = side_bar.selectbox(
        label = "Ativo Analisado",
        key = 'ticker',
        options = AVAILABLE_TICKERS[classe]
        )

    return ticker, classe

def main(ticker, classe):
            base_df = pd.read_csv(f'Results/{ticker}.csv').set_index('date')
            base_df[f'{ticker} Long Only'] = base_df[f'{ticker} Long Only'].astype(float)
            base_df[f'{ticker} Using TFT'] = base_df[f'{ticker} Using TFT'].astype(float)
            
            graph_df = pd.DataFrame({'date': base_df.index, 'Modelo TFT': ((base_df[f'{ticker} Using TFT'].add(1).cumprod()).sub(1)).mul(100), "Long Only": ((base_df[f'{ticker} Long Only'].add(1).cumprod()).sub(1)).mul(100)}).set_index('date').dropna()
            graph_df.index = pd.to_datetime(graph_df.index)
            macd = pd.read_csv("Results/macd.csv").set_index("Date").fillna(0)
            macd.index.rename("date", inplace=True)
            macd.index = pd.to_datetime(macd.index)
            macd = macd[list(graph_df.index)[0]:list(graph_df.index)[-1]]

            moskowitz = pd.read_csv("Results/moskowitz.csv").set_index("Date").fillna(0)
            moskowitz.index.rename("date", inplace=True)
            moskowitz.index = pd.to_datetime(moskowitz.index)
            moskowitz = moskowitz[list(graph_df.index)[0]:list(graph_df.index)[-1]]

            bayes = pd.read_csv("Results/bayes_results.csv").set_index("Date").fillna(0)
            bayes.index.rename("date", inplace=True)
            bayes.index = pd.to_datetime(bayes.index)
            bayes = bayes[list(graph_df.index)[0]:list(graph_df.index)[-1]]

            graph_df["Moskowitz"] = ((moskowitz[ticker]).add(1).cumprod().sub(1)).mul(100)
            # graph_df["Bayes"] = bayes[ticker]
            # graph_df["Bayes"] = graph_df["Bayes"].fillna(0).add(1).cumprod().sub(1).mul(100)
            graph_df["MACD"] = ((macd[ticker]).add(1).cumprod().sub(1)).mul(100)



            fig = px.line(graph_df, x=graph_df.index, y=graph_df.columns, labels={'value': 'Retorno Acumulado (%)', 'variable': 'Série', 'date': 'Data'}, )

            metrics, benchmark_metrics = get_metrics(pd.Series(base_df[f'{ticker} Using TFT'], dtype=float), pd.Series(base_df[f'{ticker} Long Only'], dtype=float), 0)
            fig2 = px.histogram(base_df, x=[f'{ticker} Using TFT', f'{ticker} Long Only'],nbins=10)

            
            test = {'Retorno': [], 'Probabilidade(%)': []}
            for i in pd.Series(fig2.data[0]['x']).dropna().round(4).unique():
                test.update({'Retorno': test['Retorno'] + [i], 'Probabilidade(%)': test['Probabilidade(%)'] + [pd.Series(fig2.data[0]['x']).round(4).value_counts()[i]]})
            test = pd.DataFrame(test)
            test['Probabilidade(%)'] = (test['Probabilidade(%)']/(test['Probabilidade(%)'].sum()))*100

            fig2 = px.scatter(test, x='Retorno', y='Probabilidade(%)', trendline='lowess', trendline_color_override='lightblue', trendline_options=dict(frac=0.2))
            fig2.update_layout(showlegend=False, title='Distribuição dos Retornos', xaxis_title='Retorno', yaxis_title='Probabilidade(%)', xaxis_range=[-0.01, 0.015])
            fig2.update_traces(visible=False, selector=dict(mode='markers'))
            fig2.add_vline(x=base_df[f'{ticker} Using TFT'].mean(), line_width=1, line_dash="dash", line_color="white")
            
            return fig, fig2, metrics, base_df

################################################################################################################################### Execution

if __name__ == "__main__":
    
    st.set_page_config(
        "Trend Following com o TFT - TCC",
        "📊",
        initial_sidebar_state="expanded",
        layout="wide",
    )
    ticker, classe  = config()
    
    st.header(ticker)
    fig, fig2, metrics, base_df = main(ticker, classe)
    st.plotly_chart(fig, use_container_width=True)
    st.subheader('Metrics')
    display_metrics(fig2, metrics, ticker)
    # base_df.set_index('date', inplace=True)
    # base_df = base_df.rename(columns={f'{ticker}': 'tmp'})
    # base_df = base_df.rename(columns={'Rfinal': f'{ticker}'})
    st.download_button('Download Series CSV', base_df.to_csv(), file_name=f'{ticker}.csv', mime='text/csv')
    
