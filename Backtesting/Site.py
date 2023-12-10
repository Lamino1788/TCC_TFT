import datetime
import pandas as pd
import streamlit as st
import plotly.express as px
from utils import *

pd.options.plotting.backend = "plotly"

# from pathlib import Path
AVAILABLE_TICKERS = {
      "Renda Fixa": ["Carteira", 'ZF=F', 'ZT=F', 'ZB=F', "ZN=F"],
      "Equity": ["Carteira", "ES=F", "YM=F", "NQ=F"],
      "Commodity": ["GC=F"],
      "Moeda": ["Carteira", "EUR=X", "JPY=X", "GBP=X", 'BRL=X', "MXN=X", "CAD=X"]
}

AVAILABLE_CLASS = [
    "Equity",
    "Renda Fixa",
    "Commodity",
    "Moeda"
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
        if classe == "Renda Fixa":
            classe = "Renda_Fixa"

    return ticker, classe

def main(ticker, classe):
            if ticker == "Carteira":
                base_df = pd.read_csv(f'Backtesting/Results/{ticker}{classe}.csv').set_index('date')
            else:
                base_df = pd.read_csv(f'Backtesting/Results/{ticker}.csv').set_index('date')
            base_df[f'{ticker} Long Only'] = base_df[f'{ticker} Long Only'].astype(float)
            base_df[f'{ticker} Using TFT'] = base_df[f'{ticker} Using TFT'].astype(float)
            
            graph_df = pd.DataFrame({'date': base_df.index, 'Modelo TFT': ((base_df[f'{ticker} Using TFT'].add(1).cumprod()).sub(1)).mul(100), "Long Only": ((base_df[f'{ticker} Long Only'].add(1).cumprod()).sub(1)).mul(100)}).set_index('date').dropna()
            graph_df.index = pd.to_datetime(graph_df.index)
            
            macd = pd.read_csv("Backtesting/Results/macd.csv").set_index("Date").fillna(0)
            macd.index.rename("date", inplace=True)
            macd.index = pd.to_datetime(macd.index)
            macd = macd.loc[graph_df.index]

            moskowitz = pd.read_csv("Backtesting/Results/moskowitz.csv").set_index("Date").fillna(0)
            moskowitz.index.rename("date", inplace=True)
            moskowitz.index = pd.to_datetime(moskowitz.index)
            moskowitz = moskowitz.loc[graph_df.index]

            bayes = pd.read_csv("Backtesting/Results/bayes_results.csv").set_index("Date").fillna(0)
            bayes.index.rename("date", inplace=True)
            bayes.index = pd.to_datetime(bayes.index)
            bayes = bayes[list(graph_df.index)[0]:list(graph_df.index)[-1]]

            if ticker == "Carteira":
                graph_df["Moskowitz"] = ((moskowitz[ticker+classe]).add(1).cumprod().sub(1)).mul(100)
                graph_df["MACD"] = ((macd[ticker+classe]).add(1).cumprod().sub(1)).mul(100)
                if classe == "Equity":
                    graph_df["Bayes"] = bayes[ticker+classe].add(1).cumprod().sub(1).mul(100)
            else:
                graph_df["Moskowitz"] = ((moskowitz[ticker]).add(1).cumprod().sub(1)).mul(100)
                graph_df["MACD"] = ((macd[ticker]).add(1).cumprod().sub(1)).mul(100)
                if classe == "Equity" or ticker == "GC=F":
                    graph_df["Bayes"] = bayes[ticker].add(1).cumprod().sub(1).mul(100)

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
        initial_sidebar_state="auto",
        layout="wide",
    )
    tab1, tab2, tab3 = st.tabs(["Introdução", "Resultados", "Equipe"])

    with tab1:
        st.header("Análise de Séries Temporais Financeiras e Trend Following Utilizando o Modelo Temporal Fusion Transformer")
        col1, col2 = st.columns(2, gap="large")
        with col1:
            st.header("O que é Trend Following?")
            st.write("Trend Following é um tipo de estratégia de investimentos, a qual baseia-se em definir o tamanho da posição de investimento em um determinado ativo, a partir da identificação de tendências na sua série temporal de preços.")
            st.write("")
            st.header("Por que usar o Temporal Fusion Transformer?")
            st.write("Classicamente, as estratégias de Trend Following depedem de parâmetros subjetivos. Assim, é interessante analisar a viabilidade de abordagens mais orientadas à dados. \n Desta forma, o Temporal Fusion Transformer (TFT) surge como uma alternativa ao problema. Pois, sua arquitetura é voltada para análise de séries temporais e interpretabilidade – essencial no contexto de alocação de patrimônio.")

        with col2:
            st.header("Atributos do Modelo")
            st.write("A partir da séries de preços de fechamento dos ativos, indicadores de Trend Following foram incorporados aos atributos modelo, que são: \
                \n - Mês \n - Dia do mês \n - Retorno atrasado correspondente ao tamanho do encoder \n - Retorno atrasado correspondente ao tamanho da previsão \n \
                - Média móvel de 22 dias do retorno \n - Média móvel da volatilidade \n - Média móvel exponencial do retorno \n \
                - Índice de força relativa (7, 14, 22, 30 e 60 dias) \n - Convergência Divergência de médias móveis")

        
        st.header("Desenvolvimento da Estratégia")
        st.write("Extraímos o preço de fechamento de contratos futuros da API do Yahoo Finance. \
            Geramos indicadores deTrend Following a partir da séries de preços e combinamos todos esses dados como entradas do TFT. \
            \n A partir das entradas, o TFT prevê os próximos preços de fechamento dos contratos. Com o preço previsto, épossível estimar o retorno esperado. \
            \n A estratégia então consiste em comprar o ativo caso o retorno esperado seja positivo, e vender caso contrário.")
        st.image("Backtesting/Images/Arquitetura.png",  width= 1000, caption="Arquitetura do Modelo")

        st.header("Resultados")
        st.write("Usando uma carteira igualmente balanceada de futuros de ETFs. A estratégia desenvolvida supera outras estratégias de Trend Following nas pricipais métricas de desempenho.")
        st.image("Backtesting/Images/Tabela.png", width=1000, caption="Resultados por Modelo")

    with tab2:
        ticker, classe  = config()
        try:
            st.header(ticker + " - " + classe)
            fig, fig2, metrics, base_df = main(ticker, classe)
            st.plotly_chart(fig, use_container_width=True)
            st.subheader('Metrics')
            display_metrics(fig2, metrics, ticker)
            st.download_button('Download Series CSV', base_df.to_csv(), file_name=f'{ticker}.csv', mime='text/csv')
        except:
            pass
    
    with tab3:
        col1, col2 = st.columns(2, gap="small")

        with col1:
            st.header("Alunos: ")
            st.write(" - Luiz Felipe Alamino de Lima \n - Rhenan Silva Nehlsen")

            st.header("Orientador: ")
            st.write(" - Prof. Dr. Edson Satoshi Gomi")
            
            st.header("Co-Orientador: ")
            st.write(" - Fábio Katsumi Shinohara de Souza")
        with col2:
            st.header("Universidade: ")
            st.write(" - Escola Politécnica da Universidade de São Paulo")

            st.header("Links: ")
            st.write(" - Banner: \n - Press Release: \n - Monografia:")
    
