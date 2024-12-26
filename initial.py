import streamlit as st
import pandas as pd

st.set_page_config(page_title="Healthcare App", page_icon=":hospital:")

with st.container():
    st.title("Healthcare App")
    st.write("This is a simple app that predicts diseases")
    st.write("Please select a disease from the sidebar to get started")
    st.write("You can also select a page from the navigation bar")

@st.cache_data
def carregar_dados():
    tabela = pd.read_csv('heart_cleveland.csv')
    return tabela

with st.container():
    st.write('---')
    qtde_dias = st.selectbox('Selecione a quantidade de dias', ['1D', '5D', '10D', '20D', '30D', '45D'])
    num_dias = int(qtde_dias.replace('D', ''))
    dados = carregar_dados()
    dados = dados[-num_dias:]
    st.area_chart(dados, x='age', y='chol')
    