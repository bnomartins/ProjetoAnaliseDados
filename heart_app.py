import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



st.subheader('Previsão de Doença Cardiovascular')

st.html(
    """
    <h1>Previsão de Doença Cardiovascular</h1>
    <p>Este é um aplicativo simples que preve doença cardiovascular</p>
    <p>
        Atributos:\n
    </p>
    <ul>
    <li>idade: idade em anos</li>
    <li>sexo: sexo (1 = masculino; 0 = feminino)</li>
    <li>cp: tipo de dor no peito</li>
    <li>-- Valor 0: angina típica</li>
    <li>-- Valor 1: angina atípica</li>
    <li>-- Valor 2: dor não anginosa</li>
    <li>-- Valor 3: assintomática</li>
    <li>trestbps: pressão arterial em repouso (em mm Hg na admissão ao hospital)</li>
    <li>col: colesterol sérico em mg/dl</li>
    <li>fbs: (glicemia em jejum > 120 mg/dl) (1 = verdadeiro; 0 = falso)</li>
    <li>restecg: resultados eletrocardiográficos em repouso</li>
    <li>-- Valor 0: normal</li>
    <li>-- Valor 1: com anormalidade da onda ST-T (inversões da onda T e/ou elevação ou depressão do segmento ST de > 0,05 mV)</li>
    <li>-- Valor 2: mostrando hipertrofia ventricular esquerda provável ou definitiva pelos critérios de Estes</li>
    <li>thalach: frequência cardíaca máxima alcançada</li>
    <li>exang: angina induzida por exercício (1 = sim; 0 = não)</li>
    <li>oldpeak = depressão do segmento ST induzida pelo exercício em relação ao repouso</li>
    <li>inclinação: a inclinação do pico do segmento ST do exercício</li>
    <li>-- Valor 0: ascendente</li>
    <li>-- Valor 1: plano</li>
    <li>-- Valor 2: descendente</li>
    <li>ca: número de vasos principais (0-3) coloridos pela fluorosopia</li>
    <li>thal: 0 = normal; 1 = defeito corrigido; 2 = defeito reversível</li>
    <li>e o rótulo</li>
    <li>condição: 0 = sem doença, 1 = doença</li>
    </ul>
    <p>
        Agradecimentos:
        <ul>
        <li>Dados publicados no Kaggle: https://www.kaggle.com/ronitf/heart-disease-uci</li>
        <li>Descrição dos dados acima: https://www.kaggle.com/ronitf/heart-disease-uci/discussion/105877</li>
        <li>Dados originais https://archive.ics.uci.edu/ml/datasets/Heart+Disease</li>
        </ul>
        
        Fonte:
        <ul>
        <li>Instituto Húngaro de Cardiologia. Budapeste: Andras Janosi, MD</li>
        <li>Hospital Universitário, Zurique, Suíça: William Steinbr</li>
        <li>Instituto Húngaro de Cardiologia. Budapeste: Andras Janosi, MD</li>
        <li>Hospital Universitário, Zurique, Suíça: William Steinbrunn, MD</li>
        <li>Hospital Universitário, Basiléia, Suíça: Matthias Pfisterer, MD</li>
        <li>VA Medical Center, Long Beach e Cleveland Clinic Foundation: Robert Detrano, MD, Ph.D.</li>
        <li>Doador: David W. Aha (aha '@' ics.uci.edu) (714) 856-8779</li>
        </ul>
    </p>
        
    """
)


df = pd.read_csv('heart_cleveland.csv')

st.subheader('Informações dos dados')
user_input = st.sidebar.text_input('Digite o seu nome')

st.write("Paciente: ", user_input)

# Dados de Entrada
X= df.drop(columns=['condition'], axis=1)
Y = df['condition']

# Separa dados em treinaemnto e teste
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

# Dados do usuário com a função
def get_user_data():
    age = st.sidebar.slider('Idade', 0, 130, 3)
    sex = st.sidebar.slider('Sexo', 0, 1, 0)
    cp = st.sidebar.slider('Tipo de Dor no Peito', 0, 3, 0)
    trestbps = st.sidebar.slider('Pressão Arterial (Em Repouso)', 0, 2, 0)
    chol = st.sidebar.slider('Colesterol', 0.0, 700.0, 50.0)
    fbs = st.sidebar.slider('Glicemia (Em Jejum)', 0, 1, 0)
    restecg = st.sidebar.slider('Res. Eletrocardiograma (Em Repouso)', 0, 2, 1)
    thalach = st.sidebar.slider('Frequência Cardíaca Máxima', 60, 202, 120)
    exang = st.sidebar.slider('Angina Induzida (Exercício). ', 0, 1, 0)
    oldpeak = st.sidebar.slider('Oldpeak ', 0.1, 5.0, 4.0)
    slope = st.sidebar.slider('Inclinação (Pico Exercício)', 0, 2, 1)
    ca = st.sidebar.slider('CA', 0, 2, 1)
    thal = st.sidebar.slider('THAL', 0, 1, 1)
    
    user_data = {
        'age' : age,
        'sex' : sex,
        'cp' : cp,
        'trestbps' : trestbps,
        'chol' : chol,
        'fbs' : fbs,
        'restecg' : restecg,
        'thalach' : thalach,
        'exang' : exang,
        'oldpeak' : oldpeak,
        'slope' : slope,
        'ca' : ca,
        'thal' : thal
    }
    
    features = pd.DataFrame(user_data, index=[0])
    
    return features

user_input_variables = get_user_data()

# Gráfico
graf = st.bar_chart(user_input_variables)




# #vamos aproveitar o dataframe criado anteriormente
# df2 = pd.DataFrame(df, columns=['Cholesterol', 'MaxHR'])
# #exemplo de gráfico 
# st.area_chart(df2)

st.scatter_chart(df, x='age', y='oldpeak', color='condition')



st.subheader("Dados do Usuário")
st.write(user_input_variables)

dtc = DecisionTreeClassifier(criterion='entropy', max_depth=3)
dtc.fit(X_train, y_train)

# Acurácia do Modelo
st.subheader("Acurácia do Modelo")
st.write(accuracy_score(y_test, dtc.predict(X_test))*100)

# Previsão
prediction = dtc.predict(user_input_variables)

st.subheader("Previsão")
st.write(prediction)

st.subheader("Tabela de Dados")
st.dataframe(df)

cx_mult = st.multiselect(
    'Selecione as colunas abaixos',
    df.columns
)
st.dataframe(df[cx_mult])



