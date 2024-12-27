import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.subheader('Prevendo Custos Médicos (Fictícios)')

# DATA APPS
st.write("""
         App que utilizaa Machine Learning para prever custos médicos\n
         Fonte: (Kaggle)
         """)

df = pd.read_csv('healthcare_clean.csv')

st.subheader('Informações dos dados')
user_input = st.sidebar.text_input('Digite o seu nome')

st.write("Paciente: ", user_input)

# Dados de Entrada
X= df.drop(columns=['Billing Amount'], axis=1)
Y = df['Billing Amount']

# Separa dados em treinaemnto e teste
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

# Dados do usuário com a função
def get_user_data():
    Age = st.sidebar.slider('Idade', 0, 130, 18)
    Gender = st.sidebar.slider('Gênero', 0, 1, 0)
    BloodType = st.sidebar.slider('Pressão Sanguínea', 0, 9, 2)
    MedicalCondition = st.sidebar.slider('Condição Médica', 1.0, 9.0, 9.0)
    BillingAmount = st.sidebar.slider('Custo', 0.0, 9999999999.0, 20000.00)
    AdmissionType = st.sidebar.slider('Tipo de Admissão', 0, 2, 1)
    
    user_data = {
        'Age': Age,
        'Gender': Gender,
        'Blood Type': BloodType,
        'Medical Condition': MedicalCondition,
        'Billing Amount': BillingAmount,
        'Admission Type': AdmissionType        
    }
    
    features = pd.DataFrame(user_data, index=[0])
    
    return features

user_input_variables = get_user_data()

# Gráfico
graf = st.bar_chart(user_input_variables)

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
st.write(prediction) ## Previsão: 0 : Não, 1: Sim
st.write("Custo Médico Previsto: ", prediction)
st.write("Custo Médico Real: ", user_input_variables['Billing Amount'])


st.subheader("Tabela de Dados")
st.dataframe(df)

cx_mult = st.multiselect(
    'Selecione as colunas abaixos',
    df.columns
)
st.dataframe(df[cx_mult])
