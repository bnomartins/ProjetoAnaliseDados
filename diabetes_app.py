import pandas as pd
import streamlit as st

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

st.subheader('Prevendo Diabetes')

# DATA APPS
st.write("""
         App que utilizaa Machine Learning para prever possível diabestes\n
         Fonte: PIMA - INDIA (Kaggle)
         """)

df = pd.read_csv('diabetes.csv')

st.subheader('Informações dos dados')
user_input = st.sidebar.text_input('Digite o seu nome')

st.write("Paciente: ", user_input)

# Dados de Entrada
X= df.drop(columns=['Outcome'], axis=1)
Y = df['Outcome']

# Separa dados em treinaemnto e teste
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=42)

# Dados do usuário com a função
def get_user_data():
    pregnancies = st.sidebar.slider('Gravidez', 0, 17, 3)
    glucose = st.sidebar.slider('Glicose', 0, 200, 110)
    blood_pressure = st.sidebar.slider('Pressão Sanguínea', 0, 122, 72)
    skin_thickness = st.sidebar.slider('Espessura da Pele', 0, 99, 23)
    insulin = st.sidebar.slider('Insulina', 0.0, 846.0, 30.0)
    bmi = st.sidebar.slider('IMC', 0.0, 67.1, 32.0)
    dpf = st.sidebar.slider('Histórico Familiar', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Idade', 21, 81, 29)
    
    user_data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf,
        'Age': age
        
    }
    
    features = pd.DataFrame(user_data, index=[0])
    
    return features

user_input_variables = get_user_data()

# Gráfico
graf = st.bar_chart(user_input_variables)



st.scatter_chart(df, x='Glucose', y='Insulin')




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
st.write(prediction) ## 0 : Não Diabético, 1: Diabético
st.write("Diagnóstico: ")
if prediction == 1:
    st.write("Provável Diabético")
else:
    st.write("Provável Não Diabético")


st.subheader("Tabela de Dados")
st.dataframe(df)

cx_mult = st.multiselect(
    'Selecione as colunas abaixos',
    df.columns
)
st.dataframe(df[cx_mult])
