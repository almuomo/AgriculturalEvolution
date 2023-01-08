
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

#escribir texto
st.title('input example')
texto = st.text_input('insert text')
st.markdown(f'el texto que has escrito es: {texto}')

#mostrar tabla de datos (con el @ haces que esto se guarde ne chache y no lo vuelve a cargar)
@st.cache
def get_wine():
    df = pd.read_csv('https://raw.githubusercontent.com/KaonToPion/datasets/main/final_wine.csv')
    return df

#mostrar una selección    
st.radio('movie',('comedy', 'terror', 'action'))

df = get_wine()
st.dataframe(df.head())
