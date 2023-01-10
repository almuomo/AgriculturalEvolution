import streamlit as st
import numpy as np
import pandas as pd

import matplotlib as mpl
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from statsmodels.graphics.factorplots import interaction_plot
from plotly.subplots import make_subplots


st.title("EVOLUCIÓN AGRÍCOLA EN ESTADOS UNIDOS")

st.image('../NOTEBOOKS/images/agricultura-de-precisin.jpg', caption='Fuente: Acre Group')
#st.subheader('How to run streamlit from windows')
st.write('El cambio climático afecta directamente a la agricultura; estas alteraciones están principalmente relacionadas con el incremento de temperaturas que tienden a prolongar los periodos de sequía y con las precipitaciones que cada vez son menos frecuentes y más concentradas en el tiempo. Ambos factores pueden generar problemas como: inundaciones, cambios anormales en las temperaturas en distintas épocas del año, sequias prolongadas, escasez de agua, etc. Los efectos, entre otros, que estas problemáticas pueden tener sobre los cultivos son: estrés hídrico, proliferación de plagas, floraciones en épocas del año que no corresponden, incendios, inundaciones o dificultad en el desarrollo vegetativo.')
st.write('Estos factores han hecho que la agricultura trate de adaptarse a los cambios producidos por estos problemas tendiendo hacia un cambio en los cultivos, ya sea modificando el tipo de cultivo de la región o cambiando a nuevas variedades de cultivos más resistentes.')
st.write('Este trabajo tiene como objetivo el estudio de los cambios y modificaciones que se han ido produciendo en los cultivos de la región de Estados Unidos.')
st.write('Autor: Alejandro Muñoz Molina')

st.sidebar.image('../NOTEBOOKS/images/agricultura-de-precisin.jpg', width=300)

##############
#indice lateral
sidebar = st.sidebar

st.title("Representación gráfica de las variables de estudio")

#Carga datos
df =  pd.read_csv('../DATOS/archivos creados analisis/agrupation.csv')
df.set_index('date', inplace=True)

#Mostrar tabla de datos:
df_display = sidebar.checkbox("Mostrat tabla Variables", value=True)
if df_display:
    st.write(df)

#Creación rango años
slider = sidebar.slider('Años a representar', min_value= 1950, max_value=2021, value =  2021, step=1)

subfig = make_subplots(specs=[[{"secondary_y": True}]])
fig1 = px.line(x = slider, y = df['Temperature'], color_discrete_sequence=px.colors.qualitative.G10)
fig2 = px.line(x = slider, y = df['Precipitation'], color_discrete_sequence=px.colors.qualitative.Dark2)
fig2.update_traces(yaxis='y2')
subfig.add_traces(fig1.data + fig2.data)
subfig.layout.yaxis.title="Temperatura"
subfig.layout.yaxis2.title="Precipitación"

#st.plotly_chart(subfig, theme=None, use_container_width=True)
st.write(subfig, slider)
