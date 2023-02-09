import streamlit as st
import numpy as np
import pandas as pd
from fpdf import FPDF
import base64
import matplotlib as mpl
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from statsmodels.graphics.factorplots import interaction_plot
from plotly.subplots import make_subplots

# LECTURA DATAFRAMES
#------------------------------------------------#
df_agrupation =  pd.read_csv('../DATOS/archivos creados analisis/agrupation.csv')
df_agrupation.set_index('date', inplace=True)

df_pp = pd.read_csv('../DATOS/archivos creados analisis/datos_PP.csv')
df_pp.set_index('date', inplace=True)

df_T = pd.read_csv('../DATOS/archivos creados analisis/datos_T.csv')
df_T.set_index('date', inplace=True)

states_pp = pd.pivot_table(df_pp, index = df_pp.index, values = 'Precipitation', columns='State', aggfunc=np.mean)
states_T = pd.pivot_table(df_T, index = df_T.index, values = 'Temperature', columns='State', aggfunc=np.mean)

df_sup_estado = pd.read_csv('../DATOS/archivos creados analisis/superficie_por_estado.csv')
df_sup_cultivo =  pd.read_csv('../DATOS/archivos creados analisis/superficie_por_cultivos.csv')


#INDICE SUPERIOR
tab1, tab2, tab3, tab4, tab5 = st.tabs(["INTRODUCCIÓN", "CLIMA", "SUPERFICIE", "EEUU", "DATOS"])

with tab1:
    #INTRODUCCIÓN
    st.title("EVOLUCIÓN AGRÍCOLA EN ESTADOS UNIDOS")

    st.image('../NOTEBOOKS/images/agricultura-de-precisin.jpg', caption='Fuente: Acre Group')
    st.write('Estados Unidos es uno de los principales productores agrícolas a nivel global, especialmente en productos como la soja, maíz y trigo. Esto es posible gracias a las vastas extensiones de superficies agrícolas que pueden llegar a encontrarse y también a la diversidad de climas que están presentes en este país.')
    st.write('A lo largo de la historia estas superficies agrícolas se han ido modificando. Actualmente el crecimiento de estas superficies no es una opción viable pero sí que lo es la modificación y adaptación de los cultivos influenciados por los nuevos factores a los que el sector agrario se enfrenta a día de hoy.')
    st.write('Algunos ejemplos de estos factores pueden ser: el cambio climático, introducción de nuevas tecnologías en la agricultura o las políticas que se han ido desarrollando a lo largo de los años.')
    st.write('Este trabajo está orientado al estudio de los factores climáticos y cuál ha sido la influencia en los cultivos agrícolas de Estados Unidos.')
    st.write('Una vez realizado el estudio, se ha visto que la evolución de las temperaturas a lo largo de las décadas sí que ha tenido influencia en la modificación y evolución de las superficies de los principales cultivos de EEUU mientras que la precipitación no ha sido una variable tan influyente, ya que esta puede ser sustituida por técnicas de riego.')

    st.write('*Toda esta información se puede encontrar en el documento "Memoria.pdf" el cual queda disponible en el enlace de descarga situado más abajo.*')
   
    st.write('Autor: Alejandro Muñoz Molina')
    st.write('Linkedin: https://www.linkedin.com/in/alex245/')

    #DESCARGA MEMORIA DEL PROYECTO
    with open("../Memoria.pdf", "rb") as pdf_file:
        PDFbyte = pdf_file.read()

    st.download_button(label="Descarga Memoria proyecto",
                       data=PDFbyte,
                       file_name="Memoria.pdf",
                       mime='application/octet-stream')


with tab2:
    st.write('Este apartado tiene como objetivo mostrar la evolución de las variables de temperatura y precipitación a lo largo de los años en Estados Unidos.')
    st.write('*Nota: Haciendo doble click en los diferentes nombres de las leyenda se quedaran representado únicamente el elemento seleccionado, también se pueden ir añadiendo o eliminando elementos de las leyendas para una mejor visualización de las gráficas.*')

    #VARIABLES TEMPERATURA Y PRECIPITACIÓN
    df_display = st.checkbox("Evolución variables clima en EEUU", value=True)
    if df_display:
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=df_agrupation.index, y=df_agrupation['Temperature'], name="Temperature"), secondary_y=False,)
        fig.add_trace(go.Scatter(x=df_agrupation.index, y=df_agrupation['Precipitation'], name="Precipitation"), secondary_y=True,)

        fig.update_layout(title_text="Evolución de las variables Temperatura y Precipitación en EEUU")

        #fig.update_xaxes(title_text="date")
        fig.update_yaxes(title_text="<b>Temperatura (ºC)</b>", secondary_y=False)
        fig.update_yaxes(title_text="<b>Precipitation (mm)</b>", secondary_y=True)

        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=0.68))

        st.plotly_chart(fig, theme=None, use_container_width=True)

    df_display = st.checkbox("Evolución Precipitación EEUU", value=True)
    if df_display:
        fig = px.line(states_pp)
        fig.update_layout(title_text="Evolución de la Precipitación en los diferentes estados de EEUU")
        fig.update_yaxes(title_text="<b>Precipitación (mm)</b>", secondary_y=False)
        st.plotly_chart(fig, theme=None, use_container_width=True)
        

    df_display = st.checkbox("Evolución Temperatura EEUU", value=True)
    if df_display:
        fig = px.line(states_T)
        fig.update_layout(title_text="Evolución de la Temperatura en los diferentes estados de EEUU")
        fig.update_yaxes(title_text="<b>Temperature (ºC)</b>", secondary_y=False)
        st.plotly_chart(fig, theme=None, use_container_width=True)


with tab3:
    st.write('Este apartado tiene como objetivo mostrar la evolución de las superficie de cada uno de los cultivos y en cada uno de los estados a lo largo de los años en EEUU.')
    st.write('*Nota: Haciendo doble click en los diferentes nombres de las leyenda se quedaran representado únicamente el elemento seleccionado, también se pueden ir añadiendo o eliminando elementos de las leyendas para una mejor visualización de las gráficas.*')
    #VARIABLEs SUPERFICIEs CULTIVOS
    df_display = st.checkbox("Evolución de la superficie de cultivos", value=True)
    if df_display:
        fig = px.line(df_sup_cultivo, x = 'year', y = 'hectare', color = 'commodity_desc')
        fig.update_yaxes(title_text='<b>Superficie (ha)</b>')
        fig.update_layout(title_text='Evolución de la superficie diferenciado por cultivos')
        st.plotly_chart(fig, theme=None, use_container_width=True)

    df_display = st.checkbox("Evolución de la superficie de cultivos en cada estado", value=True)
    if df_display:
        fig = px.line(df_sup_estado, x = 'year', y = 'hectare', color = 'state_name')
        fig.update_yaxes(title_text='<b>Superficie (ha)</b>')
        fig.update_layout(title_text='Evolución de la superficie diferenciado por estados')
        st.plotly_chart(fig, theme=None, use_container_width=True)


with tab4:
    st.write('Esta sección muestra la evolución a lo largo de los años para cada estado en EEUU de las diferentes variables de estudio.')
    st.write('*Para esta visualización se debe hacer click en el botón "play" situado debajo de cada una de las figuras.*')

    #EVOLUCIÓN TEMPERATUTA Y PRECIPITACIÓN EN EEUU POR ESTADOS
    df_display = st.checkbox("Evolución temperatura EEUU", value=True)
    if df_display:
        fig = px.choropleth(df_T, locations="code", # used plotly express choropleth for animation plot
                            locationmode= "USA-states",
                            color="Temperature", 
                            color_continuous_scale='PuBu',
                            hover_name="State",
                            scope="usa",
                            hover_data=['Temperature'],
                            animation_frame =df_T.index,
                            labels={'Temperature':'The Temperature Change', 'Temperature':'Temperature'},
                            title = 'Temperature Change - 1950 - 2021')
        fig.update_layout(height=600)
        st.plotly_chart(fig, theme=None, use_container_width=True)


    #EVOLUCIÓN PRECIPITACIÓN EN EEUU POR ESTADOS
    df_display = st.checkbox("Evolución precipitación EEUU", value=True)
    if df_display:
        fig = px.choropleth(df_pp, locations="code", # used plotly express choropleth for animation plot
                            locationmode= "USA-states",
                            color="Precipitation", 
                            color_continuous_scale=px.colors.diverging.BrBG,
                            hover_name="State",
                            scope="usa",
                            hover_data=['Precipitation'],
                            animation_frame =df_pp.index,
                            labels={'Precipitation':'Precipitation', 'Precipitation':'Precipitation'},
                            title = 'Precipitation Change - 1950 - 2021')
        fig.update_layout(height=600)
        st.plotly_chart(fig, theme=None, use_container_width=True)

    #EVOLUCIÓN SUPERFICIE EN EEUU POR ESTADOS
    df_display = st.checkbox("Evolución superficie EEUU", value=True)
    if df_display:
        fig = px.choropleth(df_sup_estado, locations="code", 
                            locationmode= "USA-states",
                            color="hectare", 
                            color_continuous_scale=px.colors.sequential.algae,
                            hover_name="state_name",
                            scope="usa",
                            hover_data=['hectare'],
                            animation_frame =df_sup_estado.year,
                            labels={'hectare':'Hectáreas'},
                            title = 'Evolución de la superficie agrícola')
        fig.update_layout(height=600)
        st.plotly_chart(fig, theme=None, use_container_width=True)


#TABLA DE DATOS CON OPCIÓN A DESCARGA
with tab5:
    st.write('Esta sección muestra los datos que se han utilizado para realizar todo el estudio, haciendo click en el botón "CSV" es posible descargar las tablas de datos que sean de interés.')
   
    df_display = st.checkbox("Variables de estudio para todo EEUU", value=True)
    if df_display:
        #Opción de descarga de datos:
        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(df_agrupation)

        st.download_button(
            label="CSV",
            data=csv,
            file_name='Variables proyecto Evolución agrícola EEUU.csv',
            mime='text/csv',
        )
        st.write(df_agrupation)

    df_display = st.checkbox("Variable Precipitación para cada estado de EEUU", value=True)
    if df_display:
            #Opción de descarga de datos:
        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(states_pp)

        st.download_button(
            label="CSV",
            data=csv,
            file_name='Precipitación_estados_EEUU.csv',
            mime='text/csv',
        )
        st.write(states_pp)

    df_display = st.checkbox("Variable Temperatura para cada estado de EEUU", value=True)
    if df_display:
        #Opción de descarga de datos:
        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')

        csv = convert_df(states_pp)

        st.download_button(
            label="CSV",
            data=csv,
            file_name='Temperatura_estados_EEUU.csv',
            mime='text/csv',
        )    
        st.write(states_T)
