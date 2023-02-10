## EVOLUCIÓN AGRÍCOLA EN ESTADOS UNIDOS
![Agricultura image](./NOTEBOOKS/images/tecnologia_agricultura.jpg)
<sub>Fuente: Liste Maquinaria</sub>

Este proyecto ha sido realizado como trabajo final del Máster de *Data Science* impartido por la escuela digital *KSchool*

## Propósito del proyecto
Estados Unidos es uno de los principales productores agrícolas a nivel global, especialmente en productos como la soja, maíz y trigo. Esto es posible gracias a las vastas extensiones de superficies agrícolas que pueden llegar a encontrarse y también a la diversidad de climas que están presentes en este país. 

A lo largo de la historia estas superficies agrícolas se han ido modificando. Actualmente el crecimiento de estas superficies no es una opción viable pero sí que lo es la modificación y adaptación de los cultivos influenciados por los nuevos factores a los que el sector agrario se enfrenta a día de hoy.

Algunos ejemplos de estos factores pueden ser: el cambio climático, introducción de nuevas tecnologías en la agricultura o las políticas que se han ido desarrollando a lo largo de los años.

Este trabajo está orientado al estudio de los factores climáticos y cuál ha sido la influencia en los cultivos agrícolas de Estados Unidos.

Una vez realizado el estudio, se ha visto que la evolución de las temperaturas a lo largo de las décadas sí que ha tenido influencia en la modificación y evolución de las superficies de los principales cultivos de EEUU mientras que la precipitación no ha sido una variable tan influyente, ya que esta puede ser sustituida por técnicas de riego.


### Enlace para descargar datos para este trabajo:
https://drive.google.com/drive/folders/1q5XLojqI__ObQZ4a8lkxqT_3HwAyReSN?usp=share_link

Los notebooks en los cuales se distribullen el trabajo son:

- 1_WebScraping
- 2_Transformación_datos_Temp_y_PP
- 3_Transformación_datos_Sup_de_cultivo
- 4.1_Análisis_PP
- 4.2_Análisis_T
- 4.3_Anáisis_superficies
- 4.4_Análisis_multivariante
- 5_modelo
- 6_streamlit 

Cada notebook es independiente uno del otro y no es necesario ejecutar el notebook anterior para pasar al siguiente.
Los tres primeros notebook junto con el archivo de Rstudio, hacen referencia a la descarga y transformación de datos, todos ellos recogidos en el archivo DATOS, descargados con el enlace anterior
Los notebooks con punto 4 hace referencia al análisis de las variables y los notebooks 5 y 6 como su propio nombre indican es donde se encuentra el estudio de modelos y aplicación web.

Nota: Para ejecutar el codigo sin problemas el archivo "DATOS" descargado se debe descomprimir y localizar en la carpeta AgriculturalEvolution junto con los archivos de NOTEBOOK y Graphviz.

### Librerias y elementos instalados para la lectura de los notebook:
- pandas
- matplotlib
- plotly
- seaborn
- bs4
- selenium
- nbformat
- openpyxl
- statsmodels
- phik
- pingouin
- pydot
- graphviz
- shap
- ipywidgets
- xgboost
- opencv-python
- -U scikit-learn
- fpdf
- writefile
- streamlit
- altair
- minepy: Antes de instalar esta libreria es necesario tener instalado "Microsoft C++ Build Tools", en caso de no tenerlo instalado para visual studio se puede descargar en:  https://visualstudio.microsoft.com/es/visual-cpp-build-tools/ 
          Este enlace explica como realizar la instalación del paquete de C++ necesario: https://youtu.be/CwT490K4TAo
          Una vez realizado este proceso se puede proceder a instalar minepy: pip install minepy --use-pep517
- chromedriver: este driver se utiliza para correr el notebook 1_WebScraping, se encuentra localizado en "../DATOS/chromedriver", en el caso de que el codigo de error revisar que tanto el chromedriver como el buscador, en este caso chrome, tienen la misma versión en caso contrario, descargar el chromedriver igual a la versión de google chrome. Enlace a la pagina web: https://chromedriver.chromium.org/downloads 

#### Autor: Alejandro Muñoz Molina
#### LinkedIn: https://www.linkedin.com/in/alex245/     