#--BASE--#
import pandas as pd
import seaborn as sns
import numpy as np
import os
import warnings
import pydot
import graphviz 
import minepy
import plotly.graph_objects as go

#--VISUALIZACIÓN--#
import matplotlib as mpl
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from statsmodels.graphics.factorplots import interaction_plot
import graphviz 
import shap
import pydot

#--MODELOS--#
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn import tree
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, classification_report
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
import xgboost as xgb
from scipy.stats import pearsonr
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.tools.eval_measures import mse, meanabs, rmse, stde
from scipy import stats


def crosscorr(datax: pd.Series, datay: pd.Series, lag: int = 0, wrap: bool = False) -> float:
    ''' 
    Lag-N cross correlation. 
    Shifted data filled with NaNs 
    
    Parameters
    ----------
    lag : int, default 0
        Default lag between series given
        
    datax, datay : pandas.Series 
        Objects of equal length. Series we want to lag.
    
    Returns
    ----------
    crosscorr : float
        The correlation between given series
    '''
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))


from typing import Tuple, Optional, List, Dict, Callable
def corr_lag(serie1 : str,serie2: str,df: pd.DataFrame ,start_time: str = None, window_size: int = 200) -> Tuple[pd.DataFrame,int,int]:
    '''
    Function that computes cross lagged correlation between two series form a pandas.DataFrame
    
    Parameters
    ----------
    serie1 : str
        Name of the first series in the dataframe df
        
    serie2 : str
        Name of the second series in the dataframe df
        
    df : pandas.DataFrame
        The DataFrame that contains the series we want to calculate lagged correlation with.
        
    start_time: str
        Date where we want to start computing lagged correlation with the format dd-mm-yyyy. Default None which means the whole dataframe
    
    window_size: int
        Number of laggs computed. Default 200.

    Returns
    -------
    rs: pd.DataFrame
        DataFrame containing correlation list between both series and their lags. For the first window_size elements, we have the first series lagged and the second one not lagged. For the next Window_size elments, the second one is lagged.
        
    offset: int
        Point where both series are not lagged 
        
    Peak: int
        Possition of the highest correlation peak within rs.
    '''
    if start_time:
        df = df[df.index > start_time]
    d1 = df[serie1]
    d2 = df[serie2]
    rs = pd.DataFrame([crosscorr(d1,d2, lag) for lag in range(-int(window_size),int(window_size+1))])
    offset = np.floor(len(rs)/2)-np.argmax(rs)
    peak = np.argmax(abs(rs))
    return rs, offset, peak


def nonlinear(x: pd.Series,y: pd.Series) -> int:
    """
    Computes MIC between two time series passed with default arguments.
    
    Parameters
    ----------
    x : pd.Series
        Time series to compute MIC with y.
        
    y : pd.Series
        Time series to compute MIC with x.
    
    Returns
    -------
    mine.mic() : int
        MIC between x and y
    """
    mine = minepy.MINE(alpha=0.6, c=15) 

    mine.compute_score(x,y) 
    return mine.mic()

def mic_variable (X: pd.DataFrame,y:str):
    """
    Computes MIC between one time serie passed with default arguments and all the time series in a dataframe passed with defautl arguments .
    
    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with the Time series to compute MIC with y.
        
    y : pd.Series
        Time series to compute MIC with x.
    
    Returns
    -------
    df_Mic : DataFrame 
        MIC between the time series in X and y.
    """
    Mic=[]
    
    for i in range(np.size(X,1)):
        aux=nonlinear(X[y],X.iloc[:,i])
        Mic.append(aux)
    df_Mic=pd.DataFrame({'Mic':Mic},index=X.columns)
    return(df_Mic)

            
def matrix_mic(X:pd.DataFrame):
    """
    Computes MIC between all the Time Series in a dataFrame passed with default arguments .
    
    Parameters
    ----------
    X : pd.DataFrame
        Dataframe with the Time series to compute MIC.
    
    Returns
    -------
    m_mic: DataFrame 
        MIC between the time series in X.
    """
    Matrix_mic=pd.DataFrame(index=X.columns)
    
    for i in range(np.size(X,1)):
        Matrix_mic[X.iloc[:,i].name]=mic_variable(X=X,y=X.iloc[:,i].name)['Mic']
    
    return(Matrix_mic) 



def Decision_Tree (df1, df2, max_leaf_nodes = 10, image_name = 'Decision Tree'):
    '''
    Function that presents a plot with the importance of the variables through a decision tree with a maximum of 15 nodes.

    Parameters
    ----------
    df1: Independent variables
    df2: dependent variables
    max_leaft_nodes = nudes number, predefined in 10 nodes
    name: image name, predefined in 'Decision Tree'

    Returns
    -------
    A Decisión Tree
    '''
    Dt_model = tree.DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes)

    Dt_model.fit(df1, df2)

    os.environ['PATH'] += os.pathsep + '../Graphviz/bin/'
    dot_data = tree.export_graphviz(Dt_model, out_file=None,
                                feature_names=df1.columns.values,
                                filled=True, rounded=True,
                                special_characters=True, leaves_parallel = False)  
    graph = graphviz.Source(dot_data)

    (graph, ) = pydot.graph_from_dot_data(dot_data)
    graph.write_png('../NOTEBOOKS/images/Decision_Tree/'+ str(image_name) + ".png")



def Random_Forest(df1, df2, title, n_estimators=500, random_state = 1000,  min_samples_leaf= 5, image_name = 'Random Forest'):
    '''
    Plot of the representation of the importance of the independent variables compared to the dependent variable
    
    Parameters
    ----------
    df1: Independent variables
    df2: dependent variables
    title: title given to the figure
    name: image name, predefined in 'Random Forest'

    Return
    ------
    A plot wiht the importance score for every variable of study
    '''
    BA_model = RandomForestRegressor(n_estimators=n_estimators, random_state = random_state,  min_samples_leaf = min_samples_leaf)
    BA_model.fit(df1, df2)
    feature_scores = pd.Series(BA_model.feature_importances_, index = df1.columns).sort_values(ascending = False)

    fig, ax = plt.subplots(figsize = (5, 2))
    ax = sns.barplot(x=feature_scores, y=feature_scores.index)
    ax.set_title(title, fontsize=10)
    ax.set_yticklabels(feature_scores.index, fontsize=10)
    ax.set_xlabel("Feature importance score", fontsize=10)
    ax.set_ylabel("Features", fontsize=10)
    plt.grid(axis='x',linestyle='dotted', color='b')
    plt.savefig('../NOTEBOOKS/images/Random_Forest/' + str(image_name) + ".png")
    plt.close(fig)



def Linear_Regression(X_train, X_test, y_train, y_test):
    '''
    Linear Regression Model to study the selections of variables 

    Parameters
    ----------
    X_train: independent variables train
    X_test: independent variables test
    y_train: dependent variable train
    y_test: dependent variable test

    Return
    ---------
    MAE: Test Mean absolute error
    MSE: Test mean squared error
    R^2
    Graphic representation comparing the dependent variables Test and Prediction
    Graphis residue studio
    '''
    #Se aplica el modelo
    modelo = LinearRegression()
    modelo.fit(X_train,y_train)
    y_pred = modelo.predict(X_test)

    #Resultados
    print('(MAE) Test Mean absolute error:',mean_absolute_error(y_test,y_pred).round(2))
    print('(MSE) Test mean squared error:', np.sqrt(mean_absolute_error(y_test,y_pred)).round(2))
    print('Test R2 score:',r2_score(y_test, y_pred).round(2))

    #Representación grafica de los valores predecidos y los valores de Test
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(2004, 2022)), y = y_pred, mode='lines', name = 'Predicción'))
    fig.add_trace(go.Scatter(x = list(range(2004, 2022)), y = y_test, mode='lines', name = 'Test'))
    fig.update_layout(title='Comparativa Test y Predicción', yaxis_title='Superficie (ha)')
    fig.show()

    #Diagnostico de los residuos
    # Diagnóstico errores (residuos) de las predicciones de entrenamiento
    prediccion_train = modelo.predict(X_train)
    residuos_train   = prediccion_train - y_train

    # Gráficos
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],'k--', color = 'black', lw=2)
    axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 7)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[0, 1].set_xlabel('id')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)

    sns.histplot(data = residuos_train, stat = "density", kde = True, line_kws= {'linewidth': 1}, color = "firebrick", alpha = 0.3, ax = axes[1, 0])

    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)


    sm.qqplot(residuos_train, fit = True, line = 'q', ax = axes[1, 1], color = 'firebrick', alpha = 0.4, lw = 2)
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)

    axes[2, 0].scatter(prediccion_train, residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
    axes[2, 0].set_xlabel('Predicción')
    axes[2, 0].set_ylabel('Residuo')
    axes[2, 0].tick_params(labelsize = 7)

    # Se eliminan los axes vacíos
    fig.delaxes(axes[2,1])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold")


def K_Nearest_Neighbour_Regressor(X_train, X_test, y_train, y_test):

    '''
    K Nearest Neighbour Regressor Model to study the selections of variables 

    Parameters
    ----------
    X_train: independent variables train
    X_test: independent variables test
    y_train: dependent variable train
    y_test: dependent variable test

    Return
    ---------
    MAE: Test Mean absolute error
    MSE: Test mean squared error
    R^2
    Graphic representation comparing the dependent variables Test and Prediction
    Graphis residue studio
    '''

    #Se aplica el modelo
    modelo = KNeighborsRegressor(n_neighbors=2, weights = 'uniform')
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    #Resultados
    print('(MAE) Test Mean absolute error:',mean_absolute_error(y_test,y_pred).round(2))
    print('(MSE) Test mean squared error:', np.sqrt(mean_absolute_error(y_test,y_pred)).round(2))
    print('Test R2 score:',r2_score(y_test,y_pred).round(2))

    #Representación grafica de los valores predecidos y los valores de Test
    fig = go.Figure()
    fig.add_trace(go.Scatter(x = list(range(2004, 2022)), y = y_pred, mode='lines', name = 'Predicción'))
    fig.add_trace(go.Scatter(x = list(range(2004, 2022)), y = y_test, mode='lines', name = 'Test'))
    fig.update_layout(title='Comparativa Test y Predicción', yaxis_title='Superficie (ha)')
    fig.show()

    #Diagnostico de los residuos
    # Diagnóstico errores (residuos) de las predicciones de entrenamiento
    prediccion_train = modelo.predict(X_train)
    residuos_train   = prediccion_train - y_train

    # Gráficos
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],'k--', color = 'black', lw=2)
    axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 7)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[0, 1].set_xlabel('id')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)

    sns.histplot(data = residuos_train, stat = "density", kde = True, line_kws= {'linewidth': 1}, color = "firebrick", alpha = 0.3, ax = axes[1, 0])

    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)

    sm.qqplot(residuos_train, fit = True, line = 'q', ax = axes[1, 1], color = 'firebrick', alpha = 0.4, lw = 2)
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)

    axes[2, 0].scatter(prediccion_train, residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
    axes[2, 0].set_xlabel('Predicción')
    axes[2, 0].set_ylabel('Residuo')
    axes[2, 0].tick_params(labelsize = 7)

    # Se eliminan los axes vacíos
    fig.delaxes(axes[2,1])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold")


def Random_Forest_Tuning(X_train, y_train):
    '''
    Function to obtain the best parameters from the study variables to be able to apply them in the Random Forest regression prediction model

    Parameters
    ----------
    X_train: independent variables train
    X_test: independent variables test
    gamma, max_depth max_features, min_child_weight, n_estimators: variables defined in the Random Forest Tuning function

    Return
    ---------
    variables defined in the Random Forest Tuning function:
    n_estimators
    max_features
    max_depth    
    criterion    
    '''
    #Ajuste del modelo
    random_forest_tuning = RandomForestRegressor()
    param_grid = {
    'n_estimators': [1, 3, 5, 10, 15, 20, 50, 100, 300],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [1,2,3,4,5,6,7,8],
    'criterion' :['squared_error', 'absolute_error', 'friedman_mse', 'poisson']}
    
    GSCV = GridSearchCV(estimator=random_forest_tuning, param_grid=param_grid, cv=5)
    GSCV.fit(X_train, y_train)
    print('Random Forest Parameters:',GSCV.best_params_)


def Random_Forest_regressor(X_train, X_test, y_train, y_test, criterion:str, max_depth:int, max_features: str, n_estimators:int):
    '''
    Random Forest Regressor Model to study the selections of variables 

    Parameters
    ----------
    X_train: independent variables train
    X_test: independent variables test
    y_train: dependent variable train
    y_test: dependent variable test
    criterion, max_depth max_features, n_estimators: variables defined in the Random Forest Tuning function

    Return
    ---------
    MAE: Test Mean absolute error
    MSE: Test mean squared error
    R^2
    Graphic representation comparing the dependent variables Test and Prediction
    Graphis residue studio
    '''
    #Se aplica el modelo
    modelo = RandomForestRegressor(oob_score=True)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    #Resultados
    print('(MAE) Test Mean absolute error:',mean_absolute_error(y_test,y_pred).round(3))
    print('(MSE) Test mean squared error:', np.sqrt(mean_absolute_error(y_test,y_pred)).round(3))
    print('Test R2 score:',r2_score(y_test,y_pred).round(3))

    #Representación grafica de los valores predecidos y los valores de Test
    fig = go.Figure()
    fig.add_trace(go.Scatter(y = y_pred, mode='lines', name = 'Predicción'))
    fig.add_trace(go.Scatter(y = y_test, mode='lines', name = 'Test'))
    fig.update_layout(title='Comparativa Test y Predicción', yaxis_title='Superficie (ha)')
    fig.show()

    #Diagnostico de los residuos
    # Diagnóstico errores (residuos) de las predicciones de entrenamiento
    prediccion_train = modelo.predict(X_train)
    residuos_train   = prediccion_train - y_train

    # Gráficos
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],'k--', color = 'black', lw=2)
    axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 7)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[0, 1].set_xlabel('id')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)

    sns.histplot(data = residuos_train, stat = "density", kde = True, line_kws= {'linewidth': 1}, color = "firebrick", alpha = 0.3, ax = axes[1, 0])

    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)

    sm.qqplot(residuos_train, fit = True, line = 'q', ax = axes[1, 1], color = 'firebrick', alpha = 0.4, lw = 2)
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)

    axes[2, 0].scatter(prediccion_train, residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
    axes[2, 0].set_xlabel('Predicción')
    axes[2, 0].set_ylabel('Residuo')
    axes[2, 0].tick_params(labelsize = 7)

    # Se eliminan los axes vacíos
    fig.delaxes(axes[2,1])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold")


def XGBoost_Tuning(X_train, y_train):
    '''
    Function to obtain the best parameters from the study variables to be able to apply them in the XGBoost regression prediction model

    Parameters
    ----------
    X_train: independent variables train
    X_test: independent variables test
    gamma, max_depth max_features, min_child_weight, n_estimators: variables defined in the Random Forest Tuning function

    Return
    ---------
    variables defined in the Random Forest Tuning function:
    n_estimators
    min_child_weight
    gamma
    max_depth   
    '''
    XGBoost_tuning = XGBRegressor()
    param_grid ={
    'n_estimators': [1, 3, 5, 10, 15, 20, 50, 100, 300],
    'min_child_weight': [1, 2, 3, 4, 5, 6], 
    'gamma': [0.00001, 0.0001, 0.001, 0.01, 1, 1.5, 2, 3, 5],
    'max_depth': [1,2,3,4,5,6,7,8]}
   
    GSCV = GridSearchCV(estimator=XGBoost_tuning, param_grid=param_grid, cv=5)
    GSCV.fit(X_train, y_train)
    print('XGBoost parameters:',GSCV.best_params_)


def XGB_Regressor(X_train, X_test, y_train, y_test, gamma: float, max_depth: int, min_child_weight: int, n_estimators: int):
    '''
    XGBoost Regressor Model to study the selections of variables 

    Parameters
    ----------
    X_train: independent variables train
    X_test: independent variables test
    y_train: dependent variable train
    y_test: dependent variable test
    gamma, max_depth max_features, min_child_weight, n_estimators: variables defined in the Random Forest Tuning function

    Return
    ---------
    MAE: Test Mean absolute error
    MSE: Test mean squared error
    R^2
    Graphic representation comparing the dependent variables Test and Prediction
    Graphis residue studio
    '''
    #Se aplica el modelo
    modelo = XGBRegressor()
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    #Resultados
    print('(MAE) Test Mean absolute error:',mean_absolute_error(y_test,y_pred).round(3))
    print('(MSE) Test mean squared error:', np.sqrt(mean_absolute_error(y_test,y_pred)).round(3))
    print('Test R2 score:',r2_score(y_test,y_pred).round(3))


    #Representación grafica de los valores predecidos y los valores de Test
    fig = go.Figure()
    fig.add_trace(go.Scatter(y = y_pred, mode='lines', name = 'Predicción'))
    fig.add_trace(go.Scatter(y = y_test, mode='lines', name = 'Test'))
    fig.update_layout(title='Comparativa Test y Predicción', yaxis_title='Superficie (ha)')
    fig.show()

    #Diagnostico de los residuos
    # Diagnóstico errores (residuos) de las predicciones de entrenamiento
    prediccion_train = modelo.predict(X_train)
    residuos_train   = prediccion_train - y_train

    # Gráficos
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 8))

    axes[0, 0].scatter(y_train, prediccion_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 0].plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()],'k--', color = 'black', lw=2)
    axes[0, 0].set_title('Valor predicho vs valor real', fontsize = 10, fontweight = "bold")
    axes[0, 0].set_xlabel('Real')
    axes[0, 0].set_ylabel('Predicción')
    axes[0, 0].tick_params(labelsize = 7)

    axes[0, 1].scatter(list(range(len(y_train))), residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[0, 1].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[0, 1].set_title('Residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[0, 1].set_xlabel('id')
    axes[0, 1].set_ylabel('Residuo')
    axes[0, 1].tick_params(labelsize = 7)

    sns.histplot(data = residuos_train, stat = "density", kde = True, line_kws= {'linewidth': 1}, color = "firebrick", alpha = 0.3, ax = axes[1, 0])

    axes[1, 0].set_title('Distribución residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 0].set_xlabel("Residuo")
    axes[1, 0].tick_params(labelsize = 7)

    sm.qqplot(residuos_train, fit = True, line = 'q', ax = axes[1, 1], color = 'firebrick', alpha = 0.4, lw = 2)
    axes[1, 1].set_title('Q-Q residuos del modelo', fontsize = 10, fontweight = "bold")
    axes[1, 1].tick_params(labelsize = 7)

    axes[2, 0].scatter(prediccion_train, residuos_train, edgecolors=(0, 0, 0), alpha = 0.4)
    axes[2, 0].axhline(y = 0, linestyle = '--', color = 'black', lw=2)
    axes[2, 0].set_title('Residuos del modelo vs predicción', fontsize = 10, fontweight = "bold")
    axes[2, 0].set_xlabel('Predicción')
    axes[2, 0].set_ylabel('Residuo')
    axes[2, 0].tick_params(labelsize = 7)

    # Se eliminan los axes vacíos
    fig.delaxes(axes[2,1])

    fig.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.suptitle('Diagnóstico residuos', fontsize = 12, fontweight = "bold")