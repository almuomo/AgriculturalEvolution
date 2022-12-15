
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Callable


def crosscorr(datax: pd.Series, datay: pd.Series, lag: int = 0, wrap: bool = False) -> float:
    """ Lag-N cross correlation. 
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
    """
    if wrap:
        shiftedy = datay.shift(lag)
        shiftedy.iloc[:lag] = datay.iloc[-lag:].values
        return datax.corr(shiftedy)
    else: 
        return datax.corr(datay.shift(lag))


from typing import Tuple, Optional, List, Dict, Callable
def corr_lag(serie1 : str,serie2: str,df: pd.DataFrame ,start_time: str = None, window_size: int = 200) -> Tuple[pd.DataFrame,int,int]:
    """
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
    """
    if start_time:
        df = df[df.index > start_time]
    d1 = df[serie1]
    d2 = df[serie2]
    rs = pd.DataFrame([crosscorr(d1,d2, lag) for lag in range(-int(window_size),int(window_size+1))])
    offset = np.floor(len(rs)/2)-np.argmax(rs)
    peak = np.argmax(abs(rs))
    return rs, offset, peak