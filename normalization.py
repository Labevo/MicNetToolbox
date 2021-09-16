'''
functions to normalize data
@dlegor

'''
import numpy as np
from numpy.random.mtrand import dirichlet
from typing import Union
import pandas as pd
from skbio.stats.composition import clr

__all__=['clr','normalize','to_fractions']

def normalize(frame:Union[np.ndarray,pd.DataFrame], axis:int=1):
    '''
    Normalize counts by sample total.
    
    Parameters
    ----------
    axis : {0, 1}
        0 : normalize each column
        1 : normalize each row

    Returns new instance of same class as input frame.
    '''    
    #To do for axis=0
    if axis==0:
        return frame/frame.sum(axis=0,keepdims=True)
    else:
        #to do for the axis=1
        return frame/frame.sum(axis=1,keepdims=True)
    
def to_fractions(frame:Union[np.ndarray,pd.DataFrame], method:str='dirichlet',
                 p_counts:int=1, axis:int=1):
    '''
    Covert counts to fraction using given method.
    
    Parameters
    ----------
    method : string {'dirichlet' (default) | 'normalize' | 'pseudo'}
        dirichlet - randomly draw from the corresponding posterior 
                    Dirichlet distribution with a uniform prior.
                    That is, for a vector of counts C, 
                    draw the fractions from Dirichlet(C+1). 
        normalize - simply divide each row by its sum.
        pseudo    - add given pseudo count (defualt 1) to each count and
                    do simple normalization.
    p_counts : int/float (default 1)
        The value of the pseudo counts to add to all counts.
        Used only if method is dirichlet
    axis : {0 | 1}
        0 : normalize each column.
        1 : normalize each row.
    
    Returns
    -------
    fracs: frame/array
        Estimated component fractions.
        Returns new instance of same class as input frame.
    '''
    #Validation
    if isinstance(frame,pd.DataFrame):
        frame=frame.values

    #Define Dirichlet Funtion
    def dirichlet_fun(x):
        a = x+int(p_counts)
        return  dirichlet(a)

    #normalize case
    if method == 'normalize':
        fracs = normalize(frame, axis)
        return fracs

    #Dirichlet Case    
    elif method =='dirichlet':
         fracs = np.apply_along_axis(dirichlet_fun, axis, frame)
         return fracs 
    else:
        raise ValueError('Unsupported method "%s"' %method)