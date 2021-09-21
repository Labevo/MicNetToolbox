# -*- coding: utf-8 -*-
"""
Auxiliar funtions
@author: Natalia Favila
"""
from typing import Union
import networkx as nx
import numpy as np
import pandas as pd


def _normalize_corr(corr:Union[np.ndarray,pd.DataFrame]):
    '''
    Function that takes the interaction matrix from Sparcc and normalizes its
    values to range from 0 to 1. Any value that was 0 before is left as 0.
    
    Parameters
    ----------
    corr : correlation matrix read as a pandas DataFrame

    Returns
    -------
    corr : pandas DataFrame with normalized correlations
    '''

    if type(corr)==pd.DataFrame:
        corr=corr.values


    max_val = corr.max().max()
    min_val = corr.min().min()
    corrnorm = (corr - min_val) / (max_val - min_val)
    corrnorm = np.round(corrnorm, 3)
    # leave 0s as zeros
    corrnorm=np.where(corr==0,0,corrnorm)
    
    return corrnorm


def _build_network(corr:Union[np.ndarray,pd.DataFrame]):
    '''
    Function that takes the interaction matrix (as pandas Dataframe or numpy matrix) from Sparcc and returns the 
    corresponding networkx graph
    
    Parameters
    ----------
    corr : correlation matrix read as a pandas DataFrame or numpy matrix

    Returns
    -------
    G : netowrkx graph
    '''
    if type(corr)==pd.DataFrame:
        corr=corr.values


    #corr = np.matrix(corr)
    G = nx.from_numpy_matrix(corr)
    return G

def create_normalize_graph(corr:Union[np.ndarray,pd.DataFrame])->nx.Graph:

    corrnorm=_normalize_corr(corr)
    graph=_build_network(corrnorm)
    return graph

def filter_by_pvalues(raw_corr:pd.DataFrame, pvals:pd.DataFrame, p:int = 0.05):
    
    '''
    Function that filters the raw correlations from Sparcc using the pvalues
    obtained from the Monte Carlo simulation.
    
    Parameters
    ----------
    raw_corr : raw corelation obtain from SparCC read as a pandas DataFrame
    pvals : pvals obtained from Sparcc. Read as pandas DataFrame, has to be 
            the same size as raw_corr.
    p : the p-value to filter

    Returns
    -------
    corr : filtered correaltions by p
    '''
    
    #Filtered correlations
    raw_corr[pvals >= p] = 0
    
    return raw_corr


