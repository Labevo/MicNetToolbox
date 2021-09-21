"""
SUBNETOWRKS AND COMMUNITIES
@author: Natalia Favila
"""

from typing import Optional,Union,Dict
import community
import networkx as nx
import pandas as pd
import numpy as np
from .utils import _build_network,_normalize_corr

def community_analysis(corr:Union[pd.DataFrame,np.ndarray],
                       taxa:Optional[pd.DataFrame]=None)->Dict[str,pd.DataFrame]:
    '''
    Function that performs community analysis and returns a description of each
    community subnetwork.
    
    Parameters
    ----------
    corr : interaction matrix as a pandas dataframe or numpy matrix of 
           dimension m x m. 
    taxa: dataframe with ASV and/or taxa of dimension m x n.

    Returns
    -------
    num_com = number of different communities found
    df = Community with taxa id
    com_dict = dictionary with a dataframe for each community found. 
               Each dataframe contains the 'Nodes', 'Diameter',
               'Clustering coefficient', and 'Average shortest path length'.

    '''
    
    ## check that correlation matrix is square
    if corr.shape[0] != corr.shape[1]:
        raise ValueError('''The correaltion matrix or data frame input is not square. \
            Dimensions should be m x m.''')

    corr = _normalize_corr(corr)
    G = _build_network(corr)
    com = community.best_partition(G)
    


    ## Check taxa and corr match
    if taxa:
        if corr.shape[0] != taxa.shape[0]:
            raise ValueError('''The correaltion and the taxa dataframes do not match. \
                If correlation matrix is of size m x m, then taxa dataframe should be of size m x n''')
        else:
            taxa['Community_id'] = com.values()
    else:
        taxa=pd.DataFrame()
        taxa['Community_id'] = com.values()


        
    n_com = len(set(com.values()))
    data = []
    #Subnetwork analysis
    for com_id in range(0,n_com):
        subnet = [key  for (key, value) in com.items() if value == com_id]
        Gc=nx.subgraph(G,subnet)
        data.append([Gc.number_of_nodes(),nx.diameter(Gc),nx.average_clustering(Gc),nx.average_shortest_path_length(Gc)])
    
    #transpose data
    datat =[list(i) for i in zip(*data)]
    
    com_df = pd.DataFrame(
        datat, 
        index = ['Nodes', 'Diameter','Clustering coef', 'Average shortest path'],
        columns = [f'Community_{i}' for i in range(0,n_com)]
        )

    data_dict = {
        'Number of communities':n_com,
        'Community_data': taxa,
        'Communities_topology': com_df,
        }
    
    return data_dict

def HDBSCAN_subnetwork(corr:Union[np.ndarray,pd.DataFrame], 
                       HDBSCAN:pd.DataFrame)->pd.DataFrame:
    '''  
    Function that performs an analysis of the clusters found by HDBSCAN and 
    returns a description of each cluster subnetwork characteristics.
    
    Parameters
    ----------
    corr : interaction matrix as a pandas dataframe or numpy matrix of 
            dimension m x m. 
    
    HDBSCAN: pandas dataframe of dimensions m x 1 containing the clusted id for each cluster 
             found, where outliers are identified with value -1. 

    Returns
    -------
    hdbscan_df = dataframe in which each column represents a different HDBSCAN cluster.
                 For each cluster we obtain : 'Nodes', 'Edges','Density', 'Average degree',
                 'degree std' and 'Clustering coefficient'.
                 

    '''
    ## check that correlation matrix is square
    if corr.shape[0] != corr.shape[1]:
        raise ValueError('''The correaltion matrix or data frame input is not square. Dimensions should be m x m.''')     
                            
    ## Check taxa and corr match
    if corr.shape[0] != HDBSCAN.shape[0]:
        raise ValueError('''The correaltion and the taxa dataframes do not match. If correlation matrix is of size m x m, then taxa dataframe should be of size m x n''')

    
    groups = list(HDBSCAN.iloc[:,0])
    groups_id = set(groups)
    
    #Dont include ouliers as a group
    if -1 in groups_id:    
        groups_id.remove(-1)
        
    ng = len(groups_id)
    
    #Graph with all nodes
    corrn = _normalize_corr(corr)
    G =  _build_network(corrn)
    
    group_data =[]
    
    #create subnetwork and get their metrics
    for gid in groups_id:
        subnet = [i for i, x in enumerate(groups) if x == gid]
        Gc=nx.subgraph(G,subnet)
        group_data.append([Gc.number_of_nodes(),
                           Gc.number_of_edges(),
                           nx.density(Gc),
                           np.mean([Gc.degree(n) for n in Gc.nodes()]),
                           np.std([Gc.degree(n) for n in Gc.nodes()]),
                           nx.average_clustering(Gc)])
    #Transpose data
    datat =[list(i) for i in zip(*group_data)]
    
    #Save all in one dataFrame
    hdbscan_df = pd.DataFrame(
        datat, 
        index = ['Nodes', 'Edges','Density', 'Average degree','degree std', 'Clustering coefficient'],
        columns = [f'Cluster_{i}' for i in range(0,ng)]
        )
    
    return hdbscan_df