from typing import Union
import numpy as np
import pandas as pd

def single_node_relationships(corr:Union[np.ndarray,pd.DataFrame],
                              taxa:Union[pd.Series,pd.DataFrame],
                              node_index:int)->pd.DataFrame:
        
    
    '''
     Function that returns all nodes related to a desired node (node_index).
     
    Parameters
    ----------
    corr : correlation matrix read as a pandas DataFrame or numpy matrix
    taxa : pandas dataframe with the related nodes  ASV and/or taxa
    n: index of the node which individual relationships want to be extracted
       
    Returns
    -------
    data: dataframe with the nodes that are correalted with the desired node,
          including the index of the nodes, the correlations with the desired node,
          as well as the taxa of the correlated nodes.    
        
    '''
    ## check that correlation matrix is square
    if corr.shape[0] != corr.shape[1]:
        raise ValueError('''The correaltion matrix or data frame input is not square. Dimensions should be m x m.''')     
                            
    ## Check taxa and corr match
    if corr.shape[0] != taxa.shape[0]:
        raise ValueError('''The correaltion and the taxa dataframes do not match. If correlation matrix is of size m x m, then taxa dataframe should be of size m x n''')
      
    
    corr = pd.DataFrame(corr)
    taxa = pd.DataFrame(taxa)
    #correlations of the desired index
    noder = corr.iloc[node_index,:]
    #Keep only correaltions different to 0
    nodein = list(noder.index[noder != 0])
    test_in = [int(i) for i in nodein]
    
    
    nodetaxa = taxa.loc[test_in,:]
    nodetaxa = nodetaxa.reset_index(drop = True)
    noderel = noder.iloc[test_in]
    noderel=noderel.reset_index(drop = True)
    
    data = pd.DataFrame()
        
    data['Node'] = nodein
    data['Correlation'] = noderel
    data = data.join(nodetaxa)

    return data    
 
def taxa_relationships(corr:Union[np.ndarray,pd.DataFrame],
                       taxa:Union[pd.Series,pd.DataFrame])->pd.DataFrame:
    '''
    Function that returns the mean correlation and number of positive and negative
    correlations between taxa at the lowest level specified. 
    This should be run with taxa information, not ASV information.
     
    Parameters
    ----------
    corr : correlation matrix read as a pandas DataFrame or numpy matrix of 
           size m x m. 
    taxa : pandas dataframe with the related nodes taxa. Taxa dataframe should
           be of size m x 1. Thus, one column with the taxa of the noeds.
           
      
    Returns
    -------
    data: dictionary where keys are the taxa and associated there is a 
          dataframe with all posible taxa and the mean correlation, number of 
          positive, negative and null correlations to they key taxa.
    '''
    ## check that correlation matrix is square
    if corr.shape[0] != corr.shape[1]:
        raise ValueError('''The correaltion matrix or data frame input is not square. Dimensions should be m x m.''')     
                            
    ## Check taxa and corr match
    if corr.shape[0] != taxa.shape[0]:
        raise ValueError('''The correaltion and the taxa dataframes do not match. If correlation matrix is of size m x m, then taxa dataframe should be of size m x n''')
      
    
    corr = pd.DataFrame(corr)
    taxa = pd.DataFrame(taxa)
    #All taxa without repetion
    species = list(set(taxa.iloc[:,0]))
    
    ids_dict = {}
    for sp in species:
        sp_ids = list(taxa.index[taxa.iloc[:,0] == sp])
        ids_dict[sp] = sp_ids
    
    data = {}
    for sp in species:
        #ids of species sp1
        sp1 = ids_dict[sp]
        df = pd.DataFrame(columns=('Taxa', 'Mean correlation',
                                          'Positive corrs', 'Negative corrs',
                                          'Null corrs'))
        spd = sp
        i = 0
        for sp in species:
            #ids of species sp2
            sp2 = ids_dict[sp]
            corrs = np.array(corr.iloc[sp1, sp2]).flatten()
            meancorr = np.mean(np.array(corrs).flatten())
            poscorr = sum(corrs>0)
            negcorr = sum(corrs<0)
            nullcorr = sum(corrs==0)
            df.loc[i]= [sp,  meancorr, poscorr, negcorr,nullcorr]
            i = i+1
        data[spd] = df
        
    return data    
       