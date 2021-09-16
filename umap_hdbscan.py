'''
Class to process UMAP and HDBSCAN together
@dlegor

'''
from pandas.core.base import DataError
import pandas as pd 
import numpy as np 
import umap
import hdbscan


__all__=['Embedding_Output']

class Embedding_Output:
    '''
    Class to estimate the embedding 
    and detect the outliers from the embedding
    '''

    def __init__(self,n_neighbors:int=15,min_dist:float=0.1,
    n_components:int=2,metric_umap:str='euclidean',metric_hdb:str='euclidean',
    min_cluster_size:int=15,min_sample:int=5,get_embedding:bool=True,get_outliers:bool=True,quantile_limit:float=0.9,
    output:bool=False):
        
        self.n_neighbors=n_neighbors
        self.min_dist=min_dist
        self.n_components=n_components
        self.metric_umap = metric_umap
        self.metric_hdb=metric_hdb
        self.min_cluster_size=min_cluster_size
        self.min_sample=min_sample
        self.get_embedding=get_embedding
        self.get_outliers=get_outliers
        self.quantile_limit=quantile_limit
        self.output=output
        # self.file_output=file_output

        

    def _validation_data(self,X):

        if all(X.select_dtypes('float').dtypes==float):
            pass
        else:
            raise DataError

    def fit(self,X):

        self._validation_data(X)
        
        #UMAP
        hyperbolic_mapper = umap.UMAP(output_metric='hyperboloid',
        metric=self.metric_umap,n_neighbors=self.n_neighbors,
        min_dist=self.min_dist,n_components=self.n_components,
        random_state=42).fit(X)

        self._shape_embedding=hyperbolic_mapper.embedding_.shape
  
        #HDBSCAN
        clusterer = hdbscan.HDBSCAN(min_samples=self.min_sample,
        min_cluster_size=self.min_cluster_size,
        metric=self.metric_hdb).fit(hyperbolic_mapper.embedding_)
        
        #Outliers
        threshold = pd.Series(clusterer.outlier_scores_).quantile(self.quantile_limit)
        outliers = np.where(clusterer.outlier_scores_ > threshold)[0]
        
        if self.output:
            return hyperbolic_mapper.embedding_,X.index.isin(outliers).astype(int),clusterer.labels_


    def __repr__(self) -> str:
        print('Embedding_Outliers')
