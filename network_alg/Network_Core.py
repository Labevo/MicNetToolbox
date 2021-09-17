"""
Mar 2021
@author: Natalia
"""

from os import error
import networkx as nx
from networkx.generators.directed import scale_free_graph
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import community  
from random import sample 
from itertools import zip_longest
from typing import Optional, Union,Dict


def get_weight(graph:nx.Graph):
    weights=np.asarray(list(nx.get_edge_attributes(graph, 'weight').values()))
    return weights

class NetWork_MicNet:

    def __init__(self,name:str='MicNet') -> None:
        self.name=name

    def basic_statistics(self,graph:nx.Graph)->None:
        
        wgh=get_weight(graph)
        self.posInt = sum(wgh>0)
        self.negInt = sum(wgh<0)
        self.nodes=graph.number_of_nodes()
        self.total_interactions=graph.number_of_edges()
        self.diameter=nx.diameter(graph)
        self.density=nx.density(graph)
        self.average_degree=np.mean([graph.degree(n) for n in graph.nodes()])
        self.average_clustering=nx.average_clustering(graph)
        self.average_shortest_path_length=nx.average_shortest_path_length(graph)


        return self

    
    def get_basic_statistics(self)->Dict[str,float]:

        #self.basic_statistics(graph)
        return {'Nodes':self.nodes,
                'Total interactions':self.total_interactions,
                'Positive interactions':self.posInt,
                'Negative interactions':self.negInt,
                'Density':self.density, 
                'Average degree':self.average_degree}

    
    def small_world_index(self,graph:nx.Graph, n_iter = 10):
        '''
        Takes a netowrk built with either nx.nx.from_numpy_matrix(data) or 
        with build netowrk
        n_iter = how many small index average
        Returns small-world index (See Humphries & Gurney, 2008)
        '''

        assert hasattr(self,'average_clustering'),'Error: you need to have the basic stats'


        small_world = []
        for i in range(0,int(n_iter)):
            # Build erdos reny netowrk with same number of nodes and connection probability
            G_rand =  nx.erdos_renyi_graph(n = self.nodes, p = self.density, seed=None, directed=False)

            #Obtain clustering coefficients
            cc_rand = nx.average_clustering(G_rand)

            #Obtain shortest path
            l_rand =  nx.average_shortest_path_length(G_rand)
    
            #Index numerator and denominator
            num = self.average_clustering/cc_rand
            den = self.average_shortest_path_length/l_rand
        
            # small world index
            small_world.append(num/den)
        
    
        return np.mean(small_world)

    @staticmethod
    def scale_free_index(graph:nx.Graph):
        '''
        Takes a nx network built with build_network
        Return scale free index (Li et al 2005)
        FALTA NORMALIZAR
        '''

        A = nx.to_numpy_matrix(graph)
        it = np.nditer(A, flags=['multi_index'])
        d_sum = 0
        for ind in it:
            if int(ind) == 1:
                i = it.multi_index[0]
                j = it.multi_index[1]
                mult = len(graph[i])*len(graph[j])
                d_sum = d_sum + mult
    
        #Smax
        smax = 1 
    
        #TODO calcualted Smax to normalize
        
        # Scale free index
        #scale_free = d_sum/smax
    
        return d_sum/smax

    def pvalue(self,graph:nx.Graph, topology:str='random', 
               metric:str='diameter', n_boot:int=100):

        '''
        Takes a graph G 
        the topology to compare to: random, scale-free or small-world
        the metric to compare: diamater, short_path or clust_coef
        number of bootstrapping simulations: set to 100 as default
        returns pval one-sided
        '''
        #Basic info from G
        n = self.nodes
        p = self.density
        k = int(self.average_degree)
        m = int(self.total_interactions/n)

        #Establishing which metric to use
        if metric == 'diameter':
            met_fun = nx.diameter
        elif metric == 'short_path':
            met_fun = nx.average_shortest_path_length
        elif metric == 'clust_coef':
            met_fun = nx.average_clustering

        met_list = []
        #Boostrapping to obtain pvals
        for i in range(0,n_boot):
            #Create network
            if topology == 'random':
                Gc = nx.erdos_renyi_graph(n=n, p=p, seed=None, directed=False)
            elif topology == 'scale_free':
                Gc = nx.barabasi_albert_graph(n=n,m=m, seed=None)
                Gc.to_undirected()
            elif topology == 'small_world':
                Gc = nx.watts_strogatz_graph(n=n, k=k, p=0.1,seed=None)
                Gc.to_undirected()
        
            #Calcualte metric
            met_list.append(met_fun(Gc))

        #Real network value
        real_met = met_fun(graph)
        met_list = np.array(met_list)
        pval = sum(met_list>=real_met)/len(met_list)
    
        return pval
    
    
    def topology_metrics(self,graph:nx.Graph, topology:str='random', n_boot:int=10):
        '''
        Takes a nx network build with build_network 
        Returns the following topological metric in a dict:
        diameter
        clustering coefficient
        average shortest path length
        small world index 
        scale free index (possibly)        
        '''

        data_dict={}
        
        data_dict['Diameter']=(self.diameter, self.pvalue(graph, metric='diameter', topology=topology, n_boot=n_boot))
        
        data_dict['Clustering coefficient']=(self.average_clustering, self.pvalue(graph,metric = 'clust_coef', 
        topology=topology,n_boot=n_boot))
        
        data_dict['Shortest average path length']=(self.average_shortest_path_length,self.pvalue(graph,metric = 'short_path', 
        topology=topology,n_boot=n_boot))

        data_dict['Small-world index']=(self.small_world_index(graph))

        data_dict['Scale-free index']=self.scale_free_index(graph)
        
        return data_dict

    @staticmethod    
    def structural_balance(graph:nx.Graph):
        '''
        Takes the raw correaltions obtained from sparCC (without normalization)
        Returns the percentage of balanced and unbalanced relationships
        And the percentage of each type of triangle
        '''
        #Build netowrk with relationships as 1 or -1
        
        edges = nx.get_edge_attributes(graph, 'weight')
        Gn = nx.Graph()
        for kv in edges.items():
            if kv[1] > 0:
                r = 1
            elif kv[1]<0:
                r = -1
            Gn.add_edges_from([kv[0]],relationship = r)
        #Find all triangles in network
        triangles = [c for c in nx.cycle_basis(Gn) if len(c)==3]
    
        #Classifiy triangles
        balanced1 = 0
        balanced2 = 0
        unbalanced1 =0 
        unbalanced2 = 0

        for triangle in triangles:
            #Get subgraph of triangle
            tri=nx.subgraph(Gn,triangle)
            data =  nx.get_edge_attributes(tri, 'relationship')
            rel = list(data.values())
            #Take the product of the relationships
            prod = rel[0]+rel[1]+rel[2]
            if prod == 3:
                balanced1+=1
            elif prod == -1:
                balanced2+=1
            elif prod == 1:
                unbalanced1+=1
            elif prod == -3:
                unbalanced2+=1
            
        D=len(triangles)
        baltotal = (balanced1 + balanced2)/D
        unbaltotal = (unbalanced1 + unbalanced2)/D
        bal_1 = balanced1/D
        bal_2 = balanced2/D
        unbal_1 = unbalanced1/D
        unbal_2 = unbalanced2/D

        data_dict = {
                'Percentage balanced': baltotal,
                'Percentage unbalanced': unbaltotal,
                'Triangles +++': bal_1,
                'Triangles --+': bal_2,
                'Triangles ++-': unbal_1,
                'Triangles ---':unbal_2}

        return data_dict

    @staticmethod    
    def key_otus(graph:nx.Graph,taxa:Union[pd.DataFrame,pd.Series]=None):
        '''
        Parameters
        ----------
        G : netowrk x graph
        taxa : dataframe with ASV and/or taxa
        n: number of top n nodes to return, default is 10
        'all' then returns all centraility values
        Returns
        -------
        key_otus: dictionary with dataframes, where each dataframe has the ASV, 
        taxa and centrality metric of the top 10 OTUS
        '''
        #Calculating centralities
        dcent = nx.degree_centrality(graph)
        #dcent = dict(sorted(dcent.items(), key=lambda item: item[1], reverse=True))
        bcent = nx.betweenness_centrality(graph)
        #bcent = dict(sorted(bcent.items(), key=lambda item: item[1], reverse=True))
        ccent = nx.closeness_centrality(graph)
        #ccent = dict(sorted(ccent.items(), key=lambda item: item[1], reverse=True))
        pRank = nx.pagerank(graph)
        #pRank = dict(sorted(pRank.items(), key=lambda item: item[1], reverse=True))
    
        #cent_metrics = [dcent, bcent, ccent, pRank]
    
        #col_name = ['Degree centrality', 'Betweeness centrality', 'Closeness centrality', 'PageRank']
    
        # if n == 'all':
        #     n = len(dcent)
    
        data_dict = {}
        if type(taxa)!='NoneType':
            data_dict['NUM_OTUS']=list(dcent.keys())
            data_dict['TAXA']=list(taxa.values)
            data_dict['Degree centrality']=list(dcent.values())
            data_dict['Betweeness centrality']=list(bcent.values())
            data_dict['Closeness centrality']=list(ccent.values())
            data_dict['PageRank']=list(pRank.values())



            # for num, data in enumerate(cent_metrics):
            #     vals = list(data.values())[:n]
            #     ind = list(data.keys())[:n]
            #     taxam = taxa.loc[ind]
            #     taxam[col_name[num]] = vals
            #     data_dict[col_name[num]] = taxam
            return data_dict

        else:
            data_dict['NUM_OTUS']=list(dcent.keys())
            data_dict['Degree centrality']=list(dcent.values())
            data_dict['Betweeness centrality']=list(bcent.values())
            data_dict['Closeness centrality']=list(ccent.values())
            data_dict['PageRank']=list(pRank.values())

            # for num, data in enumerate(cent_metrics):
            #     vals = list(data.values())[:n]
            #     ind = list(data.keys())[:n]
            #     taxam = taxa.loc[ind]
            #     taxam[col_name[num]] = vals
            #     data_dict[col_name[num]] = taxam
            return data_dict

    @staticmethod
    def community_analysis(graph:nx.Graph,taxa:Union[pd.DataFrame,pd.Series]=None):
        '''
            Parameters
            ----------
            G : graph built with build_network or nx function
            taxa: dataFrame with ASV and/or  taxa

            Returns
            -------
            num_com = number of communities
            df = Community with taxa id
            com_dict = Communities topology

            '''
        try:
            com = community.best_partition(graph)
        except:
            raise Exception("Bad graph type, use only non directed graph")


        if type(taxa) != 'NoneType':
            taxa['Community_id'] = com.values()    
        else:
            taxa=pd.DataFrame()
            taxa['Community_id'] = com.values()

        n_com = len(set(com.values()))
        data = []
        #Subnetwork analysis
        for com_id in range(0,n_com):
            subnet = [key  for (key, value) in com.items() if value == com_id]
            Gc=nx.subgraph(graph,subnet)
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



    @staticmethod
    def _build_network(frame:Union[np.ndarray,pd.DataFrame])->None:
        pass


    def __repr__(self) -> str:
        return f"{self.name}"