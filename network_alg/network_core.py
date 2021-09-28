"""
Mar 2021
@author: Natalia
"""
import networkx as nx
from networkx.classes import graph
from .utils import _normalize_corr,_build_network, create_normalize_graph
from networkx.algorithms import community as nx_comm
import pandas as pd
import numpy as np 
import community  
from random import sample 
from typing import Tuple, Union,Dict,Any,List


def get_weight(graph:nx.Graph):
    weights=np.asarray(list(nx.get_edge_attributes(graph, 'weight').values()))
    return weights


def small_world_index(G:nx.Graph,
                      n:int=0,
                      p:float=0.0,
                      cc:float=0.0,
                      l:float=0.0,
                      n_iter:int = 20)->float:
    '''
    Returns small-world index calcualted as proposed by Humphries & Gurney, 2008.
    
    Parameters
    ----------
    G : netowrk graph (obtained from normalized data)

    Returns
    -------
    small_world_index : small-world index.
    
    '''
    # Basic metrics from G

    if n==0:
        n = G.number_of_nodes()

    if p==0.0:
        p = nx.density(G)

    if cc==0.0:
        cc = nx.average_clustering(G)
    if l==0.0:
        l =  nx.average_shortest_path_length(G)
    
    small_world = []
    
    for i in range(0,int(n_iter)):
        # Build erdos reny netowrk with same number of nodes and connection probability
        G_rand =  nx.erdos_renyi_graph(n = n, p = p, seed=None, directed=False)

        #Obtain clustering coefficients
        cc_rand = nx.average_clustering(G_rand)

        #Obtain shortest path
        l_rand =  nx.average_shortest_path_length(G_rand)
    
        #Index numerator and denominator
        num = cc/cc_rand
        den = l/l_rand
        
        # small world index
        small_world.append(num/den)
        
    
    return np.mean(small_world)



class NetWork_MicNet:

    def __init__(self,p:float = 0.05,
                prem:float = 0.1, 
                per_type:str = 'random',
                groups:List[Any] = []) -> None:

        self.p=p
        self.prem=prem
        self.per_type=per_type
        self.groups=groups


    def basic_description(self,corr:Union[np.ndarray,pd.DataFrame])->None:

        self.description={}
        #TODO Mnomr
        if type(corr)==pd.DataFrame:
            corr=corr.values.copy()
        
        graph=_build_network(corr)
        wgh=get_weight(graph)
        del graph
        graph = create_normalize_graph(corr)

        #Calculate modularity
        try: 
            mod = nx_comm.modularity(graph, nx_comm.greedy_modularity_communities(graph))
        except ZeroDivisionError:
            mod = 'nan'

        #Description

        self.description['nodes']=graph.number_of_nodes()
        self.description['total_interactions']=sum(wgh>0)+sum(wgh<0)
        self.description['posInt'] = sum(wgh>0)
        self.description['negInt'] = sum(wgh<0)
        self.description['pos_neg_ratio']=self.description['posInt']/self.description['negInt'] if \
            self.description['negInt']!=0 else 0.0
        self.description['density']=nx.density(graph)        
        self.description['average_degree']=np.mean([graph.degree(n) for n in graph.nodes()])
        self.description['degree_std']=np.std([graph.degree(n) for n in graph.nodes()])
        self.description['diameter']=nx.diameter(graph)
        self.description['average_clustering']=nx.average_clustering(graph)
        self.description['average_shortest_path_length']=nx.average_shortest_path_length(graph)
        self.description['modularity']=mod
        self.description['small_world_index']=small_world_index(graph,
                                                                n=self.description['nodes'],
                                                                p=self.description['density'],
                                                                cc=self.description['average_clustering'],
                                                                l=self.description['average_shortest_path_length']
                                                                )

    def get_description(self)->Dict[str,Union[int,float]]:

        if not hasattr(self,'description'):
            raise AttributeError("There is no attribute: description")
        
        data_dict = {
        'Nodes':self.description['nodes'],
        'Total interactions':self.description['total_interactions'],
        'Positive interactions':self.description['posInt'],
        'Negative interactions':self.description['negInt'],
        'Pos-Neg ratio': self.description['pos_neg_ratio'],
        'Density':self.description['density'], 
        'Average degree':self.description['average_degree'],
        'Degree std': self.description['degree_std'],
        'Diameter':self.description['diameter'],
        'Clustering coefficient':self.description['average_clustering'],
        'Shortest average path length': self.description['average_shortest_path_length'],
        'Modularity': self.description['modularity'],
         'Small-world index': self.description['small_world_index']
        }
    
        return data_dict


    def percolation_sim(self,graph:nx.Graph)->pd.DataFrame:
        '''
        Function that performs percolation on the correlation matrix
        corrm, by removing the proportion of nodes specified in prem.
        The removal can be done : randomly, or by degree, closness or betweeness 
        centrality.
        
        Parameters
        ----------
        corr : corelation obtain from SparCC read as a pandas DataFrame or numpy matrix
        prem : proportion of nodes to remove in each iteration. Only fractional values.
        per_type : can take the values 'random', 'deg_centrality', 'clos_centrality', 'bet_centrality' 
                specifying how to remove the nodes from the interaction nework.

        Returns
        -------
        df : pandas dtaframe with each percolation iteration, displaying the percentage of removal
            and the change in the following netowrk metrics: 'Network density','Average degree', 
            'Number of components', 'Size of giant component','Fraction of giant component', 
            'Number of communities' and 'Modularity'.
        '''

        # Nodes to remove each iteration
        n = graph.number_of_nodes()
        nr = int(self.prem*n)
        nr_cum = 0
        
        if self.per_type == 'random':
            #Initial node list
            nod_list = list(graph.nodes())
        elif self.per_type == 'deg_centrality':
            cent = nx.degree_centrality(graph)
            cent = dict(sorted(cent.items(), key=lambda item: item[1], reverse=True))
            nod_list =list( cent.keys())
        elif self.per_type == 'bet_centrality':
            cent = nx.betweenness_centrality(graph)
            cent = dict(sorted(cent.items(), key=lambda item: item[1], reverse=True))
            nod_list =list( cent.keys())
        elif self.per_type == 'clos_centrality':
            cent = nx.closeness_centrality(graph)
            cent = dict(sorted(cent.items(), key=lambda item: item[1], reverse=True))
            nod_list =list( cent.keys())

        #Counter 
        j = 0
        
        #Data to store results
        df = pd.DataFrame(columns = ['Fraction of removal',
                                     'Network density','Average degree', 
                                     'Number of components', 'Size of giant component',
                                     'Fraction of giant component', 'Number of communities',
                                     'Modularity'])
        
        #Loop
        while len(nod_list)>nr:      
            # Choose nodes
            if self.per_type == 'random':
                nod_rem = sample(nod_list,nr) 
            elif self.per_type == 'deg_centrality' or self.per_type == 'bet_centrality' or self.per_type == 'clos_centrality':
                nod_rem  = nod_list[0:nr]
                
            #Nodes removed so far
            nr_cum = nr_cum + nr
            
            #Remove nodes from network
            graph.remove_nodes_from(nod_rem)
            
            #Fraction of current removal
            fr_rem = nr_cum/n
            
            #Calculate metrics and store them
            x = []
            # 1- Fraction of removal
            x.append(fr_rem)
            # 2- netowrk density
            x.append(nx.density(graph))
            # 3- average degree
            x.append(np.mean([graph.degree(n) for n in graph.nodes()]))
            # Number of  components
            x.append(nx.number_connected_components(graph))
            # Calculat size of largest component
            components = nx.connected_components(graph)
            x.append(len(max(components, key=len)))
            #Fractions of nodes belonging to the giant component
            x.append(x[4]/nx.number_of_nodes(graph))
            # Calculate comminuties
            com = community.best_partition(graph)
            x.append(len(set(com.values())))
            #Calculate modularity
            try: 
                x.append(nx_comm.modularity(graph, nx_comm.greedy_modularity_communities(graph)))
            except ZeroDivisionError:
                x.append('nan')
        
            #append current iteration to data
            df.loc[j] = x 
            
            #Node list from current network
            
            if self.per_type == 'random':
                nod_list = list(graph.nodes())
            elif self.per_type == 'deg_centrality' or self.per_type == 'bet_centrality' or self.per_type == 'clos_centrality':
                for elem in nod_rem:
                    nod_list.remove(elem)
                    
            #Update counter 
            j = j + 1
            
        return df

    @staticmethod    
    def structural_balance(graph:nx.Graph)->Dict[str,float]:
        '''
        Takes the raw correaltions obtained from sparCC (without normalization)
        Returns the percentage of balanced and unbalanced relationships
        And the percentage of each type of triangle
        '''
        #Build netowrk with relationships as 1 or -1
        
        edges = nx.get_edge_attributes(graph, 'weight')
        Gn = nx.Graph()
        for kv in edges.items():
            if kv[1] >= 0:
                r = 1
            elif kv[1]<0:
                r = -1
            else:
                print(f'Problem {kv[1]}')
                r=1

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
    def key_otus(graph:nx.Graph,taxa:Union[pd.DataFrame,pd.Series]=None)->Dict[str,List[Any]]:
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
        bcent = nx.betweenness_centrality(graph)
        ccent = nx.closeness_centrality(graph)
        try:
            pRank = nx.pagerank(graph)
        except:
            pRank={k:0 for k in dcent.keys()}
    
        data_dict = {}
        if taxa or type(taxa)==pd.DataFrame:
            data_dict['NUM_OTUS']=list(dcent.keys())
            data_dict['TAXA']=list(taxa.values)
            data_dict['Degree centrality']=list(dcent.values())
            data_dict['Betweeness centrality']=list(bcent.values())
            data_dict['Closeness centrality']=list(ccent.values())
            data_dict['PageRank']=list(pRank.values())
            return data_dict

        else:
            data_dict['NUM_OTUS']=list(dcent.keys())
            data_dict['Degree centrality']=list(dcent.values())
            data_dict['Betweeness centrality']=list(bcent.values())
            data_dict['Closeness centrality']=list(ccent.values())
            data_dict['PageRank']=list(pRank.values())
            return data_dict

    @staticmethod
    def community_analysis(graph:nx.Graph, 
                          taxa:Union[pd.Series,pd.DataFrame]=None):
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
    
        
        com = community.best_partition(graph)
        
        ## Check taxa and corr match
        if type(taxa)==pd.DataFrame:
            if len(graph.nodes) != taxa.shape[0]:
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


    def __repr__(self) -> str:
        return f"{self.name}"


class SyntheticNetwork:    
    """
    NETWORK TOPOLOGY
    @author: Natalia Favila

    TODO:Description
    """


    def __init__(self,n:int,m:int,p:float,k:int,pr:float,seed:int=None) -> None:
        self.n=n
        self.m=m 
        self.p=p 
        self.k=k
        self.pr=pr 
        self.seed=seed 

    def random_interaction(self)->pd.DataFrame:
        '''
        Function that creates a symmetric random interaction matrix with weights 
        from -1 to 1.
        
        
        Parameters
        ----------
        n : number of nodes.
        p : density of net.

        Returns
        -------
        B : symetric random network interaction matrix with weights drawn from a uniform
            distribution ranging form [-1,1]

        '''
        
        #Create netwrok
        G = nx.erdos_renyi_graph(n=self.n, p=self.p, seed=self.seed, directed=False)
        A = nx.to_numpy_matrix(G)
    
        #Assing interaction magnitude from a uniform distribution [-1,1]
        it = np.nditer(A, flags=['multi_index'])
        for ind in it:
            if int(ind) == 1:
                A[it.multi_index[0],it.multi_index[1]] =  round(np.random.uniform(-1, 1),2)
        
        #Make symectric matrix
        B = np.triu(A)
        B = B + B.T - np.diag(np.diag(A))
        
        return  pd.DataFrame(B)

    def scalef_interaction(self)->pd.DataFrame:
        '''
        Function that generates a scale-free interaction matrix with weights
        ranging for -1 to 1 using Barabasi-Albert algorithm.
        
        Parameters
        ----------
        n : number of nodes of the network.
        m : average number of edges per node (average degree/2)

        Returns
        -------
        B : symetric scale free interaction matrix with weights drawn from a 
            uniform distribution [-1,1].
        '''
        #Create netwrok
        G = nx.barabasi_albert_graph(n=self.n, m=self.m, seed=self.seed)
        A = nx.to_numpy_matrix(G)
    
        #Adding interaction magnitude from a uniform distribution [-1,1]
        it = np.nditer(A, flags=['multi_index'])
        for ind in it:
            if int(ind) == 1:
                A[it.multi_index[0],it.multi_index[1]] =  round(np.random.uniform(-1, 1),2)
        
        #Set matrix diagonal to 1 and make symectric matrix
        B = np.triu(A)
        B = B + B.T - np.diag(np.diag(A))
        
        return pd.DataFrame(B)

    def smallw_interaction(self)->pd.DataFrame:
        '''
        Function that generates a small-world interaction matrix with weights
        ranging for -1 to 1 using Watts Strogatz algorithm.
        
        Parameters
        ----------
        n : number of nodes.
        k : average degree.
        p : reconection probability
        p = 0 -- laticce network
        p = 1 -- random netowrk.

        Returns
        -------
        B : symetric small world interaction matrix with weights drawn from a 
            uniform distribution [-1,1].
        '''
        
        #Create netwrok
        G = nx.watts_strogatz_graph(n=self.n, k=self.k, p=self.pr, seed=self.seed)
        A = nx.to_numpy_matrix(G)
    
        #Assing interaction magnitude from a uniform distribution [-1,1]
        it = np.nditer(A, flags=['multi_index'])
        for ind in it:
            if int(ind) == 1:
                A[it.multi_index[0],it.multi_index[1]] =  round(np.random.uniform(-1, 1),2)
        
        #Set matrix diagonal to 1 and make symectric matrix
        B = np.triu(A)
        B = B + B.T - np.diag(np.diag(A))
        
        return pd.DataFrame(B)

    
    def __repr__(self) -> str:
        return f'Simulation Network with {self.n} Nodes'

    def __str__(self) -> str:
        return f'Simulation Network with {self.n} Nodes'


def topology_boostrap(corr:np.ndarray, n_boot:int= 100)->Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    '''
    Function that generates n_boot networks with known topology: 1) random,
    2) scale-free and 3) small-world that are comparable to the experimental network in
    corr in terms of density, number of nodes and average degree.
    For each simulated random,scale-free and small-world netowrk the following metrics are
    calculated: 'Modularity', 'Average shortest path','Clustering coefficient', 
    'degree variance' and 'Small-world index')
    
    Parameters
    ----------
    corr : interaction matrix as a pandas dataframe or numpy array
    n_boot : number of iterations 

    Returns
    -------
    df_rand,df_small,df_scale: three dataframes containing the simulations results.
                                df_rand contains the results from the simulated random
                                networks, df_small contains the results from the small-world
                                networks and df_scale contains the results from the scale-free
                                networks.
                                Each dataframe contains n_boot rows representing one simulated
                                network with the corresponding topology, and the columns are
                                the metrics obtained from that network.
    '''
    
    #Real network values
    assert corr.shape[0]==corr.shape[1],'It must be a square matrix'

    corrnorm = _normalize_corr(corr)
    G = _build_network(corrnorm)
    n = G.number_of_nodes()
    p = nx.density(G)
    k = int(np.mean([G.degree(n) for n in G.nodes()]))
    m = int(k/2)
    pr = 0.05

    SNetwork=SyntheticNetwork(n=n,m=m,p=p,k=k,pr=pr)
    
    df_rand = pd.DataFrame(columns = ('Modularity', 'Average shortest path',
                                      'Clustering coefficient', 'degree variance',
                                      'Small-world index'))
    df_scale = pd.DataFrame(columns = ('Modularity', 'Average shortest path',
                                      'Clustering coefficient', 'degree variance',
                                      'Small-world index'))
    df_small = pd.DataFrame(columns = ('Modularity', 'Average shortest path',
                                      'Clustering coefficient', 'degree variance',
                                      'Small-world index'))
    
    
    #Boostrapping to obtain pvals
    for i in range(0,n_boot):
        #Build random network
        corrSim = SNetwork.random_interaction()
        corrnorm = _normalize_corr(corrSim)
        G = _build_network(corrnorm)
        
        #Calculate modularity
        try: 
            mod = nx_comm.modularity(G, nx_comm.greedy_modularity_communities(G))
        except ZeroDivisionError:
            mod = 'nan'
    
        #Save metrics RANDOM
        l=nx.average_shortest_path_length(G)
        cc=nx.average_clustering(G)
        n=G.number_of_nodes()
        p = nx.density(G)

        df_rand.loc[i]= [mod,l,cc,np.std([G.degree(n) for n in G.nodes()]),
                         small_world_index(G,n=n,p=p,cc=cc,l=l,n_iter=30)]

        #Build small-world mnetwork
        corrSim = SNetwork.smallw_interaction()
        corrnorm = _normalize_corr(corrSim)
        Gsm = _build_network(corrnorm)

        #Calculate modularity
        try: 
            mod = nx_comm.modularity(Gsm, nx_comm.greedy_modularity_communities(Gsm))
        except ZeroDivisionError:
            mod = 'nan'
            
        #Save metrics SMALL-WORLD
        l=nx.average_shortest_path_length(Gsm)
        cc=nx.average_clustering(Gsm)
        n=Gsm.number_of_nodes()
        p = nx.density(Gsm)

        df_small.loc[i]= [mod,l,cc,np.std([Gsm.degree(n) for n in Gsm.nodes()]),
                         small_world_index(Gsm,n=n,p=p,cc=cc,l=l,n_iter=30)]
            
        
        #Build scale-free network
        corrSim = SNetwork.scalef_interaction()
        corrnorm = _normalize_corr(corrSim)
        Gsc = _build_network(corrnorm)

        #Calculate modularity
        try: 
            mod = nx_comm.modularity(Gsc, nx_comm.greedy_modularity_communities(Gsc))
        except ZeroDivisionError:
            mod = 'nan'
            
        #Save metrics SCALE-FREE
        l=nx.average_shortest_path_length(Gsc)
        cc=nx.average_clustering(Gsc)
        n=Gsc.number_of_nodes()
        p = nx.density(Gsc)

        df_scale.loc[i]= [mod,l,cc,np.std([Gsc.degree(n) for n in Gsc.nodes()]),
                         small_world_index(Gsc,n=n,p=p,cc=cc,l=l,n_iter = 30)]
    
    return df_rand,df_small, df_scale

def degree_comparison(corr:np.ndarray, topology:str = 'random', bins:int = 20)->pd.DataFrame:
    '''
    Function that returns the bins and Complementary Cumulative Distribution Function (CCDF)
    of the degrees of the interaction network provided in the corr matrix; and the bins 
    and CCDF of a comparable network (in terms of number of nodes, densitiy and average degree) 
    of a known topology which can be specified as: random, small_world or scale_free.
    
    Parameters
    ----------
    corr : interaction matrix as a pandas dataframe or numpy array
    topology : topology to compare to. It can be 'random', 'small_world' or 'scale_free'.
    bins: number of bins to divide the degree range.

    Returns
    -------
    df : pandas dataframe containing the data bins and CCDF and the simulated bins and 
         CCDF of the selected topology.
        
    '''
    #Real network values
    corrnorm = _normalize_corr(corr)
    G = _build_network(corrnorm)
    n = G.number_of_nodes()
    p = nx.density(G)
    k = int(np.mean([G.degree(n) for n in G.nodes()]))
    m = int(k/2)
    pr = 0.05

    SNetwork=SyntheticNetwork(n=n,m=m,p=p,k=k,pr=pr)
    #Real network degrees
    degrees = [G.degree(n) for n in G.nodes()]
    count, bins_count = np.histogram(degrees, bins=bins)
    pdf = count / sum(count)
    ccdf = 1-np.cumsum(pdf)

    if topology == 'random':    
        #Random network degrees
        randNet = SNetwork.random_interaction()
        corrnorm =_normalize_corr(randNet)
        Gr =_build_network(corrnorm)
        degreesR = [Gr.degree(n) for n in Gr.nodes()]
        countR, bins_countR = np.histogram(degreesR, bins=bins)
        pdf = countR / sum(countR)
        ccdfR = 1-np.cumsum(pdf)
        
    elif topology == "small_world":
        #Small  degrees
        smallNet = SNetwork.smallw_interaction()
        corrnorm = _normalize_corr(smallNet)
        Gsm = _build_network(corrnorm)
        degreesR = [Gsm.degree(n) for n in Gsm.nodes()]
        countR, bins_countR = np.histogram(degreesR, bins=bins)
        pdf = countR / sum(countR)
        ccdfR = 1-np.cumsum(pdf)
        
    elif topology == 'scale_free':
        #Scale network degrees
        scaleNet = SNetwork.scalef_interaction()
        corrnorm = _normalize_corr(scaleNet)
        Gsc = _build_network(corrnorm)
        degreesR = [Gsc.degree(n) for n in Gsc.nodes()]
        countR, bins_countR = np.histogram(degreesR, bins=bins)
        pdf = countR / sum(countR)
        ccdfR = 1-np.cumsum(pdf)

        
    df = pd.DataFrame()
    df['Data_bins'] = bins_count[1:]
    df['Data_CCDF'] = ccdf
    df['Simulated_bins'] = bins_countR[1:]
    df['Simulated_CCDF'] = ccdfR
    
    return df



     

