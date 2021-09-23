from typing import Dict, Tuple
import networkx as nx
from networkx.classes import graph
import numpy as np
from functools import partial
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use('fivethirtyeight')

def make_proxy(clr, mappable, **kwargs):
    return Line2D([0, 1], [0, 1], color=clr, **kwargs)


def color_edge(graph:nx.Graph)->Dict[Tuple[int,int],str]:
    POSITIVE_COLOR, NEGATIVE_COLOR = "green", "red"
    edge_attrs_color = {}
    for start_node, end_node,valuew  in graph.edges(data=True):
        edge_color = POSITIVE_COLOR if valuew['weight']>=0 else NEGATIVE_COLOR
        edge_attrs_color[(start_node, end_node)] = edge_color

    return edge_attrs_color

def sizes_edges_f(x,max:float,min:float):
    return round(10*((max-x)/(max-min)))+1 if np.isnan(x)!=True else 0

def edge_size(graph:nx.Graph,min:float,max:float)->Dict[Tuple[int,int],str]:
    edge_attrs_size = {}
    size_edge=partial(sizes_edges_f,max=max,min=min)

    for start_node, end_node,valuew  in graph.edges(data=True):
        edge_color =  size_edge(valuew['weight'])
        edge_attrs_size[(start_node, end_node)] = edge_color
    return edge_attrs_size



def plot_matplotlib(graph:nx.Graph,frame:pd.DataFrame,
    max:float,min:float,
    kind:str='HDBSCAN',
    kind_network:str='circular'):

    if kind_network=='circular':
        pos = nx.circular_layout(graph)
    elif kind_network=='spring':
        pos = nx.spring_layout(graph)

    else:
        raise ValueError('Invalid layout type')

    cache=[]
    for start_node, end_node,valuew  in graph.edges(data=True):
        cache.append(np.abs(valuew['weight']))


    def get_colors(kind:str='HDBSCAN'):
        cmap = plt.cm.viridis
        if kind=='HDBSCAN':
            COLORS=[(1,0,0,1) if i==-1 else cmap(i*20+100) for i in frame.HDBSCAN.to_list()]
            return COLORS
        else:
            COLORS=[cmap(i*100+50) for i in frame.Community.to_list() ] 
            return COLORS

    Map_Text={str(s):i for i,s in enumerate(frame.OTUS.tolist())}
    Map_Num={i:str(s) for i,s in enumerate(frame.OTUS.tolist())}
    Sizes_Nodes=(frame.Degree_Centrality*300+400).astype(int)

    COLORS=get_colors(kind=kind)
    edge_attrs_color=color_edge(graph=graph)
    edge_attrs_size=edge_size(graph=graph,min=min,max=max)

    fig = plt.figure(figsize=(14, 14))
    plt.title('Network - Domos', fontsize=20, fontweight='bold')
    
    h1=nx.draw_networkx_nodes(graph, pos,  
              node_size=Sizes_Nodes.to_list(),
              node_color=COLORS,
              alpha=0.5,
              linewidths=1.0,
              edgecolors='black')
    h2=nx.draw_networkx_edges(graph, pos,
              edge_color=list(edge_attrs_color.values()),
              width=cache)

    proxies = [make_proxy(clr, h2,lw=5) for clr in ['green','red']]
    plt.legend(proxies,['Postive','Negative'])

    return fig





