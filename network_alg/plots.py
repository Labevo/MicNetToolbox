from typing import Dict, Tuple
import networkx as nx
from networkx.classes import graph
import numpy as np
from functools import partial
from bokeh.plotting import from_networkx,figure
from bokeh.models import Circle
from bokeh.models import HoverTool
from bokeh.models import MultiLine
from bokeh.models import NodesAndLinkedEdges
from bokeh.models import Range1d
from bokeh.transform import linear_cmap
from bokeh.palettes import brewer
from bokeh.palettes import Spectral4
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
            COLORS=[(1,0,0,1) if i==-1 else cmap(i*100+20) for i in frame.HDBSCAN.to_list()]
            return COLORS
        else:
            COLORS=[cmap(i*100+50) for i in frame.Community.to_list() ] 
            return COLORS

    Map_Text={str(s):i for i,s in enumerate(frame.OTUS.tolist())}
    Map_Num={i:str(s) for i,s in enumerate(frame.OTUS.tolist())}
    Sizes_Nodes=(frame.Degree_Centrality*1000+100).astype(int)

    COLORS=get_colors(kind=kind)
    edge_attrs_color=color_edge(graph=graph)
    edge_attrs_size=edge_size(graph=graph,min=min,max=max)

    fig = plt.figure(figsize=(16, 16))
    plt.title(f'Network - {kind}', fontsize=20, fontweight='bold')
    
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



def plot_bokeh(graph:nx.Graph,frame:pd.DataFrame,
    max:float,min:float,
    kind:str='HDBSCAN',
    kind_network:str='circular'):
    
    graph=graph.copy()
    Map_Text={str(s):i for i,s in enumerate(frame.OTUS.tolist())}
    Map_Num={i:str(s) for i,s in enumerate(frame.OTUS.tolist())}
    Sizes_Nodes=(frame.Degree_Centrality*60+10).astype(int)
    Val_Node_Sizes={s:k for s,k in zip(Map_Text.keys(),Sizes_Nodes)}
    graph=nx.relabel_nodes(graph,Map_Num)

    edge_attrs_color=color_edge(graph=graph)
    edge_attrs_size=edge_size(graph=graph,min=min,max=max)

    nx.set_edge_attributes(graph, values=edge_attrs_color, name="edge_color")
    nx.set_edge_attributes(graph, values=edge_attrs_size, name="edge_sizes")
    nx.set_node_attributes(graph,name='Sizes',values=Val_Node_Sizes)

    if kind_network=='circular':
        graph_renderer = from_networkx(graph, nx.circular_layout)

    elif kind_network=='spring':
        graph_renderer = from_networkx(graph, nx.spring_layout)

    else:
        raise ValueError('Invalid layout type')

    plot = figure(width=900,
                  height=900,
                  x_range=Range1d(-1.1,1.1), 
                  y_range=Range1d(-1.1,1.1),
                  tooltips =[("Name", "@index")],
                  tools="pan,wheel_zoom,save,reset,box_zoom")

    plot.title.text = f"Network - {kind}"
    plot.title.text_font_size = "30px"

    graph_renderer.node_renderer.data_source.data['Degree_Centrality']=frame['Degree_Centrality'].tolist()
    graph_renderer.node_renderer.data_source.data['Betweeness_Centrality']=frame['Betweeness_Centrality'].tolist()
    graph_renderer.node_renderer.data_source.data['Closeness_Centrality']=frame['Closeness_Centrality'].tolist()
    graph_renderer.node_renderer.data_source.data['PageRank']=frame['PageRank'].tolist()


    #Colors
    if kind=='HDBSCAN':
        graph_renderer.node_renderer.data_source.data['Color_Tax_1']=frame['HDBSCAN'].tolist()
        graph_renderer.node_renderer.glyph = Circle(size='Sizes',fill_color=linear_cmap('Color_Tax_1', brewer['Spectral'][11], 
                                                                                     frame['HDBSCAN'].min(), 
                                                                                 frame['HDBSCAN'].max()))

    else:
        graph_renderer.node_renderer.data_source.data['Color_Tax_1']=frame['Community'].tolist()
        graph_renderer.node_renderer.glyph = Circle(size='Sizes',fill_color=linear_cmap('Color_Tax_1', brewer['Spectral'][11], 
                                                                                 frame['Community'].min(), 
                                                                                 frame['Community'].max()))

    
    graph_renderer.node_renderer.hover_glyph = Circle(size='Sizes', fill_color='#0e0f0f')
    graph_renderer.node_renderer.selection_glyph = Circle(size='Sizes', fill_color=Spectral4[2])


    graph_renderer.edge_renderer.glyph = MultiLine(line_color="edge_color", line_alpha=0.1, line_width="edge_sizes")
    graph_renderer.edge_renderer.selection_glyph = MultiLine(line_color=Spectral4[1], line_width="edge_sizes")
    graph_renderer.edge_renderer.hover_glyph = MultiLine(line_color="edge_color", line_width="edge_sizes")


    graph_renderer.selection_policy = NodesAndLinkedEdges()
    graph_renderer.inspection_policy = NodesAndLinkedEdges()


    plot.renderers.append(graph_renderer)

    node_hover_tool = HoverTool(tooltips=[("Index", "@index"),
                                     ("Centrality","@Degree_Centrality{(0.000)}"),
                                    ("Betweeness_Centrality","@Betweeness_Centrality{(0.000)}"),
                                    ("Closeness_Centrality","@Closeness_Centrality{(0.000)}"),
                                    ("PageRank","@PageRank{(0.000)}"),
                                    ("Comunidad","@Color_Tax_1")
                                    ])
    plot.add_tools(node_hover_tool)

    return plot