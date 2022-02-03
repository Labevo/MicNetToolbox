''' 
Application to integrate all functionalities
@dlegor

'''
from typing import List
from pathlib import Path
import hashlib
from PIL import Image
from bokeh.models.annotations import Label
from bokeh.models.layouts import Column
from bokeh.models.widgets import tables
from networkx.classes import graph
from networkx.classes.graph import Graph
import pandas as pd 
import numpy as np 
import matplotlib as mpl
import streamlit as st 
from utils import kind_file, filter_otus, get_colour_name
from umap_hdbscan import Embedding_Output 

from sparcc import SparCC_MicNet
from SessionState import get

from bokeh.models import ColumnDataSource, Plot
from bokeh.plotting import figure
import streamlit as st
import pandas as pd
from network_alg.utils import _build_network
from network_alg.utils import create_normalize_graph
from network_alg import NetWork_MicNet
from network_alg import HDBSCAN_subnetwork 
from network_alg import plot_matplotlib
from network_alg import plot_bokeh

#CONTS
key='1e629b5c8f2e7fff85ed133a8713d545678bd44badac98200cbd156d'

METRIC=['euclidean','manhattan','canberra','braycurtis',
'cosine','correlation','hellinger']
METRIC_HDB=['euclidean','manhattan','canberra','braycurtis']

PATH_IMAG=Path('images')
OPTIONS=['Menu','UMAP/HDBSCAN','SparCC','Network']

image_path=PATH_IMAG.resolve()/'logo_ie.png'
imagen_ixulabs=Image.open(image_path)

#st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')

def clean_previous_file(name_file:str)->None:
    file_name=Path(name_file)
    if file_name.is_file():
        file_name.unlink()


def menu_app():
    st.sidebar.markdown("""
    ### References
     * [:link: UMAP](https://umap-learn.readthedocs.io/en/latest/)
     * [:link: Outlier description with HDBSCAN](https://hdbscan.readthedocs.io/en/latest/outlier_detection.html)
     * [:link: SparCC](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002687)
     * [:link: NetworkX](https://networkx.org/)
    """)

    st.markdown("---")
    st.header("General Menu")
    st.markdown("""
        ### The application has three main components:
        


        * UMAP/HDBSCAN :
            Exploration of the data with UMAP and clustering algorithm HDBSCAN.
            
        * SparCC:
            Algorithm that can be run to estimate correlations from abundance data. For a more detail explanation the following paper is recommended:
                
            [Inferring Correlation Networks from Genomic Survey Data](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002687)
        
        * Networks:
            Large-scale metrics, structural balance and community information can be calculated for the network, this section uses as input the Sparcc output.

        **Note**:
            This dashobard runs in a virtual machine with limited capacity (2 CPUs and 6 GB of RAM), for large datasets please download our [repository](https://github.com/Labevo/MicNetToolbox) and run locally.

        """)


def sparcc_app():
    st.sidebar.header("File")
    file_input=st.sidebar.file_uploader("Upload abundance table",type=["csv","txt"])
    low_abundance=st.sidebar.selectbox('Filter low abudance',options=[True, False],index=1,
                                help='Do you want to filter low abundace (<5) OTUs?')

    st.sidebar.title("Correlation parameters")
    st.sidebar.header("Number of inferences")
    n_iteractions=st.sidebar.slider(label='n_iter',min_value=2,max_value=50,step=1,value=20)
    st.sidebar.header("Number of exclusions")
    x_iteractions=st.sidebar.slider(label='x_iter',min_value=2,max_value=30,step=1,value=10)
    st.sidebar.header("Exclusion threshold")
    threshold=st.sidebar.slider(label='th',min_value=0.1,max_value=0.9,step=0.05,value=0.1)
    normalization=st.sidebar.selectbox(label="Normalization type",options=['dirichlet','normalization'])
    log_transform=st.sidebar.selectbox(label="Log Transformation",options=[True,False])
    With_Covarianza=st.sidebar.selectbox(label="Covariance file",options=[False,True])
    st.sidebar.title("P - Values")
    num_simulate_data=st.sidebar.slider(label="Number of simulations",min_value=5,max_value=100,step=5,value=5)
    type_pvalues=st.sidebar.text_input(label="P-value type",value="one_sided")
    remove_taxa=st.sidebar.text_input(label='Column to remove',value='Taxa')

        
    B=st.sidebar.button(label='Run estimation')
    if file_input is not None and B==True:
        SparCC_MN=SparCC_MicNet(n_iteractions=n_iteractions,
                                    x_iteractions=x_iteractions,
                                    low_abundance=low_abundance,
                                    threshold=threshold,
                                    normalization=normalization,
                                    log_transform=log_transform,
                                    save_cov_file=None if With_Covarianza==False else "sparcc/example/cov_sparcc.csv", 
                                    num_simulate_data=num_simulate_data,
                                    perm_template="permutation_#.csv",
                                    outpath="sparcc/example/pvals/",
                                    type_pvalues=type_pvalues,
                                    outfile_pvals='sparcc/example/pvals/pvals_one_sided.csv',
                                    name_output_file="sparcc_output"
                                    )

        st.write(SparCC_MN)
        st.write('-----')

        with st.spinner("In progress"):
            if kind_file(file_input.name):
                dataframe = pd.read_table(file_input,index_col=0)
            else:
                dataframe = pd.read_csv(file_input,index_col=0)
        
            st.text("Data sample")
            st.dataframe(dataframe.head())
        

            if remove_taxa in dataframe.columns:
                dataframe=dataframe.drop(columns=[remove_taxa])

            SparCC_MN.run_all(data_input=dataframe)
            st.info("Correlation estimation has finished")

        DF_SparCC=pd.read_csv(Path(SparCC_MN.save_corr_file).resolve(),index_col=0)
        DF_PValues=pd.read_csv(Path(SparCC_MN.outfile_pvals).resolve(),index_col=0)

        


        assert DF_SparCC.shape==DF_PValues.shape , "Error with SparCC Output and Pvalues"

        DF_Output=DF_SparCC[DF_PValues<0.05]
        del DF_SparCC,DF_PValues

        DF_Output.index=SparCC_MN._Index_col
        DF_Output.columns=SparCC_MN._Index_col

        #Fill NaN with zeros
        DF_Output = DF_Output.fillna(0)

        csv = convert_df(DF_Output)

        st.download_button(label="Download correlation file",
                               data=csv,
                               file_name='SparCC_Output.csv',help='Downloads the correlation file')
        #clean files
        clean_previous_file(SparCC_MN.save_corr_file)
        clean_previous_file(SparCC_MN.outfile_pvals)
        clean_previous_file(name_file='temp_sample_output.csv')


def dashboar_app():
    st.sidebar.header("Interactive Visualizer")

    #Parameters      
    file_input=st.sidebar.file_uploader(label='Input file',type=['txt','csv'],
                                            help='Upload the file to process')
    taxa=st.sidebar.selectbox('Include Taxa',options=[True, False],index=1,
                                help='Does your file includes a column indicating taxa?')
    abudance_filter=st.sidebar.selectbox('Filter low abudance',options=[True, False],index=1,
                                help='Do you want to filter low abundace (<5) OTUs?')

    st.sidebar.markdown('---')
    st.sidebar.header('UMAP parameters')
    n_neighbors=st.sidebar.slider(label='n_neighbors',min_value=5,max_value=50,step=1,
                                      value=15,help='Check UMAP documentation')
    min_dist=st.sidebar.slider(label='min_dist',min_value=0.0,max_value=0.99,step=0.1,
                                   value=0.1,help='Check UMAP documentation')
    n_components=st.sidebar.slider(label='n_components',min_value=2,max_value=3,step=1,
                                       value=2,help='Check UMAP documentation')
    metric_umap=st.sidebar.selectbox('Select metric',options=METRIC,index=6,
                                        help='Check UMAP documentation')
    st.sidebar.markdown('---')
    st.sidebar.header('HDBSCAN parameters')
    metric_hdb=st.sidebar.selectbox('Select metric',options=METRIC_HDB,index=3,
                                        help='Check HDBSCAN documentation for more information')
    min_cluster_size=st.sidebar.slider(label='min_cluster_size',min_value=5,max_value=100,step=1,value=15,
                                            help='Check HDBSCAN documentation for more information')
    min_sample=st.sidebar.slider(label='min_sample',min_value=1,max_value=60,step=1,value=5,
                                            help='Check HDBSCAN documentation for more information')

    B=st.sidebar.button(label='Run estimation')

    embedding_outliers=Embedding_Output(n_neighbors=n_neighbors,min_dist=min_dist,
        n_components=n_components,metric_umap=metric_umap,metric_hdb=metric_hdb,min_cluster_size=min_cluster_size,min_sample=min_sample,output=True)


    if file_input is not None and B==True:
            
        if kind_file(file_input.name):
            dataframe = pd.read_table(file_input)

        else:
            dataframe = pd.read_csv(file_input)


        st.info("Data sample")
        st.dataframe(dataframe.head())


        if taxa:
            X=dataframe.iloc[:,2:].copy()
            X=X.astype('float').copy()
            indx, X=filter_otus(X, abudance_filter)
            
            Text=dataframe.iloc[indx,:2].copy()
            
            Taxa=dataframe.iloc[indx,1].str.split(';').str.get(0)+'-'+\
                    dataframe.iloc[indx,1].str.split(';').str.get(1)+'-'+\
                    dataframe.iloc[indx,1].str.split(';').str.get(5)
            TOOLTIPS=[("Name", "@Name"),("Taxa","@Taxa")]
        else:
            X=dataframe.iloc[:,1:].copy()
            indx, X=filter_otus(X,abudance_filter)
            Text=dataframe.iloc[indx,:1].copy()
            X=X.astype('float').copy()
            

            TOOLTIPS=[("Name", "@Name")]
    

        with st.spinner("In progress"):
            st.info("Embedding plot")
            embedding_,o,l=embedding_outliers.fit(X)
                
    
            x = embedding_[:, 0]
            y = embedding_[:, 1]
            z = np.sqrt(1 + np.sum(embedding_**2, axis=1))

            disk_x = x / (1 + z)
            disk_y = y / (1 + z)

            colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
            for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(l))]
            
            colors2  = [(int(r), int(g), int(b)) \
            for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(l))]

            tempd = dict(zip(l, colors2))

            TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
            
            if taxa:
                dataE=dict(x=disk_x.tolist(),y=disk_y.tolist(),Color=colors,Name=Text.iloc[:,0].tolist(),Taxa=Taxa.tolist())
            else:
                st.write(Text.head())
                dataE=dict(x=disk_x.tolist(),y=disk_y.tolist(),Color=colors,Name=Text.iloc[:,0].tolist())

            S=ColumnDataSource(dataE)

            p = figure(title="Embedding", x_axis_label='x',y_axis_label='y', output_backend = "svg",
                        x_range=[-1,1],y_range=[-1,1],width=800, height=800,tools=TOOLS,tooltips=TOOLTIPS)

            p.circle(x=0.0,y=0.0,fill_alpha=0.1,line_color='black',size=20,radius=1,fill_color=None,line_width=2,muted=False)
            p.scatter(x='x',y='y',fill_color='Color', fill_alpha=0.3,line_color='black',radius=0.03,source=S)
            p.hover.point_policy="none"
                
            st.bokeh_chart(figure=p)
            st.markdown("---")
            st.markdown(f"""
            ## Description:
               There is a total of {len(disk_x)} registers in the file, of which {sum([otu == -1 for otu in l])} were considered noise (signaled by -1 on the output file) and a total of:

                * Number of clusters:  {len(set(l))-1}
                * Number of outliers:  {np.sum(o)}""")
    


        #if name_file is not None:
        name_file='Output_UMAP_HDBSCAN.csv'
        DF=pd.DataFrame()
        if taxa:
            DF['Taxa']= Text.iloc[:,1]
        DF['Outliers']=o
        DF['Cluster']=l
        csv = convert_df(DF)
            
        st.download_button(
                label="Download file",
                data=csv,
                file_name=name_file,
                mime='text/csv',help='This file contains \
                cluster belonging and outlier information')

def network_app():
    st.title('Network analysis') 
    file_input=st.sidebar.file_uploader(label='Upload SparCC output file',type=['csv'],
    help="If you don't have this file, please calculate it at SparCC section")

    file_input2=st.sidebar.file_uploader(label='Upload HDBSCAN output file',type=['csv'],
    help="If you don't have this file, please calculate it at UMAP/HDBSCAN section")

    layout_kind=st.sidebar.selectbox(label='Plot layout',options=['Circular','Spring'],
                             help='For more information check layout in networkx')

    # KindP=st.sidebar.selectbox(label='Color de los Nodos',options=['HDBSCAN','Comunidades'])
    B=st.sidebar.button(label='Run estimation')


    if file_input is not None and file_input2 is not None and B==True:
        sparcc_corr = pd.read_csv(file_input,header=0,index_col=0).fillna(0)
        HD = pd.read_csv(file_input2,header=0,index_col=0).fillna(0)
        
        #sparcc_corr=sparcc_corr.drop(columns=[0])
        MAX=sparcc_corr.max().max()
        MIN=sparcc_corr.min().min()
        SHAPE=sparcc_corr.shape
        SHAPE2=HD.shape

        if int(SHAPE[1])!=int(SHAPE[0]):
            st.error('Error')
            raise EOFError('Error, correlation matrix is not square')

        if int(SHAPE[0])!=int(SHAPE2[0]):
            st.error('Error')
            raise EOFError('Error, correlation matrix and HDBSCAN file need to have the same number of rows')


        #Graph Process
        M=_build_network(sparcc_corr)
        Mnorm=create_normalize_graph(sparcc_corr)

        NetM=NetWork_MicNet()
        
        st.markdown("### Large-scale metrics of network")
        with st.spinner("In progress"):
            NetM.basic_description(corr=sparcc_corr)
        table1=pd.Series(NetM.get_description()).to_frame(name='Basic network information')
        st.table(table1)
        st.markdown('---')


        st.markdown("### Basic structural balance information")
        with st.spinner("In progress"):
            t2nw=NetM.structural_balance(M)
        
        table2=pd.Series(t2nw).to_frame(name='Structural balance')
        st.table(table2)

        Centrality=NetM.key_otus(Mnorm)
        CS_=int(sparcc_corr.shape[0]*.1)
        
        #embedding=Embedding_Output(metric_umap='euclidean',
        #                metric_hdb='braycurtis',
        #                n_neighbors=5,
        #                output=True,
        #                min_cluster_size=CS_,
        #                get_embedding=False)
        #HD=embedding.fit(sparcc_corr)

        # Communities table and plot
        st.markdown('---')
        st.markdown("## Communities information")
        with st.spinner("In progress"):
            Comunidades=NetM.community_analysis(Mnorm)
        
        st.write(f'Number of communities: {Comunidades["Number of communities"]}')
        st.table(Comunidades["Communities_topology"].T)

        st.text('Outliers have a value of -1')

        SparrDF=pd.DataFrame({'OTUS':Centrality['NUM_OTUS'],'ASV':sparcc_corr.index,
                      'Degree_Centrality':Centrality['Degree centrality'],
                      'Betweeness_Centrality':Centrality['Betweeness centrality'],
                      'Closeness_Centrality':Centrality['Closeness centrality'],
                      'PageRank':Centrality['PageRank'],
                      'HDBSCAN':HD.Cluster,
                      'Community':Comunidades['Community_data'].values.ravel()})
        
        fig3=plot_bokeh(graph=M,frame=SparrDF,
                       nodes = M.number_of_nodes(),
                       max=MAX,
                       min=MIN,
                       kind_network=str(layout_kind).lower(),
                       kind='Community')
        st.bokeh_chart(fig3)

        #HDBSCAN table and plot
        st.markdown('---')
        st.markdown("## HDBSCAN clusters information")
        with st.spinner("In progress"):
            Clusters=NetM.HDBSCAN_subnetwork(sparcc_corr, HD.Cluster)
        
        st.write(f'Number of clusters: {Clusters["Number of clusters"]}')
        st.table(Clusters["Clusters_topology"].T)

        fig4=plot_bokeh(graph=M,frame=SparrDF,
               nodes = M.number_of_nodes(),
               max=MAX,
               min=MIN,
               kind_network=str(layout_kind).lower(),
               kind='HDBSCAN')
        st.bokeh_chart(fig4)

        #Download centralities
        name_file='Output_Centralities.csv'
        DF=SparrDF
        csv = convert_df(DF)
        st.download_button(
                label="Download file",
                data=csv,
                file_name=name_file,
                mime='text/csv',help='This file contains \
                centralities and community information for each node')
     

def core_app():

    st.image(imagen_ixulabs,caption='Laboratorio de Evolución Molecular y Experimental',width=200)

    b1=st.sidebar.selectbox('Options',OPTIONS)
    
    if b1=='Menu':
        menu_app()

    elif b1=='UMAP/HDBSCAN':
        dashboar_app()
    
    elif b1=='SparCC':
        sparcc_app()

    elif b1=='Network':
        network_app()


        

def main():
    core_app()
    #session_state = get(password='')
    #pword=str(session_state.password)
    #
    #if hashlib.sha224(pword.encode('UTF-8')).hexdigest()!=key:
    #    pwd_placeholder = st.sidebar.empty()
    #    pwd = pwd_placeholder.text_input("Password:", value="", type="password")
    #    session_state.password = pwd
    #    pword=str(session_state.password)


    #    if hashlib.sha224(pword.encode('UTF-8')).hexdigest()==key:
    #        pwd_placeholder.empty()
    #        core_app()
    #    else:
    #        st.error("the password you entered is incorrect")
    #else:
    #    core_app()


if __name__ == "__main__":
    main()