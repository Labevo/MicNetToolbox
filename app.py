''' 
Application to integrate all functionalities
@dlegor

'''
from typing import List
from pathlib import Path
from PIL import Image
import pandas as pd 
import numpy as np 
import matplotlib as mpl
import streamlit as st 
from utils import kind_file
from umap_hdbscan import Embedding_Output 

from network_alg import NetWork_MicNet
from sparcc import SparCC_MicNet


from bokeh.models import ColumnDataSource
from bokeh.plotting import figure

import base64

import streamlit as st
import pandas as pd
 


#CONTS
NORMALIZATION=['ninguna','clr','estandar','dirichlet']
METRIC=['euclidean','manhattan','canberra','braycurtis','mahalanobis',
'cosine','correlation','hellinger']

METRIC_HDB=['euclidean','manhattan','canberra','braycurtis','mahalanobis']



PATH_IMAG=Path('images')

OPTIONS=['Menú','Dashboard','SparCC','Network']
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



def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()
    return f'<a href="data:text/plain;base6 4,{object_to_download}" download="{download_filename}">{download_link_text}</a>'

def main():

    st.image(imagen_ixulabs,caption='Laboratorio de Evolución Molecular y Experimental',width=200)
    b1=st.sidebar.selectbox('Opciones',OPTIONS)
    
    if b1=='Menú':
        st.sidebar.markdown("""
        ### Refencias
        * [:link: UMAP](https://umap-learn.readthedocs.io/en/latest/)
        * [:link: Detección de Outliers con HDBSCAN](https://hdbscan.readthedocs.io/en/latest/outlier_detection.html)
        * [:link: SparCC](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002687)
        * [:link: NetworkX](https://networkx.org/)
        """)

        st.markdown("---")
        st.header("Menú General")
        st.markdown("""
        ### La aplicación tiene 3 opciones
        


        * Dashboard :
            Exploracion del comportamiento de los datos proporcionados via UMAP.
            
        * SparCC:
            Se puede ejecutar el algoritmo para detectar las correlaciones, para revisar detalles consulta el siguiente artículo:
                
            [Inferring Correlation Networks from Genomic Survey Data](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002687)
        
        * Networks:
            Calculos estandar sobre la red que se puede construir mediante las salidas de SparCC

        """)

    elif b1=='Dashboard':
    

        st.sidebar.header("Interactive Visualizer - 4 Ciénegas")

        #Parameters
        type_normalization=st.sidebar.selectbox('Tipo de Normalizacion',NORMALIZATION,
                                    help='Selecciona la que deseas aplicar, por default no se normalizan')
        
        file_input=st.sidebar.file_uploader(label='Archivo de Entrada',type=['txt','csv'],
                                            help='Sube el archivo que deseas procesar')

        #name_file=st.sidebar.text_input(label='Nombre el Archivo de Salida',value='Output_UMAP_HDBSCAN.csv')

        st.sidebar.markdown('---')
        st.sidebar.header('Parámetros de UMAP')
        n_neighbors=st.sidebar.slider(label='n_neighbors',min_value=5,max_value=50,step=1,
                                      value=15,help='Revisa la documentacion de UMAP')
        min_dist=st.sidebar.slider(label='min_dist',min_value=0.0,max_value=0.99,step=0.1,
                                   value=0.1,help='Revisa la documentacion de UMAP')
        n_components=st.sidebar.slider(label='n_components',min_value=2,max_value=3,step=1,
                                       value=2,help='Revisa la documentacion de UMAP')
        metric_umap=st.sidebar.selectbox('Selecciona una Métrica',options=METRIC,index=7,
                                        help='Revisa la documentacion de UMAP para un mejor entendimiento')
        taxa=st.sidebar.selectbox('Tiene Taxa',options=[True, False],index=1,
                                help='Tiene datos taxonómico tus datos?')
        st.sidebar.markdown('---')
        st.sidebar.header('Parametros de HDBSCAN')
        metric_hdb=st.sidebar.selectbox('Selecciona una Métrica',options=METRIC_HDB,index=3,
                                        help='Revisa la documentacion de HDBSCAN para un mejor entendimiento')
        min_cluster_size=st.sidebar.slider(label='min_cluster_size',min_value=5,max_value=100,step=1,value=15,
                                            help='Revisa la documentacion de HDBSCAN')
        min_sample=st.sidebar.slider(label='min_sample',min_value=1,max_value=60,step=1,value=5,
                                            help='Revisa la documentacion de HDBSCAN')

        B=st.sidebar.button(label='Estimacion')

        embedding_outliers=Embedding_Output(n_neighbors=n_neighbors,min_dist=min_dist,
        n_components=n_components,metric_umap=metric_umap,metric_hdb=metric_hdb,min_cluster_size=min_cluster_size,min_sample=min_sample,output=True)


        if file_input is not None and B==True:
            
            if kind_file(file_input.name):
                dataframe = pd.read_table(file_input)
            else:
                dataframe = pd.read_csv(file_input)

            
            st.info("Muestra de Archivo Cargado")
            st.dataframe(dataframe.head())

            #X=dataframe.select_dtypes(['int','float']).copy()
            #Text=dataframe.select_dtypes(['object']).copy()
            
            #st.info("Datos a Procesar")
            #st.dataframe(X.head())
            #X=X.astype('float').copy()

            if taxa:
                X=dataframe.iloc[:,2:].copy()
                Text=dataframe.iloc[:,:2].copy()
                X=X.astype('float').copy()
 
                Taxa=dataframe.TAXA.str.split(';').str.get(0)+'-'+\
                     dataframe.TAXA.str.split(';').str.get(1)+'-'+\
                     dataframe.TAXA.str.split(';').str.get(5)
                TOOLTIPS=[("Name", "@Name"),("Taxa","@Taxa")]
            else:
                X=dataframe.iloc[:,1:].copy()
                Text=dataframe.iloc[:,:1].copy()
                X=X.astype('float').copy()

                TOOLTIPS=[("Name", "@Name")]
    

            with st.spinner("En progreso"):
                st.info("Plot del Embedding")
                embedding_,o,l=embedding_outliers.fit(X)
                
    
                x = embedding_[:, 0]
                y = embedding_[:, 1]
                z = np.sqrt(1 + np.sum(embedding_**2, axis=1))

                disk_x = x / (1 + z)
                disk_y = y / (1 + z)

                colors = ["#%02x%02x%02x" % (int(r), int(g), int(b)) \
                for r, g, b, _ in 255*mpl.cm.viridis(mpl.colors.Normalize()(l))]
                

                TOOLS="hover,crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,tap,save,box_select,poly_select,lasso_select,"
                if taxa:
                    dataE=dict(x=disk_x.tolist(),y=disk_y.tolist(),Color=colors,Name=Text.iloc[:,0].tolist(),Taxa=Taxa.tolist())
                else:
                    st.write(Text.head())
                    dataE=dict(x=disk_x.tolist(),y=disk_y.tolist(),Color=colors,Name=Text.iloc[:,0].tolist())

                S=ColumnDataSource(dataE)
                
                p = figure(title="Embedding de Otus", x_axis_label='x',y_axis_label='y',
                           x_range=[-1,1],y_range=[-1,1],width=800, height=800,tools=TOOLS,tooltips=TOOLTIPS)

                p.circle(x=0.0,y=0.0,fill_alpha=0.1,line_color='black',size=20,radius=1,fill_color=None,line_width=2,muted=False)
                p.scatter(x='x',y='y',fill_color='Color', fill_alpha=0.3,line_color='black',radius=0.03,source=S)
                p.hover.point_policy="none"
                
                st.bokeh_chart(figure=p)
                st.markdown("---")
                st.markdown(f"""
                ## Descripcion
                Se tiene un total de {len(disk_x)} registros en el archivo. Detalles:
                
                    * Numero de clusters:  {len(set(l))}
                    * Numero de outliers:  {np.sum(o)}""")
            


            #if name_file is not None:
            name_file='Output_UMAP_HDBSCAN.csv'
            DF=pd.DataFrame()
            DF['Outliers']=o
            DF['Cluster']=l
            csv = convert_df(DF)
            
            st.download_button(
                label="Descarga el archivo",
                data=csv,
                file_name=name_file,
                mime='text/csv',help='El archivo contiene informacion \
                de los cluster y los otus detectados como outliers')
            



    

    elif b1=='SparCC':
        st.sidebar.title("Parámetros Correlacion")
        st.sidebar.header("Archivo")
        file_input=st.sidebar.file_uploader("Carga el Archivo",type=["csv","txt"])
        st.sidebar.header("Número de Inferencias")
        n_iteractions=st.sidebar.slider(label='n_iter',min_value=2,max_value=50,step=1,value=20)
        st.sidebar.header("Número de exclusiones")
        x_iteractions=st.sidebar.slider(label='x_iter',min_value=2,max_value=30,step=1,value=10)
        st.sidebar.header("Umbral de Exclusión")
        threshold=st.sidebar.slider(label='th',min_value=0.1,max_value=0.9,step=0.05,value=0.1)
        normalization=st.sidebar.selectbox(label="Tipo de Normalizacion",options=['dirichlet','normalizacion'])
        log_transform=st.sidebar.selectbox(label="Log Transformation",options=[True,False])
        With_Covarianza=st.sidebar.selectbox(label="Con archivo de Covarianza",options=[False,True])
        st.sidebar.title("P - Valores")
        num_simulate_data=st.sidebar.slider(label="Num de simulaciones",min_value=5,max_value=100,step=5,value=5)
        type_pvalues=st.sidebar.text_input(label="tipo de pvals",value="one_sided")

        
        B=st.sidebar.button(label='Estimacion')
        if file_input is not None and B==True:
            SparCC_MN=SparCC_MicNet(n_iteractions=n_iteractions,
                                    x_iteractions=x_iteractions,
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

            with st.spinner("En progreso"):
                if kind_file(file_input.name):
                    dataframe = pd.read_table(file_input,index_col=0)
                else:
                    dataframe = pd.read_csv(file_input,index_col=0)
                st.text("Muestra de Datos")
                st.dataframe(dataframe.head())
        
                SparCC_MN.run_all(data_input=dataframe)
                st.info("Termino la estimacion de las correlaciones")

            DF_SparCC=pd.read_csv(Path(SparCC_MN.save_corr_file).resolve(),index_col=0)
            DF_PValues=pd.read_csv(Path(SparCC_MN.outfile_pvals).resolve(),index_col=0)


            assert DF_SparCC.shape==DF_PValues.shape , "Error with SparCC Output and Pvalues"

            DF_Output=DF_SparCC[DF_PValues<0.05]
            
            del DF_SparCC,DF_PValues

            csv = convert_df(DF_Output)



            st.download_button(label="Descargar las Correlaciones",
                               data=csv,
                               file_name='SparCC_Output.csv',help='Descarga el archivo')
            #clean files
            clean_previous_file(SparCC_MN.save_corr_file)
            clean_previous_file(SparCC_MN.outfile_pvals)
            clean_previous_file(name_file='temp_sample_output.csv')
            

    
    elif b1=='Network':
        st.write(NetWork_MicNet) 
        st.write('---')



    





if __name__ == "__main__":
    main()