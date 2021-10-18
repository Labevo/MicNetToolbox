'''
SparCC is a python module for computing correlations in compositional data 
@DLegorreta

'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import shutil

from google.protobuf.descriptor import Error


from sparcc.core.SparCC import main_alg
from sparcc.core.io_methods import read_txt
from sparcc.core.util import clean_data_folder
from sparcc.core.io_methods import write_txt

from .MakeBootstraps import main as MakeBoots
from .PseudoPvals import main as PseudoPv

from pathlib import Path
from typing import Union,Any
from .core.io_methods import read_txt

import pandas as pd

import os


def clean_previous_file(name_file:str)->None:
    file_name=Path(name_file)
    if file_name.is_file():
        file_name.unlink()


class SparCC_MicNet:

    def __init__(self,
                name:str='experiment_sparCC',
                method:str='sparcc',
                low_abundance:bool=False,
                n_iteractions:int= 2,
                x_iteractions:int= 2,
                threshold:float= 0.1,
                normalization:str= 'dirichlet',
                log_transform:bool= True,
                save_corr_file:Union[str,Path]='sparcc/example/cor_sparcc.csv',
                save_cov_file:Union[str,Any]=None,
                num_simulate_data:int= 3,
                perm_template:str= 'permutation_#.csv',
                outpath:Union[str,Path]= 'sparcc/example/pvals/',
                type_pvalues:str= 'one_sided',
                outfile_pvals :Union[str,Path] ='sparcc/example/pvals/pvals_one_sided.csv',
                name_output_file:str= 'sparcc_output'
                ) -> None:

        self.name=name
        self.method=method
        self.n_iteractions=n_iteractions
        self.x_iteractions=x_iteractions
        self.threshold=threshold
        self.normalization=normalization
        self.log_transform=log_transform
        #self.save_corr_file=save_corr_file
        self.save_cov_file=save_cov_file
        self.num_simulate_data=num_simulate_data
        self.perm_template=perm_template
        self.outpath=outpath
        self.type_pvalues=type_pvalues
        self.outfile_pvals=outfile_pvals
        self.name_output_file=name_output_file
        self.low_abundance=low_abundance

    def compute(self,data_input:Union[str,pd.DataFrame]='sparcc/example/fake_data.txt',
                    save_corr_file:str='sparcc/example/cor_sparcc.csv'):


        self._preprocess()
        
        if type(data_input)==str:
            self.data_input=data_input
        
        else:
            data_input.to_csv('temp_sample_output.csv')
            self.data_input='temp_sample_output.csv'

        #Load the file
        try:
            L1=read_txt(self.data_input,index_col=0)
        except IOError as IOE:
            raise (IOE)

        if L1.shape[0]==0:
            try:
                L1=read_txt(self.data_input,sep=',')
            except IOError as IOE:
                raise (IOE)
        assert L1.shape[0]!=0,"ERROR!"


        #SparCC Algorithm
        cor,cov=main_alg(frame=L1,
                         method=self.method,
                         norm=self.normalization,
                         n_iter=self.n_iteractions,
                         verbose=True,
                         log=self.log_transform,
                         th=self.threshold,
                         x_iter=self.x_iteractions,
                         path_subdir_cor=self.path_corr_file,
                         path_subdir_cov=self.path_cov_file,
                         savedir=self.savedir)
    
        print("Shape of Correlation Matrix:",cor.shape)
        print("Shape of Covariance Matrix:",cov.shape)

        #Save Correlation
        write_txt(frame=cor,file_name=save_corr_file)


        clean_data_folder(path_folder=self.path_corr_file)
        clean_data_folder(path_folder=self.path_cov_file)

        

    def bootstrapping(self,data_input:Union[str,pd.DataFrame]='sparcc/example/fake_data.txt'):
        
        if not hasattr(self,'data_input'):
            if type(data_input)==str:
                self.data_input=data_input
        
            else:
                data_input.to_csv('temp_sample_output.csv')
                self.data_input='temp_sample_output.csv'

        MakeBoots(counts_file=self.data_input,nperm=self.num_simulate_data,
                  perm_template=self.perm_template,outpath=self.outpath)


    def pvalues(self,save_corr_file:str='sparcc/example/cor_sparcc.csv',template:str=''):
   
        PseudoPv(save_corr_file,template,
                 self.num_simulate_data, self.type_pvalues,
                 self.outfile_pvals)
    

    def run_all(self,data_input:Union[str,pd.DataFrame]='sparcc/example/fake_data.txt',
                    save_corr_file='Cor_SparCC.csv'):
        
        clean_previous_file(save_corr_file)
        clean_previous_file('temp_sample_output.csv')
        clean_previous_file(self.outfile_pvals)

        data_input=self._validation_format(data_input)
        data_input=self._filter_otus(data_input, self.low_abundance)

        self.compute(data_input,save_corr_file)
        self.bootstrapping(data_input='')
        file_pvals=str(self.outpath)+str(self.perm_template)
        corr_file=str(self.outpath)+'perm_cor_#.csv'

        # Iteraction in files
        for i in range(int(self.num_simulate_data)):
            print('#'*100)
            print(f'Iteration: {str(i)}')
            file_pvals1=file_pvals.replace('#',str(i))
            corr_file1=corr_file.replace('#',str(i))

            self.compute(data_input=file_pvals1,save_corr_file=corr_file1)

        #Clean Files: perm_corr
        folder=Path(self.outpath)
        List_files_rm=list(folder.glob(self.perm_template.replace('#','*')))
        for f in List_files_rm:
            f.unlink()

        self.pvalues(save_corr_file,template=corr_file)

        #Clean perm_cor_* files
        List_files_rm=list(folder.glob('perm_cor_*.csv'))
        for f in List_files_rm:
            f.unlink()
        
        setattr(self,'save_corr_file',save_corr_file)

        #move log
        # if Path(save_corr_file).parent.is_dir():
        #     output=Path(save_corr_file).parent/f'{self.name}.log'
        #     source=Path(f'sparcc/example/{self.name}.log')
        #     shutil.move(source,output)

        #self.save_corr_file=save_corr_file

    def _preprocess(self):
        #Define Temp Folder
        setattr(self,'savedir','./sparcc/data')
    
        #self._check_save_files()

        if os.path.exists(self.savedir) and os.path.isdir(self.savedir):
            try:
                shutil.rmtree("./sparcc/data")

            except OSError as e:
                print("Error: {0}:{1}".format('./temp_files/*',e.strerror))
            
            self.path_corr_file=os.path.join(str(self.savedir),'corr_files')
            self.path_cov_file=os.path.join(str(self.savedir),'cov_files')
            
            if not os.path.exists(self.path_corr_file):
                os.makedirs(self.path_corr_file)

            if not os.path.exists(self.path_cov_file):
                os.makedirs(self.path_cov_file)

        else:
            os.makedirs(self.savedir)
            self.path_corr_file=os.path.join(str(self.savedir),'corr_files')
            os.makedirs(self.path_corr_file)
            self.path_cov_file=os.path.join(str(self.savedir),'cov_files')
            os.makedirs(self.path_cov_file)

    def _validation_format(self,frame:pd.DataFrame)->pd.DataFrame:
                
        #OTUS Values
        if all((frame.iloc[:,0:].dtypes!='object').values):
            values_df=frame.iloc[:,0:].values.copy()
        else:
            raise ValueError('There is something wrong with the OTUS values')

        
        DF_out=pd.DataFrame(index=frame.index,data=values_df)
        return DF_out
    
    def _filter_otus(self,frame:pd.DataFrame, low_abundance:bool=False)->pd.DataFrame:
        
        #Remove singletons
        frame=frame.loc[(frame!=0).sum(axis=1)>=2,:].copy()
        #Remove low abudance < 5
        if low_abundance==True:
            frame=frame.loc[frame.sum(axis=1)>5,:].copy()
        else: 
            pass

        self._Index_col=frame.index

        return frame



    def __repr__(self) -> str:
        return f"SparCC with n_iteractions:{str(self.n_iteractions)} and x_iteractions: {str(self.x_iteractions)}"

    def __str__(self) -> str:
        return f"SparCC with n_iteractions: {str(self.n_iteractions)} and x_iteractions: {str(self.x_iteractions)}"