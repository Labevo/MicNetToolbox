from typing import Any
import pandas as pd
def kind_file(x:str)->bool:

    if str(x).rsplit('.',1)[1]=='txt':
        return True
    else:
        return False


def filter_otus(frame:pd.DataFrame, low_abundace:bool=False):
    #Remove singletons
    frame = frame.loc[(frame!=0).sum(axis=1)>=2,:]
    
    #Remove low abudance < 5
    if low_abundace == True:
        frame = frame.loc[frame.sum(axis=1)>5,:]
    else: 
        pass

    indx = list(frame.index)

    return indx,frame