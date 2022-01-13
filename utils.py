from typing import Any
import pandas as pd
import webcolors

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

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
    return closest_name

