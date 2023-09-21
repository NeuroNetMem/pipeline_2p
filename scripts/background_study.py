'''
This file contains a standalone script to preprocess a given experimental session.
It lets the user specify the session (.tif file) and all necessary caiman parameters and desired preprocessing steps.
It saves the preprocessing outptut in the specified folder, as well as all the interim steps in another, user-specified path for in-detail inspaction.
'''

#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import pipeline.functions as fs
import time
import psutil
import shutil
import sys
import gc
import glob
import pickle 
import yaml
from pathlib import Path
import pipeline.mymemmap as mym
from pprint import pprint


import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF

#%%

animal = '441406_fiano'
session = '20230307'

animal_num = animal.split('_')[0]
tif_file = glob.glob(f'/ceph/imaging1/arie/{animal}/{session}_{animal_num}/*.tif')[0]
video = tif_file.split('/')[-1]


keep_intermediate_steps = False # if true, keeps all intermediate steps (large disk requirement)
n_background_components = [1,3,5,10]

parameters = fs.load_parameters_yaml('/ceph/imaging1/arie/preprocess_params/parameters_background_study.yml')

print('PARAMETERS:')
pprint(parameters)



for n_bkg in n_background_components:
    
    print(f'PROCESSING WITH {n_bkg} BACKGROUND COMPONENTS')
    
    parameters['cnmf_params']['nb'] = n_bkg
    

    temp_folder = f'/scratch/dspalla/2p_data/{animal}/{session}' # output path for temporary preprocessed file
    output_folder = f'/ceph/imaging1/davide/2p_data/background_components_study/{animal}_{session}_{n_bkg}'

    
    
    fs.preprocess_video(input_video=tif_file,
                        output_folder=output_folder,
                        parameters=parameters,
                        temp_folder=temp_folder,
                        keep_temp_folder=keep_intermediate_steps)


    
    



    


            
    
        