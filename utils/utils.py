import pandas as pd
import numpy as np
import tifffile

from glob import glob
import pandas as pd
import os
from pathlib import Path
import numpy as np
import pickle


def list_sessions(animal,data_path="/ceph/imaging1/davide/2p_data"):
    session_list = glob(data_path + f'/{animal}/*/', recursive = True)
    
    return session_list

def list_complete_sessions(animal,data_path="/ceph/imaging1/davide/2p_data"):
    session_list = glob(data_path + f'/{animal}/*/', recursive = True)
    complete_sessions = [s for s in session_list if check_complete_session(s)]
    return complete_sessions

def check_neural_data_file(folder_path):
    file_path = os.path.join(folder_path, 'neural_data.pickle')
    return os.path.isfile(file_path)

def check_vr_data_file(folder_path):
    file_path = os.path.join(folder_path, 'behaviour_data.pickle')
    return os.path.isfile(file_path)

def check_trial_data_file(folder_path):
    file_path = os.path.join(folder_path, 'trial_data.csv')
    return os.path.isfile(file_path)

def check_complete_session(folder_path):
    return check_neural_data_file(folder_path) and check_trial_data_file(folder_path) and check_vr_data_file(folder_path)


def print_animal_summary(animal,data_path="/ceph/imaging1/davide/2p_data"):
    '''
    Gives a summary of the preprocessed data for the given animal
    '''
    
    scanner_fps = 30.
    behaviour_fps = 1000.
    
    print(f'Overview for {animal}')
    
    sessions = list_sessions(animal)
    complete_sessions = list_complete_sessions(animal)
    print(f'# of sessions: {len(sessions)}, of which {len(complete_sessions)} complete')
    print('\n')
    
    for s in sessions:
        print(f"date: {s.split('/')[-2]}")
        
        ## NEURAL DATA
        
        try:
            with open(Path(s).joinpath(f'neural_data.pickle'),'rb') as pfile:
                n_data = pickle.load(pfile)
                n_cells = n_data['traces'].shape[0]
                recording_duration =  n_data['traces'].shape[1]/scanner_fps

        except FileNotFoundError:
            n_cells = np.nan
            recording_duration = np.nan
            
        
        ## BEHAVIOUR DATA
        
        try:
            trial_data = pd.read_csv(Path(s).joinpath('trial_data.csv'))
            n_trials = len(trial_data)

        except FileNotFoundError:
            n_trials = np.nan
            
            
        try:
            with open(Path(s).joinpath(f'behaviour_data.pickle'),'rb') as pfile:
                    b_data = pickle.load(pfile)
                    vr_duration = b_data['time'].shape[0]/behaviour_fps
                    
        except FileNotFoundError:
            vr_duration = np.nan
            

        print(f' NEURAL DATA -cells: {n_cells}, rec duration: {recording_duration:.2f} s')
        print(f' VR DATA -trials: {n_trials}, vr duration: {vr_duration:.2f} s')
        print('\n')




def read_tiff_slice(tif_file,start_frame=0,end_frame=None):
    """
    Read a slice of frames from a TIFF video file and return it as a 3D NumPy array (time x pixels x pixels).

    Parameters:
        tif_file (str): The path to the TIFF video file.
        start_frame (int, optional): The index of the first frame to read (default is 0).
        end_frame (int, optional): The index of the last frame to read (default is None).
                                   If not provided, the function will read all frames until the end of the video.

    Returns:
        numpy.ndarray: A 3D NumPy array representing the slice of frames from the TIFF video.
                       The array has dimensions (num_frames, height, width).

    Note:
        - The function uses the 'tifffile' library to read the TIFF video.
        - If 'end_frame' is not provided, all frames from 'start_frame' to the end of the video will be read.
    """

    
    with tifffile.TiffFile(tif_file) as tif:
        
        if end_frame == None:
            print('No value provided for last frame, using video end, this can take a while ...')
            end_page = len(tif.pages)
            
    
        movie = [tif.pages[page].asarray() for page in range(start_frame,end_frame)]
        movie = np.dstack(movie).transpose(2,0,1)
    
    return movie


def make_tif_list(xls_file,animal_names):
    '''list all the tif files present in the xls info filefor each of the given animals'''

    data = pd.read_excel(xls_file)
    root_folder = '/ceph/imaging1/arie'
    tif_file_list = []

    for animal in animal_names:
        subset = data[data['name']==animal]
        print (f'{len(subset)} files found for mouse: {animal}')
        for i in range(len(subset)):
            animal_num = subset['mouse'].iloc[i]
            year = subset['year'].iloc[i]
            month = str(subset['month'].iloc[i]).zfill(2)
            day = str(subset['day'].iloc[i]).zfill(2)
            rec = subset['rec'].iloc[i]
            tif_file = subset['filename.tif'].iloc[i]
            session = f'{year}{month}{day}'
            tif_path = root_folder+f'/{animal_num}_{animal}/{session}_{animal_num}/{tif_file}.tif'
            tif_file_list.append(tif_path)
    
    return tif_file_list

