'''
This modules contains utility functions for the 2p video preprocessing pipeline.
'''

from pathlib import Path
import numpy as np
import pickle
import os
import shutil
import yaml
import gc
import glob
import pipeline.mymemmap as mym

import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF

import logreader.logreader as lr

def load_parameters_yaml(yaml_file):
    
    with open(yaml_file, 'r') as stream:
        parameters=yaml.safe_load(stream)

    
    return parameters
    


def load_session_parameters(animal=None,date=None,params_folder=None):
    '''
    Loads preprocessing parameters for a session from the given folder. If session parameters are not found, defaults to default_params.yml in the folder.

    Parameters:
        animal (str): Animal identifier for the session.
        date (str): Date of the session.
        params_folder (str): Path to the folder containing the parameter files.

    Returns:
        dict: Dictionary containing the loaded preprocessing parameters.
    '''
    session_file = params_folder.joinpath(f'parameters_{animal}_{date}.yml')
    if os.path.exists(session_file):
        filename = session_file
    else:
        print(f'No session parameters found for {animal}_{date}, using default.')
        filename = params_folder.joinpath('default_parameters.yml')
    
    with open(filename, 'r') as stream:
        parameters=yaml.safe_load(stream)

    
    return parameters
    


def make_output_folder(path):
    '''
    Creates output folders for different preprocessing stages.
    This contains old functionality and will be discontinued

    Parameters:
        path (str): Path to the main output folder.

    Returns:
        None
    '''
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+'/cropping').mkdir(parents=True, exist_ok=True)
    Path(path+'/motion_correction').mkdir(parents=True, exist_ok=True)
    Path(path+'/alignment').mkdir(parents=True, exist_ok=True)
    Path(path+'/source_extraction').mkdir(parents=True, exist_ok=True)
    Path(path+'/component_evaluation').mkdir(parents=True, exist_ok=True)
    print(f'built output directory @ {path}')
    return


def compute_metrics(movie,metrics_list):
    '''
    Computes selected metrics for the given CaImAn movie.

    Parameters:
        movie (np.ndarray): CaImAn movie data.
        metrics_list (list): List of strings specifying the metrics to compute.

    Returns:
        dict: Dictionary containing the computed metrics.
    '''
    
    metric_func = {'mean_image': lambda x: np.mean(x, axis=0),
                   'corr_pnr_images': lambda x: cm.summary_images.correlation_pnr(x,swap_dim=False),
                   'corr_image': lambda x: cm.summary_images.local_correlations(x,swap_dim=False),
                   'frame_average': lambda x: np.mean(x, axis=(1,2)),
                   
                  }
    
    computed_metrics = {}
    
    for m in metrics_list:
        if m not in metric_func.keys():
            print(f'{m} is not a valid metric keyword. Skipping.')
            continue
        else:
            computed_metrics[m] = metric_func[m](movie)

    
    return computed_metrics

# TO DO: add snr, spatial consistency and cnn metrics to the save, for future selection

def save_preprocessed_data(cnmf_file,output_path,frame_ts = None):
    '''
    Takes a cnmf.hdf5 file with the results of the preprocessing and saves it to the specified path as 'neural_data.pickle'.
    Optionally takes an array of frame timestamps to populate the 'frame_ts' key of the neural data dict. 
    If not provided, this is hard coded with a framerate of 29.94 Hz.

    Parameters:
        cnmf_file (str): Path to the cnmf.hdf5 file.
        output_path (str): Path to the folder where the data will be saved.
        frame_ts (Array-like, optional): 1d array with frame times in seconds.

    Returns:
        None
    '''
    
    output_path = Path(output_path)
    
    cnmf = load_CNMF(cnmf_file)
    ests = cnmf.estimates

    
    neural_data = {}
    neural_data['traces'] = ests.C[ests.idx_components]
    neural_data['footprints'] = ests.A[:,ests.idx_components].toarray().reshape(ests.dims[0],ests.dims[1],len(ests.idx_components))
    neural_data['df_f'] = ests.F_dff[ests.idx_components]
    
    spikes = []
    for trace in ests.S[ests.idx_components]:
        spikes.append(np.where(trace>0)[0])
    neural_data['deconvolved'] =  spikes
    
    # fake-plot the components to estimate coordinates. Is there a direct function? No trace of it in the documentation
    ests.plot_contours(idx=ests.idx_components)
    
    neural_data['positions'] = np.asarray([i['CoM'] for i in ests.coordinates])[ests.idx_components]
    neural_data['contour'] = [l['coordinates'] for i,l in enumerate(ests.coordinates) if i in ests.idx_components]
    
    if frame_ts is None:
        # if frame_ts is not provided they are inferred with scanner framerate of 29.94 Hz
        scanner_fps = 29.94
        end_time = neural_data['traces'].shape[1]/scanner_fps
        neural_data['frame_ts'] = np.arange(0, end_time, 1./scanner_fps)
        
    elif len(frame_ts)-1 != neural_data['traces'].shape[1]:
        # if shape does not match, a warning is raised and the provided frame ts is not uses
        # NOTE: the caiman algorithms returns traces without the last frame, so the comparison
        # is between len(frame_ts)-1 and len neural data
        len_traces = neural_data['traces'].shape[1]
        print(f'frame_ts with len {len(frame_ts)} incompatible with traces with len {len_traces} \n defaulting to 29.94 Hz timestamps.')
        scanner_fps = 29.94
        end_time = neural_data['traces'].shape[1]/scanner_fps
        neural_data['frame_ts'] = np.arange(0, end_time, 1./scanner_fps)
        
    else:
        # if both control pass, frame_ts is used
        neural_data['frame_ts'] = bp.asarray(frame_ts[:-1]) #last frame is dropped in the caiman traces, last ts dropped for consistency
    

    save_path = output_path.joinpath('neural_data.pickle')
    filehandler = open(save_path, 'wb') 
    pickle.dump(neural_data, filehandler)
    filehandler.close()
    
    return


def correct_avg_fluctuations(movie):
    '''
    Divides each frame by its average to correct for luminance fluctuations.
    Subtracts the minimum of the corrected movie to have non-negative values.

    Parameters:
        movie (np.ndarray): Input movie data.

    Returns:
        np.ndarray: Corrected movie data.
    '''
    movie = movie - np.min(movie.flatten())
    frame_avgs = np.mean(movie,axis=(1,2))
    corrected_movie = movie.transpose(1,2,0)/frame_avgs
    corrected_movie = corrected_movie.transpose(2,0,1) 
    
    return corrected_movie

def rename_file(old_file_name,new_file_name):
    '''
    Renames a file. Used during memory mapping.

    Parameters:
        old_file_name (str): Path to the existing file.
        new_file_name (str): Path to the desired new file name.

    Returns:
        None
    '''
    try:
        os.rename(old_file_name, new_file_name)
    except FileNotFoundError:
        print(f"{old_file_name} does not exist.")
    return

def make_output_dirs(output_path,preprocessed_data_path):
    '''
    Creates directories for temporary files and for final outputs.
    If the temporary directory exists, asks if the user wants to overwrite.

    Parameters:
        output_path (str): Path to the main output folder.
        preprocessed_data_path (str): Path to the preprocessed data folder.

    Returns:
        None
    '''
    
    
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    else:
        user_input = input(f'Directory {output_path} already exists, do you want to overwrite? (Type yes to overwrite, anything else to abort).')

        if user_input.lower() == 'yes':
            print(f'Overwriting {output_path}')
            shutil.rmtree(output_path)           
            os.makedirs(output_path)
        else:
            print('Aborting')
            sys.exit()
        
    #make preprocessed data folder if not exsits
    Path(preprocessed_data_path).mkdir(parents=True, exist_ok=True)
    
    return

##### caiman functions

def preprocess_video(input_video=None,output_folder=None,parameters=None,temp_folder=None,
                     keep_temp_folder=False):
    '''
    Preprocesses a video using CaImAn pipeline with the provided parameters.

    Parameters:
        input_video (str): Path to the input video file.
        output_folder (str): Path to the main output folder.
        parameters (dict): Dictionary containing preprocessing parameters.
        temp_folder (str): Path to the temporary output folder.
        keep_temp_folder (bool): If True, keeps the temporary output folder; otherwise, removes it.

    Returns:
        None
    '''
    # CHANGES:
    # docs
    # use Path everywhere
    
    # unpack parameter set
    compute_flags = parameters['compute_flags']
    cropping_params = parameters['cropping_params']
    mc_params = parameters['mc_params']
    metrics_params = parameters['metrics_params']
    cnmf_params = parameters['cnmf_params']
    df_f_params = parameters['df_f_params']
    
    input_video = str(input_video)
    output_folder = str(output_folder)
    temp_folder = str(temp_folder)
    
    #make output folder 
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    Path(temp_folder).mkdir(parents=True, exist_ok=True)
    
    
    ## save parameters
    with open(Path(output_folder).joinpath('parameters.yml'),'w') as file:
        yaml.dump(parameters,file)
        
    
    # CROPPING AND PREPROCESSING
    print('Cropping movie ...')
    movie = cm.load(input_video)
    movie = crop_movie(movie,cropping_params=cropping_params)     
    
    if compute_flags['correct_luminance']:
        print('Correcting luminance fluctuations')
        movie = correct_avg_fluctuations(movie)

    #save processed movie as mmap file
    cropped_video_path = temp_folder+f'/cropped.mmap'
    movie.save(cropped_video_path)

    # clear memory
    del(movie)
    gc.collect()
    
    #read cropped mmap file
    mmap_files = glob.glob(str(temp_folder)+'/cropped_*.mmap')
    cropped_video_path = mmap_files[0]
    
    # MOTION CORRECTION
    print('Motion-correcting movie ...')
    opts = params.CNMFParams(params_dict=mc_params)
    print('Starting CaImAn server')

    # start caiman server
    cm.cluster.stop_server()  # stop any already running clusters
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=compute_flags['n_processes'],
                                                     single_thread=False)

    mc = MotionCorrect([cropped_video_path], dview=dview,
                       **opts.get_group('motion'))
    # Perform rigid motion correction
    mc.motion_correct(save_movie=True)
    
    # renaming file
    print('Renaming motion corrected file ...')
    
    if mc_params['pw_rigid']:
        motion_corrected_video_file = mc.fname_tot_els[0]
    else:
        motion_corrected_video_file = mc.fname_tot_rig[0]
    
    new_name = temp_folder+'/mcorr_'+ motion_corrected_video_file.split('__')[-1]
    rename_file(motion_corrected_video_file,new_name)
    cm.stop_server(dview=dview)
    
    #COMPUTE METRICS ON MOTION CORRECTED VIDEO
    print('Computing metrics on motion corrected video')
    mmap_files = glob.glob(str(temp_folder)+'/mcorr_*.mmap')
    motion_corrected_video_file = mmap_files[0]
    
    movie = cm.load(motion_corrected_video_file)
    metrics = compute_metrics(movie,metrics_params)

    filehandler = open(output_folder+'/metrics.pickle', 'wb') 
    pickle.dump(metrics, filehandler)
    filehandler.close()


    # clear memory
    del(movie)
    gc.collect()
    
    # MEMORY MAPPING
    print('Converting motion corrected file in C order...')

    mmap_files = glob.glob(str(temp_folder)+'/mcorr_*.mmap')
    motion_corrected_video_file = mmap_files[0]
    c_mapped_file = mym.save_memmap([motion_corrected_video_file], base_name='mcorr_', order='C')
    
    # SOURCE EXTRACTION
    print('Running source extraction ...')
    
    c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local',n_processes=compute_flags['n_processes'], single_thread=False)

    Yr, dims, T = cm.load_memmap(c_mapped_file)
    images = np.reshape(Yr.T, [T] + list(dims), order='F') 


    opts = params.CNMFParams(params_dict=cnmf_params)
    cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
    cnm = cnm.fit(images)
    
    if compute_flags['refit_cnmf']:
        print('Refitting cnmf ...')
        cnm =cnm.refit(images)
        
    
    print('Evaluating components ...')
    cnm.estimates.evaluate_components(images, opts, dview=dview)

    if compute_flags['deconvolve']:
        print('Deconvolve components')
        cnm.deconvolve()

    if compute_flags['detrend_df_f']:
        print('Extract df/f')
        cnm.estimates.detrend_df_f(quantileMin=df_f_params['quantileMin'], frames_window=df_f_params['frames_window'])


    print('Savign CNMF object in hdf5')
    #to temp folder
    output_cnmf_file_path = temp_folder+'/cnmf.hdf5'
    cnm.save(output_cnmf_file_path)
    #to preprocessed data folder
    cnmf_destination_file = output_folder+'/cnmf.hdf5'
    cnm.save(cnmf_destination_file)

    # stop caiman server
    cm.stop_server(dview=dview)
    
    # remove log files
    log_files = glob.glob('*_LOG_*')
    for log_file in log_files:
        os.remove(log_file)
        
    #save to output folder
    print(f'Saving neural data for downstream analysis @{output_folder}')   
    frame_ts = lr.extract_frame_timestamps(input_video)
    save_preprocessed_data(output_cnmf_file_path,output_folder,frame_ts=frame_ts)
    
    if not keep_temp_folder:
        print('Cleaning temporary output directory')
        shutil.rmtree(temp_folder)
        

    print('Done')    
    
    
    return

    

def crop_movie(movie,cropping_params=None):
    '''
    Crops the video with given pixel and time parameters.

    Parameters:
        movie (np.ndarray): Input movie data.
        cropping_params (dict): Dictionary containing cropping parameters.

    Returns:
        np.ndarray: Cropped movie data.
    '''
    
    [x1, x2, y1, y2] = cropping_params['cropping_limits']
    [t1,t2] = cropping_params['cropping_times']
    movie = movie[t1:t2, x1:x2, y1:y2]
    return movie

def motion_correct_movie(movie_path,output_basename=None,mc_params=None,n_cpu=1):
    '''
    Motion corrects the given video file and saves it with the specified output basename.

    Parameters:
        movie_path (str): Path to the input video file.
        output_basename (str): Basename to be used for the output file (without extension).
        mc_params (dict): Dictionary containing motion correction parameters.
        n_cpu (int): Number of CPU cores to be used for motion correction. Defaluts to 1.

    Returns:
        None
    '''
    
    if not os.path.isfile(movie_path):
        print('File not found.')

    opts = params.CNMFParams(params_dict=mc_params)


    # Start a new cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=n_cpu,
                                                     single_thread=False)

    mc = MotionCorrect([movie_path], dview=dview,
                       **opts.get_group('motion'))
    
    # Perform rigid motion correction
    mc.motion_correct(save_movie=True)
    
    # renaming file
    print('Renaming motion corrected file ...')
    
    if mc_params['pw_rigid']:
        motion_corrected_video_file = mc.fname_tot_els[0]
    else:
        motion_corrected_video_file = mc.fname_tot_rig[0]
    
    output_name = output_path+f'/{output_basename}_'+ motion_corrected_video_file.split('__')[-1]
    rename_file(motion_corrected_video_file,output_name)
    
    
    cm.stop_server(dview=dview)
    
    return 
    

def mmap_F2C(input_fname,output_basename=None):
    '''
    Maps the given file from Fortran (F, column major) order to C (row major) order.

    Parameters:
        input_fname (str): Path to the input file in Fortran order.
        output_basename (str): Basename to be used for the output file (without extension).

    Returns:
        None
    '''
    

    fname_new = mym.save_memmap([motion_corrected_video_file], base_name='mcorr_', order='C')
    return

def extract_sources():
    pass

def evaluate_sources():
    pass


def run_2p_pipeline():
    pass








