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




def make_output_folder(path):
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
    inputs: caiman movie, list of metric keywords
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

def save_preprocessed_data(cnmf_file,output_path):
    '''
    Takes a cnmf.hdf5 file with the results of the preprocessing and saves it to the specified path as 'neural_data.pickle'
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
    

    save_path = output_path.joinpath('neural_data.pickle')
    filehandler = open(save_path, 'wb') 
    pickle.dump(neural_data, filehandler)
    filehandler.close()
    
    return


def correct_avg_fluctuations(movie):
    '''
    Subtracts the frame average to each frame to correct for luminance fluctuations.
    Subtracts the minumum of the corrected movie to have non-negative values.
    '''
    movie = movie - np.min(movie.flatten())
    frame_avgs = np.mean(movie,axis=(1,2))
    corrected_movie = movie.transpose(1,2,0)/frame_avgs
    corrected_movie = corrected_movie.transpose(2,0,1) 
    
    return corrected_movie

def rename_file(old_file_name,new_file_name):
    try:
        os.rename(old_file_name, new_file_name)
    except FileNotFoundError:
        print(f"{old_file_name} does not exist.")
    return

def make_output_dirs(output_path,preprocessed_data_path):
    '''
    Create directory for temporary files and for final outputs.
    If temporary directory exists, ask if user wants to overwrite.
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

def preprocess_video(input_video=None,output_folder=None,parameters=None,temp_folder=None):
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
    save_preprocessed_data(output_cnmf_file_path,output_folder)
    
    
    print('Cleaning temporary output directory')
    shutil.rmtree(temp_folder)
        

    print('Done')    
    
    
    return

    

def crop_movie(movie,cropping_params=None):
    '''
    Crops video with given pixel and time parameters
    TO DO: implement data checks (crop params, boundary compatibility).
    '''
    
    [x1, x2, y1, y2] = cropping_params['cropping_limits']
    [t1,t2] = cropping_params['cropping_times']
    movie = movie[t1:t2, x1:x2, y1:y2]
    return movie

def motion_correct_movie(movie_path,output_basename=None,mc_params=None,n_cpu=1):
    '''
    Motion corrects movie, saves to given output basename
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
    maps given file from F to C order
    '''
    

    fname_new = mym.save_memmap([motion_corrected_video_file], base_name='mcorr_', order='C')
    return

def extract_sources():
    pass

def evaluate_sources():
    pass


def run_2p_pipeline():
    pass








