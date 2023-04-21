from shutil import rmtree
from pathlib import Path
import glob
import logreader.logreader as lr
import pipeline.functions as fs


raw_data_path = Path('/ceph/imaging1/arie')
temp_output_path = Path('/scratch/dspalla/2p_data')
preprocessed_data_path = Path('/ceph/imaging1/davide/2p_data')


# SESSION TO PREPROCESS
sessions = {'441406_fiano':['20230301','20230306','20230307','20230308','20230309','20230315','20230316','20230317','20230320','20230321'],
            '441394_ribolla':['20230301','20230306','20230307','20230308','20230309','20230315','20230316','20230317','20230320','20230321']
           }

# PREPROCESSING STEPS
preprocess_vr_data = True
preprocess_2p_video = False

# PARAMETERS
compute_flags = {'correct_luminance':True,
                 'refit_cnmf':True,
                 'deconvolve' :True,
                 'detrend_df_f': True,
                 'n_processes': 5 #number of parallel processes.
                }

cropping_params = {'cropping_limits': [10,-10, 10,-10],
                   'cropping_times':[0,-1]}

mc_params = {# Caiman parameters
             'max_shifts': [20, 20],  #maximum allowed rigid shifts (in pixels)
             'num_frames_split':300,
             'strides': [48, 48], # start a new patch for pw-rigid motion correction every x pixels
             'overlaps': [24, 24], # overlap between pathes (size of patch strides+overlaps)
             'max_deviation_rigid': 5,  # maximum allowed shifts from rigid template (in pixels)
             'border_nan': 'copy',
             'pw_rigid': True, # flag for performing non-rigid motion correction
             'use_cuda': True,
             'gSig_filt': None}

# metrics to compute on motion corrected video
metrics_params = ['mean_image','frame_average','corr_image']

cnmf_params  = {'fr': 30, # framerate of the video, very important!
                'p': 1,   # order of autoregressive process contstraint
                'nb': 2,  # number of backround components       
                'merge_thr': 0.85, # correlation th to merge to sources
                'rf': 20,    # half-size of patch in pixels
                'stride': 12, # "overlap between patches in pixels, should be roughly neuron diameter
                'K': 6,      # number of neurons per pathc
                'gSig': [6, 6], # half-size of neuron in pixels (row,columns)
                'ssub': 1, # spatial compression, if larger than one compresses
                'tsub': 1, # temporal compression, if larger than one compresses
                'method_init': 'greedy_roi',
                'min_SNR': 3.0,  # min snr for good components, used in an OR rule with min_cnn_thr and rval_thr
                'min_SNR_reject': 3.0, # min snr for good components, used in an AND rule with min_cnn_lowest and r_values_lowest
                'rval_thr': 0.9, # spatial footprint consistency
                'use_cnn': True,
                'min_cnn_thr': 0.9,
                'cnn_lowest': 0.9,
                'decay_time': 0.4,
                }
df_f_params = {'quantileMin':8,
               'frames_window':250       
                }

parameters = {'compute_flags': compute_flags,
              'cropping_params':cropping_params,
              'mc_params': mc_params,
              'metrics_params':metrics_params,
              'cnmf_params': cnmf_params,
              'df_f_params':df_f_params
                }


for animal in sessions.keys():
    animal_num = animal.split('_')[0]
    
    print(f'PROCESSING {animal} ...')
    
    for date in sessions[animal]:
        print(f'session date: {date}')
        
        
        session_path = raw_data_path.joinpath(f'{animal}/{date}_{animal_num}')
        print(session_path)
        
        try:
            tif_file = glob.glob(str(session_path)+'/*.tif')[0]
            log_file = glob.glob(str(session_path)+'/*.b64')[0]
        
        except IndexError:
            print('Files not found, skipping session')
        
        
        print(f'log file: {log_file}')
        print(f'tif file: {tif_file}')
        
        #make temp dir
        temp_path = temp_output_path.joinpath(f'{animal}/{date}')
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        
        #make output dir
        output_path = preprocessed_data_path.joinpath(f'{animal}/{date}')
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        #Decode log and process behaviour
        if preprocess_vr_data:
            try:
                vr_data = lr.preprocess_vr_data(tif_file = tif_file, log_file=log_file)
                lr.save_processed_vr_data(output_path,vr_data)
            except:
                print(f'Error in session {animal}_{date}, skipping ...')
                continue
        
        #Run caiman pipeline 
        if preprocess_2p_video:
            fs.preprocess_video(input_video=tif_file,
                            output_folder=output_path,
                            parameters=parameters,
                            temp_folder=temp_path)
        
        
        #remove temp dir 
        rmtree(temp_path)
        
        






