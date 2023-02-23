#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import time
import psutil
import gc
import glob
import pickle 
import yaml
from pathlib import Path


import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF

#%%

animal = '429420_toms'
session = '20230210'
animal_num = animal.split('_')[0]
tif_file = glob.glob(f'/ceph/imaging1/arie/{animal}/{session}_{animal_num}/*.tif')[0]
video = tif_file.split('/')[-1]


output_path = f'/scratch/dspalla/2p_data/{animal}/{session}' # output path for temporary preprocessed file
preprocessed_data_path = f'/ceph/imaging1/davide/2p_data/{animal}/{session}'

compute_flags = {'crop': False,
                 'motion_correct': True,
                 'deconvolve' :True,
                 'detrend_df_f': True,
                 'compute_raw_metrics': False,
                 'compute_mcorr_metrics': False
                }

metrics_params = {'raw':['mean_image','corr_pnr_images','frame_average'],
                  'mcorr':['mean_image','frame_average']}

cropping_params = {'cropping_limits': [100, -100, 100, -100],
                   'cropping_times':[1000,30000]}

mc_params = {# Caiman parameters
             'max_shifts': [5, 5],  #maximum allowed rigid shifts (in pixels)
             'strides': [48, 48], # start a new patch for pw-rigid motion correction every x pixels
             'overlaps': [24, 24], # overlap between pathes (size of patch strides+overlaps)
             'max_deviation_rigid': 5,  # maximum allowed rigid shifts (in pixels)
             'border_nan': 'copy',
             'pw_rigid': True,  # flag for performing non-rigid motion correction
             'gSig_filt': None}

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
                'min_SNR': 3.0,  # min snr for good components
                'rval_thr': 0.9, # spatial footprint consistency
                'use_cnn': False,
                'min_cnn_thr': 0.8,
                'cnn_lowest': 0.1,
                'decay_time': 0.4,
                }
df_f_params = {'quantileMin':8,
               'frames_window':250       
                }


print(f'Analyzing file: {tif_file}')

#make output folder 
Path(output_path).mkdir(parents=True, exist_ok=True)
Path(preprocessed_data_path).mkdir(parents=True, exist_ok=True)

## save parameters
parameter_set = {'metrics': metrics_params,
                 'cropping':cropping_params,
                 'mcorr': mc_params,
                 'cnmf': cnmf_params,
                 'df_f':df_f_params
                }
with open(Path(output_path).joinpath('parameters.yml'),'w') as file:
    yaml.dump(parameter_set,file)



#%% cropping
cropped_video_path = output_path+f'/cropped_{video}'
if compute_flags['crop']:
    print('Cropping movie ...')
    start_time = time.time()
    
    movie = cm.load(tif_file)
    [x1, x2, y1, y2] = cropping_params['cropping_limits']
    [t1,t2] = cropping_params['cropping_times']
    movie = movie[t1:t2, x1:x2, y1:y2]
    
    movie.save(cropped_video_path)
    
    # clear memory
    del(movie)
    gc.collect()

    print(f'Done in {(time.time()-start_time):.2f} seconds')

else:
    print('skipping cropping.')
    

# compute metrics on raw video
if compute_flags['compute_raw_metrics']:
    
   
    print('Computing metrics on raw video ...')
    start_time = time.time()
    if not os.path.isfile(cropped_video_path):
        print('File not found')

    movie = cm.load(cropped_video_path)
    metrics = fs.compute_metrics(movie,metrics_params['raw'])
    
    filehandler = open(output_path+'/raw_metrics.pickle', 'wb') 
    pickle.dump(metrics, filehandler)
    filehandler.close()
    
    
    # clear memory
    del(movie)
    gc.collect()

    print(f'Done in {(time.time()-start_time):.2f} seconds')
else:
    print('skipping metric calculations for raw file')
    
    

if compute_flags['motion_correct']:

    # motion correction
    print('Motion-correcting movie ...')
    start_time = time.time()

    mc_input_movie = output_path+f'/cropped_{video}'

    if not os.path.isfile(mc_input_movie):
        print('File not found.')

    opts = params.CNMFParams(params_dict=mc_params)
    


    print('Starting CaImAn server')

    # start caiman server
    n_processes = psutil.cpu_count()  # counts local cpus
    cm.cluster.stop_server()  # stop any already running clusters
    # Start a new cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=10,
                                                     single_thread=False)

    mc = MotionCorrect([mc_input_movie], dview=dview,
                       **opts.get_group('motion'))
    # Perform rigid motion correction
    mc.motion_correct(save_movie=True)

    print(f'Done in {(time.time()-start_time):.2f} seconds')


    # MEMORY MAPPING
    print('Memory mapping ...')
    # memory map the file in order 'C'
    if mc_params['pw_rigid']:
        motion_corrected_video_file = mc.fname_tot_els
        fname_new = cm.save_memmap(mc.fname_tot_els, base_name='memmap_', order='C',
                                   dview=dview) # exclude borders
    else:
        motion_corrected_video_file = mc.fname_tot_rig
        fname_new = cm.save_memmap(mc.fname_tot_rig, base_name='memmap_', order='C',
                                   dview=dview) # exclude borders

    print('Done')
    
    # restart cluster to clean up memory
    print('Stop server to clean memory')
    cm.stop_server(dview=dview)


else:
    print('Skipping motion correction.')
    


## Compute motion corrected metrics
if compute_flags['compute_mcorr_metrics']:
    try:
        print('Computing metrics on motion corrected video ...')
        start_time = time.time()

        movie = cm.load(motion_corrected_video_file)
        metrics = fs.compute_metrics(movie,metrics_params['mcorr'])

        filehandler = open(output_path+'/mcorr_metrics.pickle', 'wb') 
        pickle.dump(metrics, filehandler)
        filehandler.close()


        # clear memory
        del(movie)
        gc.collect()

        print(f'Done in {(time.time()-start_time):.2f} seconds')
        
    except NameError:
        print('Motion corrected video not in memory, skipping metric calculation')
        
        
else:
    print('skipping metric calculations for motion corrected file')

    

c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=10, single_thread=False)


# Source extraction

print('Running source extraction ...')
    
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 


opts = params.CNMFParams(params_dict=cnmf_params)
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)

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
output_cnmf_file_path = output_path+'/cnmf.hdf5'
cnm.save(output_cnmf_file_path)


cm.stop_server(dview=dview)

log_files = glob.glob('*_LOG_*')
for log_file in log_files:
    os.remove(log_file)
    
print(f'Saving neural data for downstream analysis @{preprocessed_data_path}')   
fs.save_preprocessed_data(output_cnmf_file_path,preprocessed_data_path)
with open(Path(preprocessed_data_path).joinpath('parameters.yml'),'w') as file:
    yaml.dump(parameter_set,file)

print('Done')


