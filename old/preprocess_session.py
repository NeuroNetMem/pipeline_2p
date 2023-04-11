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
import mymemmap as mym


import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF

#%%

animal = '429420_toms'
session = '20230211'
animal_num = animal.split('_')[0]
tif_file = glob.glob(f'/ceph/imaging1/arie/{animal}/{session}_{animal_num}/*.tif')[0]
video = tif_file.split('/')[-1]


output_path = f'/scratch/dspalla/2p_data/{animal}/{session}_no_equalization' # output path for temporary preprocessed file
preprocessed_data_path = f'/ceph/imaging1/davide/2p_data/{animal}/{session}'

compute_flags = {'crop': False,
                 'correct_luminance':False,
                 'compute_raw_metrics':False,
                 'motion_correct': False,
                 'compute_mcorr_metrics': False,
                 'deconvolve' :True,
                 'detrend_df_f': True,
                 'n_processes': 5 #number of parallel processes.
                }

metrics_params = {'raw':['mean_image','frame_average'],
                  'mcorr':['mean_image','frame_average']}

cropping_params = {'cropping_limits': [10, 500, 10,500],
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
             'gSig_filt': None}# number of chunks used in memory mapping. More chunks= less required memory}

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
    
    if compute_flags['correct_luminance']:
        print('Correcting luminance fluctuations')
        movie = fs.correct_avg_fluctuations(movie)
    
    
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
                                                     n_processes=compute_flags['n_processes'],
                                                     single_thread=False)

    mc = MotionCorrect([mc_input_movie], dview=dview,
                       **opts.get_group('motion'))
    # Perform rigid motion correction
    mc.motion_correct(save_movie=True)

    print(f'Done in {(time.time()-start_time):.2f} seconds')


    

else:
    print('Skipping motion correction.')
    mmap_files = glob.glob(str(output_path)+'/*.mmap')
    
    if len(mmap_files)==0:
        print('No motion corrected file, perform motion correction fist')
    if len(mmap_files)>1:
        print(f'More than one .mmap file, using {mmap_files[0]}')
    motion_corrected_video_file = mmap_files[0]
    
# MEMORY MAPPING
print('Memory mapping ...')
# memory map the file in order 'C'

if compute_flags['motion_correct']:
    if mc_params['pw_rigid']:
        motion_corrected_video_file = mc.fname_tot_els
    else:
        motion_corrected_video_file = mc.fname_tot_rig
       
    fname_new = mym.save_memmap(motion_corrected_video_file, base_name='memmap_', order='C',
                               dview=dview) # exclude borders
    
else:
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=10,
                                                     single_thread=False)
    
    fname_new = mym.save_memmap([motion_corrected_video_file], base_name='memmap_', order='C',
                                   dview=dview)
    
print(f'Done: fname_new={fname_new}')

# restart cluster to clean up memory
print('Stop server to clean memory')
cm.stop_server(dview=dview)

    
    


## Compute motion corrected metrics
if compute_flags['compute_mcorr_metrics']:
    try:
        print('Computing metrics on motion corrected video ...')
        start_time = time.time()

        movie = cm.load(fname_new)
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
    backend='local',n_processes=compute_flags['n_processes'], single_thread=False)


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


