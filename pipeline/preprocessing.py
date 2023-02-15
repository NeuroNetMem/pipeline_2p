#%%
import os
import numpy as np
import matplotlib.pyplot as plt
import functions as fs
import time
import psutil
import gc
import pickle 
from pathlib import Path


import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF

#%%

animal = '429420_toms'
session = '20230203_429420'
video = '20230203_429420_00002.tif'

tif_file = f'/ceph/imaging1/arie/{animal}/{session}/{video}'

output_path = f'/scratch/davide/2p_data/{animal}/{session}'


compute_metrics = {'raw':True,'mcorr':True}

cropping_params = {'crop': False, 'cropping_limits': [100, -100, 100, -100]}

mc_params = {'motion_correct': True, 
             #'len_mcorr_chunk': 200, #len of temporal chunks for motion correction, in frames
             # Caiman parameters
             'max_shifts': [5, 5],  #maximum allowed rigid shifts (in pixels)
             'strides': [48, 48], # start a new patch for pw-rigid motion correction every x pixels
             'overlaps': [24, 24], # overlap between pathes (size of patch strides+overlaps)
             'max_deviation_rigid': 5,  # maximum allowed rigid shifts (in pixels)
             'border_nan': 'copy',
             'pw_rigid': True,  # flag for performing non-rigid motion correction
             'gSig_filt': None,}

cmnf_params  = {'fr': 30, # framerate of the video, very important!
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
                'min_SNR': 2.0,  # min snr for good components
                'rval_thr': 0.7, # spatial footprint consistency
                'use_cnn': False,
                'min_cnn_thr': 0.8,
                'cnn_lowest': 0.1,
                'decay_time': 0.4,
                }

#make metrics object

metrics = {'mean_img_raw':np.nan,
           'corr_img_raw':np.nan,
           'mean_img_mcorr':np.nan,
           'corr_img_mcorr':np.nan}


#make output folder 
Path(output_path).mkdir(parents=True, exist_ok=True)



#%% cropping
cropped_video_path = output_path+f'/cropped_{video}'
if cropping_params['crop']:
    print('Cropping movie ...')
    start_time = time.time()
    
    movie = cm.load(tif_file)
    [x1, x2, y1, y2] = cropping_params['cropping_limits']
    movie = movie[:, x1:x2, y1:y2]
    
    movie.save(cropped_video_path)
    
    # clear memory
    del(movie)
    gc.collect()

    print(f'Done in {(time.time()-start_time):.2f} seconds')

else:
    print('skipping cropping.')
    

if compute_metrics['raw']:
    print('Loading movie ...')
    start_time = time.time()
    if not os.path.isfile(cropped_video_path):
        print('File not found')

    movie = cm.load(cropped_video_path)
    mean_img = np.mean(movie, axis=0)
    corr_img,pnr_img = cm.summary_images.correlation_pnr(movie,swap_dim=False)
    
    metrics['mean_img_raw'] = mean_img
    metrics['corr_img_raw'] = corr_img
    metrics['pnr_img_raw'] = pnr_img
    
    # plot mean image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(mean_img, cmap='gray')
    plt.savefig(output_path+'/mean_image_raw.png')
    plt.close()
    
    
    
    
    # clear memory
    del(movie)
    gc.collect()

    print(f'Done in {(time.time()-start_time):.2f} seconds')
else:
    print('skipping metric calculations for RAW file')
    
    

if mc_params['motion_correct']:

    # motion correction
    print('Motion-correcting movie ...')
    start_time = time.time()

    mc_input_movie = output_path+f'/cropped_{video}'

    if not os.path.isfile(mc_input_movie):
        print('File not found.')

    # Calculate movie minimum to subtract from movie
    
    #movie = cm.load(mc_input_movie)
    #min_mov = np.min(movie)
    #len_movie = movie.shape[0]
    
    #mc_params['min_mov'] = min_mov
    #mc_params['splits_rig'] = int(len_movie/mc_params['len_mcorr_chunk'])
    #mc_params['splits_els'] = int(len_movie/mc_params['len_mcorr_chunk'])
    # Apply the parameters to the CaImAn algorithm
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

    # move video to motion corrected
    #destination_file = output_path+'/motion_correction/' + \
    #    mc.fname_tot_rig[0].split('/')[-1]
    #os.rename(mc.fname_tot_rig, destination_file)

else:
    print('Skipping motion correction.')
    

if compute_metrics['mcorr']:
    print('Loading movie ...')
    start_time = time.time()
    
    if mc_params['pw_rigid']:
        movie = cm.load(mc.fname_tot_els)
        
    else:
        movie = cm.load(mc.fname_tot_rig)
        
    mean_img = np.mean(movie, axis=0)
    #corr_img,pnr_img = cm.summary_images.correlation_pnr(movie,swap_dim=False)
    
    metrics['mean_img_mcorr'] = mean_img
    #metrics['corr_img_mcorr'] = corr_img
    #metrics['pnr_img_mcorr'] = pnr_img
    
    # plot mean image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(mean_img, cmap='gray')
    plt.savefig(output_path+'/mean_image_mcorr.png')
    plt.close()
    
    
    
    # clear memory
    del(movie)
    gc.collect()

    print(f'Done in {(time.time()-start_time):.2f} seconds')
else:
    print('skipping metric calculations for mcorr file')


print('Savign metrics in pickle')

filehandler = open(output_path+'/metrics.pickle', 'wb') 
pickle.dump(metrics, filehandler)
filehandler.close()


# MEMORY MAPPING
print('Memory mapping ...')
# memory map the file in order 'C'
if mc_params['pw_rigid']:
    fname_new = cm.save_memmap(mc.fname_tot_els, base_name='memmap_', order='C',
                               dview=dview) # exclude borders
else:
    fname_new = cm.save_memmap(mc.fname_tot_rig, base_name='memmap_', order='C',
                               dview=dview) # exclude borders

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F') 
    #load frames in python format (T x X x Y)

print('Done')

#%%
# restart cluster to clean up memory
print('Restart server to clean memory')
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=10, single_thread=False)

# Source extraction

#%%
print('Running source extraction ...')
opts = params.CNMFParams(params_dict=cmnf_params)
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)
cnm =cnm.refit(images)

#%%

print('Savign CNMF object in pickle')

filehandler = open(output_path+'/cnmf.pickle', 'wb') 
pickle.dump(cnm.estimates, filehandler)
filehandler.close()

print('Savign CNMF object in hdf5')
output_cnmf_file_path = output_path+'/cnmf.hdf5'
cnm.save(output_cnmf_file_path)


print('Done')
