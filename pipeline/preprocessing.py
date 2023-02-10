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
session = '20221207_429420'
video = '20221206_429420_00002.tif'

tif_file = f'/ceph/imaging1/arie/{animal}/{session}/{video}'

output_path = f'/ceph/imaging1/davide/2p_data/{animal}/{session}'

plot_raw_image = False

cropping_params = {'crop': False, 'cropping_limits': [100, -100, 100, -100]}

mc_params = {'motion_correct': True, 'pw_rigid': True, 'save_movie_rig': False,
             'gSig_filt': (5, 5), 'max_shifts': (5, 5), 'niter_rig': 1,
             'strides': (48, 48),
             'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
             'max_deviation_rigid': 10,
             'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True, 'border_nan': 'copy'}

#make output folder 
Path(output_path).mkdir(parents=True, exist_ok=True)

if plot_raw_image:
    print('Loading movie ...')
    start_time = time.time()
    if not os.path.isfile(tif_file):
        print('File not found')

    movie = cm.load(tif_file)
    # plot mean image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(np.mean(movie, axis=0), cmap='gray')
    plt.savefig(output_path+'/raw_mean_image.png')
    plt.close()

    print(f'Done in {(time.time()-start_time):.2f} seconds')
else:
    print('skipping raw image plotting.')


#%% cropping
if cropping_params['crop']:
    print('Cropping movie ...')
    start_time = time.time()

    [x1, x2, y1, y2] = cropping_params['cropping_limits']
    movie = movie[:, x1:x2, y1:y2]
    cropped_video_path = output_path+f'/cropped_{video}'
    movie.save(cropped_video_path)

    # plot mean image
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(np.mean(movie, axis=0), cmap='gray')
    plt.savefig(output_path+'/mean_image.png')
    plt.close()

    # clear memory
    del(movie)
    gc.collect()

    print(f'Done in {(time.time()-start_time):.2f} seconds')

else:
    print('skipping cropping.')

if mc_params['motion_correct']:

    # motion correction
    print('Motion-correcting movie ...')
    start_time = time.time()

    mc_input_movie = output_path+f'/cropped_{video}'

    if not os.path.isfile(mc_input_movie):
        print('File not found.')

    # Calculate movie minimum to subtract from movie
    min_mov = np.min(cm.load(mc_input_movie))
    mc_params['min_mov'] = min_mov
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
    mc.motion_correct_rigid(save_movie=True, template=None)

    print(f'Done in {(time.time()-start_time):.2f} seconds')
    print(f'mmap file: {mc.fname_tot_rig}')

    # move video to motion corrected
    #destination_file = output_path+'/motion_correction/' + \
    #    mc.fname_tot_rig[0].split('/')[-1]
    #os.rename(mc.fname_tot_rig, destination_file)

else:
    print('Skipping motion correction.')

#%%

# MEMORY MAPPING
print('Memory mapping ...')
# memory map the file in order 'C'
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
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)

#%%

print('Savign CNMF object')
filehandler = open(output_path+'/cnmf.pickle', 'wb') 
pickle.dump(cnm, filehandler)
filehandler.close()


print('Done')

# %%
fig = plt.figure(figsize=(10,10))
Cn = cm.local_correlations(images.transpose(1,2,0))
Cn[np.isnan(Cn)] = 0
cnm.estimates.plot_contours_nb(img=Cn)
plt.savefig(output_path+'/components_footprints.png')

# %%

# %%
