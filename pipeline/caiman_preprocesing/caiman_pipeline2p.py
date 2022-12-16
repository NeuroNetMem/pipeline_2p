
import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params

from caiman.source_extraction.cnmf import cnmf
from caiman.utils.utils import download_demo
from caiman.utils.visualization import plot_contours, nb_view_patches, nb_plot_contour

import numpy as np
import os
import psutil
import logging
import matplotlib.pylab as plt

### define directory path
dir_path = '/ceph/imaging1/melisa/photon2_test/test_2022/'
figure_path = '/scratch/melisa/photon2_test/figures/'

### define multiple files name
file_name = '20220117_testing_1_00001.tif'
file_name_ma = '20220117_testing_1_00001.tif-RunAv(10).tif'

fnames = [dir_path + file_name_ma]  # filename to be processed

# dataset dependent parameters
fr = 34                             # imaging rate in frames per second
decay_time = 0.4                    # length of a typical transient in seconds

# motion correction parameters
strides = (48, 48)          # start a new patch for pw-rigid motion correction every x pixels
overlaps = (24, 24)         # overlap between pathes (size of patch strides+overlaps)
max_shifts = (6,6)          # maximum allowed rigid shifts (in pixels)
max_deviation_rigid = 3     # maximum shifts deviation allowed for patch with respect to rigid shifts
pw_rigid = False             # flag for performing non-rigid motion correction

# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
gSig = (5, 5)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (21, 21)     # average diameter of a neuron, in general 4*gSig+1
merge_thr = 0.85            # merging threshold, max correlation allowed
rf = 15                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 6             # amount of overlap between the patches in pixels

K = None                       # number of components per patch
gSig = [15, 15]               # expected half size of neurons in pixels
method_init = 'corr_pnr'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 1                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization

# parameters for component evaluation
min_SNR = 2.0               # signal to noise ratio for accepting a component
rval_thr = 0.85              # space correlation threshold for accepting a component
cnn_thr = 0.99              # threshold for CNN based classifier
cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

min_corr = .5       # min peak value from correlation image
min_pnr = 1       # min peak to noise ration from PNR image



opts_dict = {'fnames': fnames,
            'fr': fr,
            'decay_time': decay_time,
            'strides': strides,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid}

opts = params.CNMFParams(params_dict=opts_dict)


#%% start a cluster for parallel processing (if a cluster already exists it will be closed and a new session will be opened)
if 'dview' in locals():
    cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)


# first we create a motion correction object with the parameters specified
mc = MotionCorrect(fnames, dview=dview, **opts.get_group('motion'))
# note that the file is not loaded in memory

#%% Run piecewise-rigid motion correction using NoRMCorre
mc.motion_correct(save_movie=True)
m_els = cm.load(mc.fname_tot_rig)
border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
    # maximum shift to be used for trimming against NaNs



#%% MEMORY MAPPING
# memory map the file in order 'C'
fname_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                           border_to_0=border_to_0, dview=dview) # exclude borders



K = None                       # number of components per patch
gSig = [15, 15]               # expected half size of neurons in pixels
method_init = 'corr_pnr'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 1                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization

# parameters for component evaluation
min_SNR = 2.0               # signal to noise ratio for accepting a component
rval_thr = 0.85              # space correlation threshold for accepting a component
cnn_thr = 0.99              # threshold for CNN based classifier
cnn_lowest = 0.1 # neurons with cnn probability lower than this value are rejected

min_corr = .2       # min peak value from correlation image
min_pnr = 1       # min peak to noise ration from PNR image


opts_dict = {'p': p,
            'nb': gnb,
            'rf': rf,
            'K': K,
            'gSig': gSig,
            'gSiz': gSiz,
            'stride': stride_cnmf,
            'rolling_sum': True,
            'only_init': True,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr,
            'min_SNR': min_SNR,
            'rval_thr': rval_thr,
            'method_init': method_init,
            'min_corr': min_corr,
            'min_pnr': min_pnr,
            'use_cnn': True,
            'min_cnn_thr': cnn_thr,
            'cnn_lowest': cnn_lowest}

# now load the file
Yr, dims, T = cm.load_memmap(fname_new)
images = np.reshape(Yr.T, [T] + list(dims), order='F')
    #load frames in python format (T x X x Y)

#%% restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

#%% RUN CNMF ON PATCHES
# First extract spatial and temporal components on patches and combine them
# for this step deconvolution is turned off (p=0). If you want to have
# deconvolution within each patch change params.patch['p_patch'] to a
# nonzero value
cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)

figure, axes = plt.subplots()
axes.imshow(np.mean(images,axis=0),cmap = 'gray')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(images[0,:,:]), 0.2, 'max')
for c in coordinates:
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    axes.plot(*v.T, c='b')
figure.savefig(figure_path + 'source_extraction_ma_2022_corr_pnr.png')


### plot traces

figure, axes = plt.subplots(1)
C_0 = cnm.estimates.C
C_0[0] += C_0[0].min()
for i in range(1, len(C_0)):
    C_0[i] += C_0[i].min() + C_0[:i].max()
    axes.plot(C_0[i])
axes.set_xlabel('t [frames]')
axes.set_yticks([])
# axes.vlines(timeline,0, 150000, color = 'k')
axes.set_ylabel('activity')
figure.set_size_inches([50., .5 * len(C_0)])
figure.savefig(figure_path + 'source_extraction_ma_traces_2022.png')