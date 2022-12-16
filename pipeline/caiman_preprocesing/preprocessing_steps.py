
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

file_name_motion_corrected = 'recording_20211112_00001_mc.tif'
file_name_motion_corrected_pw = 'recording_20211112_00001_mc_pw.tif'

## create file path and verify if it exists
file_path = dir_path + file_name
if not os.path.isfile(file_path):
    print('File does not exist')
#load tif file
#m_orig = cm.load(file_path)

## create file path and verify if it exists
file_path_ma = dir_path + file_name_ma
if not os.path.isfile(file_path_ma):
    print('File does not exist')
#load tif file
m_orig = cm.load(file_path_ma)
#m_orig_copy = m_orig.copy()
#m_orig_copy[np.where(m_orig_copy < 0)] = 0
#m_orig_copy.save(dir_path + file_name_clipped)

# define parameters for cluster inicialization
n_processes = psutil.cpu_count()
#cm.cluster.stop_server()
# Start a new cluster
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,
                                                 single_thread=False)

## parameters for motion correction
max_shifts = (5,5)  # maximum allowed rigid shift in pixels (view the movie to get a sense of motion)
strides =  (48, 48)  # create a new patch every x pixels for pw-rigid correction
overlaps = (24, 24)  # overlap between pathes (size of patch strides+overlaps)
num_frames_split = 100  # length in frames of each chunk of the movie (to be processed in parallel)
max_deviation_rigid = 5   # maximum deviation allowed for patch with respect to rigid shifts
pw_rigid = True  # flag for performing rigid or piecewise rigid motion correction
shifts_opencv = True  # flag for correcting motion using bicubic interpolation (otherwise FFT interpolation is used)
border_nan = 'copy'  # replicate values along the boundary (if True, fill in with NaN)
fr = 33                             # imaging rate in frames per second
decay_time = 0.4                    # length of a typical transient in seconds

opts_dict = {'fnames': [file_path_ma],
            'fr': fr,
            'decay_time': decay_time,
            'strides': strides,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid
             }


# create a motion correction object
mc = MotionCorrect([file_path], dview=dview, max_shifts=max_shifts,
                  strides=strides, overlaps=overlaps,
                  max_deviation_rigid=max_deviation_rigid,
                  shifts_opencv=shifts_opencv, nonneg_movie=True,
                  border_nan=border_nan)

mc.motion_correct(save_movie=True)

m_rig = cm.load(mc.mmap_file)
m_rig_path = (dir_path+ file_name_motion_corrected)
m_rig.save(m_rig_path)

#mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
#mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)

#mc.motion_correct(save_movie=True, template=mc.total_template_rig)
#m_els = cm.load(mc.fname_tot_els)
#m_els.save(dir_path+ file_name_motion_corrected_pw)



#mc.pw_rigid = True  # turn the flag to True for pw-rigid motion correction
#mc.template = mc.mmap_file  # use the template obtained before to save in computation (optional)

#mc.motion_correct(save_movie=True, template=mc.total_template_rig)
#m_els = cm.load(mc.fname_tot_els)
#m_els.save(dir_path+ file_name_motion_corrected_pw)

figure, axes = plt.subplots(1,2)

axes[0].imshow(np.mean(m_orig,axis=0), cmap = 'gray')
axes[1].imshow(np.mean(m_rig,axis=0), cmap = 'gray')

axes[0].set_title('Mean Original Movie')
axes[1].set_title('Mean Motion Corrected Movie')

figure.savefig(figure_path + 'motion_correction_mean_ma_2022.png')

border_to_0 = 0 if mc.border_nan is 'copy' else mc.border_to_0
m_new = cm.save_memmap(mc.mmap_file, base_name='memmap_', order='C',
                           border_to_0=border_to_0, dview=dview) # exclude borders


bord_px_rig = np.ceil(np.max(mc.shifts_rig)).astype(np.int)

final_size = np.subtract(mc.total_template_rig.shape, 2*bord_px_rig)
winsize = 100
swap_size = False
tml_orig, correlations_orig, flows_orig,norms_orig,crispness_orig = cm.motion_correction.compute_metrics_motion_correction(
    file_path,final_size[0],final_size[1],swap_dim=swap_size,winsize=winsize,play_flow=False)

tml_rig, correlations_rig, flows_rig,norms_rig,crispness_rig = cm.motion_correction.compute_metrics_motion_correction(
    m_rig_path,final_size[0],final_size[1],swap_dim=swap_size,winsize=winsize,play_flow=False)


corr1 = m_orig.local_correlations(eight_neighbours=True, swap_dim=False)
corr2 = m_rig.local_correlations(eight_neighbours=True, swap_dim=False)

figure, axes = plt.subplots(1,2)

axes[0].imshow(corr1, cmap = 'gray')
axes[1].imshow(corr2, cmap = 'gray')

axes[0].set_title('Correlation Original Movie')
axes[1].set_title('Correlation Motion Corrected Movie')

figure.savefig(figure_path + 'motion_correction_correlation_ma_2022.png')


#axes[0].set_title('Crispness = ' + f'{crispness_orig}')
#axes[1].set_title('Crispness = ' + f'{crispness_rig}')



#######################################################################

#### SORUCE EXTRACTION

# parameters for source extraction and deconvolution
p = 1               # order of the autoregressive system
K = None           # upper bound on number of components per patch, in general None
gSig = (7, 7)       # gaussian width of a 2D gaussian kernel, which approximates a neuron
gSiz = (29, 29)     # average diameter of a neuron, in general 4*gSig+1
Ain = None          # possibility to seed with predetermined binary masks
merge_thr = .7      # merging threshold, max correlation allowed
rf = 40             # half-size of the patches in pixels. e.g., if rf=40, patches are 80x80
stride_cnmf = 20    # amount of overlap between the patches in pixels
#                     (keep it at least large as gSiz, i.e 4 times the neuron size gSig)
tsub = 2            # downsampling factor in time for initialization,
#                     increase if you have memory problems
ssub = 1            # downsampling factor in space for initialization,
#                     increase if you have memory problems
#                     you can pass them here as boolean vectors
low_rank_background = None  # None leaves background of each patch intact,
#                     True performs global low-rank approximation if gnb>0
gnb = 0             # number of background components (rank) if positive,
#                     else exact ring model with following settings
#                         gnb= 0: Return background as b and W
#                         gnb=-1: Return full rank background B
#                         gnb<-1: Don't return background
nb_patch = 0        # number of background components (rank) per patch if gnb>0,
#                     else it is set automatically
min_corr = .1       # min peak value from correlation image
min_pnr = 0.5       # min peak to noise ration from PNR image



# parameters for source extraction and deconvolution
p = 1                       # order of the autoregressive system
gnb = 2                     # number of global background components
merge_thr = 0.85            # merging threshold, max correlation allowed
rf = 35                     # half-size of the patches in pixels. e.g., if rf=25, patches are 50x50
stride_cnmf = 6             # amount of overlap between the patches in pixels
K = 1                       # number of components per patch
gSig = [15, 15]               # expected half size of neurons in pixels
method_init = 'greedy_roi'  # initialization method (if analyzing dendritic data using 'sparse_nmf')
ssub = 1                    # spatial subsampling during initialization
tsub = 1                    # temporal subsampling during intialization


opts_dict = {'fnames': [file_path_ma],
            'fr': fr,
            'decay_time': decay_time,
            'strides': strides,
            'overlaps': overlaps,
            'max_shifts': max_shifts,
            'max_deviation_rigid': max_deviation_rigid,
            'pw_rigid': pw_rigid,
            'p': p,
            'nb': gnb,
            'rf': rf,
            'K': K,
            'stride': stride_cnmf,
            'expected_comps': 30,
            'rolling_sum': True,
            'only_init': True,
            'ssub': ssub,
            'tsub': tsub,
            'merge_thr': merge_thr

             }
opts = params.CNMFParams(params_dict=opts_dict)
opts.change_params(params_dict={'method_init': 'corr_pnr',  # use this for 1 photon
                                'K': K,
                                'gSig': gSig,
                                'gSiz': gSiz,
                                'merge_thr': merge_thr,
                                'p': p,
                                'tsub': tsub,
                                'ssub': ssub,
                                'rf': rf,
                                'stride': stride_cnmf,
                                'only_init': False,    # set it to True to run CNMF-E
                                'nb': gnb,
                                'nb_patch': nb_patch,
                                'method_deconvolution': 'oasis',       # could use 'cvxpy' alternatively
                                'low_rank_background': low_rank_background,
                                'update_background_components': True,  # sometimes setting to False improve the results
                                'min_corr': min_corr,
                                'min_pnr': min_pnr,
                                'normalize_init': False,               # just leave as is
                                #'center_psf': True,                    # leave as is for 1 photon
                                'del_duplicates': True})                # number of pixels to not consider in the borders)

#opts = params.CNMFParams(params_dict=parameters_motion_correction)
#mc = MotionCorrect(file_path, dview=dview, **opts.get_group('motion'))
motion_file = dir_path + 'memmap__d1_512_d2_512_d3_1_order_C_frames_4000_.mmap'
Yr, dims, T = cm.load_memmap(motion_file)
images = np.reshape(Yr.T, [T] + list(dims), order='C')


#%% restart cluster to clean up memory
cm.stop_server(dview=dview)
c, dview, n_processes = cm.cluster.setup_cluster(
    backend='local', n_processes=None, single_thread=False)

cnm = cnmf.CNMF(n_processes, params=opts, dview=dview)
cnm = cnm.fit(images)

#Cn = cm.local_correlations(images.transpose(1,2,0))
#Cn[np.isnan(Cn)] = 0
#cnm.estimates.plot_contours_nb(img=Cn)

figure, axes = plt.subplots()
axes.imshow(np.mean(images,axis=0),cmap = 'gray')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, np.shape(images[0,:,:]), 0.2, 'max')
for c in coordinates:
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    axes.plot(*v.T, c='b')
figure.savefig(figure_path + 'source_extraction_ma_2022.png')


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
figure.savefig(figure_path + 'source_extraction_ma_2022.png')
