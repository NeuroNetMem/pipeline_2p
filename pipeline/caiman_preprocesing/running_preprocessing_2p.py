
import caiman as cm
import numpy as np
import os
import psutil
import logging
import data_base as db
import matplotlib.pylab as plt
from preprocessing import run_cropper, run_motion_correction, run_alignment, run_source_extraction, cropping_interval, run_component_evaluation
from caiman.source_extraction.cnmf.cnmf import load_CNMF

figure_path = '/scratch/melisa/photon2_test/figures/'

mouse = 429419
year = 2022
month = 12
date = 1
example = 0


states = db.open_data_base()
selected_row = db.select(states, mouse , year,month,date)

n_processes = psutil.cpu_count()
#cm.cluster.stop_server()
# Start a new cluster
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,
                                                 single_thread=False)

if not os.path.isfile(eval(selected_row.iloc[0]['raw_output'])['main']):
    print('File does not exist')
#load tif file
#

parameters_cropping = cropping_interval()  # check whether it is better to do it like this or to use the functions get
### run cropper
for i in range(1):
    row = selected_row.iloc[i]
    updated_row = run_cropper(row, parameters_cropping)
    states = db.update_data_base(states, updated_row)
    db.save_analysis_states_database(states)

selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,0,0,0,0])

### general dictionary for motion correction 
parameters_motion_correction = {'pw_rigid': True, 'save_movie_rig': False,
                                'gSig_filt': (5, 5), 'max_shifts': (5, 5), 'niter_rig': 1,
                                'strides': (48, 48),
                                'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
                                'max_deviation_rigid': 10,
                                'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True, 'border_nan': 'copy'}

### run motion correction
for i in range(1):
    row = selected_rows.iloc[i]
    updated_row = run_motion_correction(row, parameters_motion_correction, dview)
    states = db.update_data_base(states, updated_row)
    db.save_analysis_states_database(states)

# parameters_alignment = {'make_template_from_trial': '1', 'gSig_filt': (5, 5), 'max_shifts': (5,5), 'niter_rig': 1,
#                         'strides': (48, 48), 'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
#                         'max_deviation_rigid': 10, 'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True,
#                         'border_nan': 'copy'}
#
# ### run alignment
# selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,0,0])
# new_selected_rows = run_alignment(selected_rows, parameters_alignment, dview)
#
# for i in range(2):
#     new_name = db.replace_at_index1(new_selected_rows.iloc[i].name, 7, new_selected_rows.iloc[i].name[5])
#     row_new = new_selected_rows.iloc[i].copy()
#     row_new.name = new_name
#     states = db.update_data_base(states, row_new)
#     db.save_analysis_states_database(states)


for i in [0]:
    movie = cm.load(eval(selected_row.iloc[i]['raw_output'])['main'])
    cropped_rows =  db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,0,0,0,0])
    movie_cropped = cm.load(eval(cropped_rows.iloc[i]['cropping_output'])['main'])
    selected_motion_corrected =  db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,0,0])
    movie_corrected = cm.load((eval(selected_motion_corrected.iloc[i]['motion_correction_output'])['main']))

    figure, axes = plt.subplots(1,3)
    axes[0].imshow(np.mean(movie,axis=0), cmap = 'gray')
    axes[0].set_title('Mean Image')
    axes[1].imshow(np.mean(movie_cropped,axis=0), cmap = 'gray')
    axes[1].set_title('Mean Image Cropped')
    axes[2].imshow(np.mean(movie_corrected,axis=0), cmap = 'gray')
    axes[2].set_title('Mean Image Corrected')
    figure.set_size_inches([20,10])
    figure.savefig(figure_path + 'motion_correction_'+str(mouse)+'_'+str(month)+'_'+str(date)+'.png')


corr1 = movie.local_correlations(eight_neighbours=True, swap_dim=False)
corr2 = movie_cropped.local_correlations(eight_neighbours=True, swap_dim=False)
corr_3 =  movie_corrected.local_correlations(eight_neighbours=True, swap_dim=False)

figure, axes = plt.subplots(1,3)

axes[0].imshow(corr1, cmap = 'gray')
axes[1].imshow(corr2, cmap = 'gray')
axes[2].imshow(corr_3, cmap = 'gray')

axes[0].set_title('Original Movie')
axes[1].set_title('Cropped Movie')
axes[2].set_title('MC Movie')

figure.suptitle('CORRELATION IMAGE')
figure.savefig(figure_path + 'correlation_image_'+str(mouse)+'_'+str(i)+'.png')
#
# figure, axes = plt.subplots(1,2)
#
# corr1_thr = np.ones_like(corr_3)
# max_corr1 = np.max(corr_3)
# corr1_thr[np.where(corr_3 > 0.7 * max_corr1)] = 0
#
#
# axes[0].imshow(corr_3, cmap = 'gray')
# axes[1].imshow(corr1_thr, cmap = 'gray')
#
# axes[0].set_title('Original Movie')
# axes[1].set_title('Cropped Movie')
#
# figure.suptitle('CORRELATION IMAGE ')
# figure.savefig(figure_path + 'correlation_threshold_Janek2022.png')
#


gSig = 5
gSiz = 2 * gSig + 1
min_corr = 0.25
min_pnr = 1.0
##trial_wise_parameters
parameters_source_extraction = {'fr': 30, 'decay_time': 0.1,
                                'min_corr': min_corr,
                                'min_pnr': min_pnr, 'p': 1, 'K': 5, 'gSig': (gSig,gSig),
                                'gSiz': (gSiz,gSiz),
                                'merge_thr': 0.7, 'rf': 60, 'stride': 30, 'tsub': 1, 'ssub': 2, 'p_tsub': 1,
                                'p_ssub': 2, 'low_rank_background': None, 'nb': 0, 'nb_patch': 0,
                                'ssub_B': 2,
                                'init_iter': 2, 'ring_size_factor': 1.4, 'method_init': 'corr_pnr',
                                'method_deconvolution': 'oasis', 'update_background_components': True,
                                'center_psf': True, 'border_pix': 0, 'normalize_init': False,
                                'del_duplicates': True, 'only_init': True}


### run source extraction
selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, example = 0, analysis_version = [1,1,0,0,0])

mouse_row_new =run_source_extraction(selected_rows.iloc[0], parameters_source_extraction, states, dview, multiple_files= False)
states = db.update_data_base(states, mouse_row_new)
db.save_analysis_states_database(states)

### run component evaluation evaluation
dview.terminate()
parameters_component_evaluation = {'min_SNR': 3.5,
                                   'rval_thr': 0.8,
                                   'use_cnn': False}

selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,1,0])
mouse_row_new = run_component_evaluation(selected_rows.iloc[0], parameters_component_evaluation, states, multiple_files= False)
states = db.update_data_base(states, mouse_row_new)
db.save_analysis_states_database(states)

##### ADD HERE THE REGISTRATION#####

#################################33

#### printing and ploting

selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date,example=0, analysis_version = [1,1,0,1,0])
mouse_row_new = selected_rows.iloc[0]
cnm = load_CNMF(eval(mouse_row_new['source_extraction_output'])['main'])
output_source_extraction = eval(mouse_row_new.loc['source_extraction_output'])
corr_path = output_source_extraction['meta']['corr']['main']
cn_filter = np.load(corr_path)

selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date,example=0, analysis_version = [1,1,0,0,0])
row_local = selected_rows.iloc[0]
# Load memmory mappable input file
input_memmap_file = eval(row_local.loc['motion_correction_output'])['main']
if os.path.isfile(input_memmap_file):
    Yr, dims, T = cm.load_memmap(input_memmap_file)
    #        logging.debug(f'{index} Loaded movie. dims = {dims}, T = {T}.')
    images = Yr.T.reshape((T,) + dims, order='F')

import caiman as cm
# #
# cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], swap_dim=False)
# figure, axes = plt.subplots(2, 2)
# axes[0, 0].imshow(corr1, cmap='gray')
# axes[0, 1].imshow(corr2, cmap='gray')
# axes[1, 0].imshow(corr_3, cmap='gray')
# axes[1, 1].imshow(cn_filter, cmap='gray')
#
# axes[0, 0].set_title('Original Movie')
# axes[0, 1].set_title('Cropped Movie')
# axes[1, 0].set_title('MC Movie')
# axes[1, 1].set_title('CN filter')
#
# figure.suptitle('CORRELATION IMAGE')
# figure.savefig(figure_path + 'correlation_gSig_Janek2022.png')
#
# for gSig in range(0,25):
#     print(gSig)
#     gSiz = 4 * gSig + 1
#     min_corr = 1
#     min_pnr = 10
#     ##trial_wise_parameters
#     parameters_source_extraction = {'fr': 34, 'decay_time': 0.1,
#                                     'min_corr': min_corr,
#                                     'min_pnr': min_pnr, 'p': 1, 'K': None, 'gSig': (gSig,gSig),
#                                     'gSiz': (gSiz,gSiz),
#                                     'merge_thr': 0.7, 'rf': 60, 'stride': 30, 'tsub': 1, 'ssub': 2, 'p_tsub': 1,
#                                     'p_ssub': 2, 'low_rank_background': None, 'nb': 0, 'nb_patch': 0,
#                                     'ssub_B': 2,
#                                     'init_iter': 2, 'ring_size_factor': 1.4, 'method_init': 'corr_pnr',
#                                     'method_deconvolution': 'oasis', 'update_background_components': True,
#                                     'center_psf': True, 'border_pix': 0, 'normalize_init': False,
#                                     'del_duplicates': True, 'only_init': True}
#
#     cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=parameters_source_extraction['gSig'][0], swap_dim=False)
#
#     figure, axes = plt.subplots(2,2)
#
#     axes[0,0].imshow(corr1, cmap = 'gray')
#     axes[0,1].imshow(corr2, cmap = 'gray')
#     axes[1,0].imshow(corr_3, cmap = 'gray')
#     axes[1,1].imshow(cn_filter, cmap = 'gray')
#
#     axes[0,0].set_title('Original Movie')
#     axes[0,1].set_title('Cropped Movie')
#     axes[1,0].set_title('MC Movie')
#     axes[1,1].set_title('CN filter')
#
#     figure.suptitle('CORRELATION IMAGE')
#     figure.savefig(figure_path + 'correlation_gSig_'+f'{gSig}' +'_Janek2022.png')
#
i=0
figure, axes = plt.subplots()
#axes.imshow(np.mean(movie_corrected,axis = 0),cmap = 'gray')
axes.imshow(cn_filter,cmap = 'gray')

coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape, 0.2, 'max')
counter = 0
for c in coordinates:
    if counter > 0:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        #axes.plot(*v.T, c='r')
        axes.plot(*v.T)
    counter = counter + 1
figure.savefig(figure_path + 'source_extraction_gSig_'+str(5)+'_corr_'+str(min_corr)+'_pnr_'+str(min_pnr)+'_'+str(mouse)+'_'+str(i)+'.png')

figure, axes = plt.subplots()
C_0 = cnm.estimates.C.copy()
#C_0[0] += C_0[0].min()
for i in range(0, len(C_0)):
    #C_0[i] += C_0[i].min() + C_0[:i].max()
    axes.plot(C_0[i,:]/np.max(C_0[i,:])+i)
    #axes.plot(C_0[i]/np.max(C_0[i])+i, c = 'k')
figure.set_size_inches([15., 25])
figure.savefig(figure_path + 'source_extraction_traces_gSig_'+str(5)+'_corr_'+str(min_corr)+'_pnr_'+str(min_pnr)+'_'+str(mouse)+'_'+str(i)+'.png')


########################3########################3########################3########################3########################3########################3
########################3########################3########################3########################3########################3
########################3########################3########################3########################3########################3


cnm = load_CNMF(eval(selected_rows.iloc[0]['component_evaluation_output'])['main'])
output_source_extraction = eval(selected_rows.iloc[0]['source_extraction_output'])
corr_path = output_source_extraction['meta']['corr']['main']
cn_filter = np.load(corr_path)
#cn_filter = mean_image

figure, axes = plt.subplots(1,2)
axes[0].imshow(cn_filter,cmap = 'gray')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A[:,cnm.estimates.idx_components], cn_filter.shape, 0.2, 'max')
counter = 0
for c in coordinates:
    if counter > 0:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes[0].plot(*v.T, c='b')
    counter = counter + 1

axes[1].imshow(cn_filter,cmap = 'gray')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A[:,cnm.estimates.idx_components_bad], cn_filter.shape, 0.2, 'max')
counter = 0
for c in coordinates:
    if counter > 0:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes[1].plot(*v.T, c='r')
    counter = counter + 1
axes[0].set_title('Selected Cells', fontsize = 15)
axes[1].set_title('Unselected Cells', fontsize = 15)
figure.savefig(figure_path + 'source_extraction_havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_complete_mean.png')

figure, axes = plt.subplots()
C_0 = cnm.estimates.C.copy()
C_0[0] += C_0[0,:].min()
for i in range(1, len(C_0)):
    C_0[i,:] += np.min(C_0[i,:]) + np.max(C_0[:i,:])
    if i in cnm.estimates.idx_components :
        #axes.plot(C_0[i]/np.max(C_0[i])+i, c = 'b')
        axes.plot(C_0[i,:] + np.max(C_0[:i,:]) , c = 'b')
    if i in cnm.estimates.idx_components_bad:
        #axes.plot(C_0[i]/np.max(C_0[i])+i, c = 'r')
        axes.plot(C_0[i,:] + np.max(C_0[:i,:]) , c = 'r')
figure.set_size_inches([15., 25])
figure.savefig(figure_path + 'source_extraction_havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces2_complete.png')


data_path = '/scratch/melisa/photon2_test/data/'
file_name = 'havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces_accepted_complete.npy'
np.save(data_path + file_name , cnm.estimates.C[cnm.estimates.idx_components,:])
file_name = 'havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces_rejected_complete.npy'
np.save(data_path + file_name , cnm.estimates.C[cnm.estimates.idx_components_bad,:])


#######################################################################################################################
#######################################################################################################################
#######################################################################################################################

#plot masked trace, no model, original data
import math
import matplotlib
from scipy.ndimage import gaussian_filter1d
selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,1,0])

#cnm = load_CNMF(eval(selected_rows.iloc[0]['component_evaluation_output'])['main'])
cnm = load_CNMF(eval(selected_rows.iloc[0]['source_extraction_output'])['main'])
output_source_extraction = eval(selected_rows.iloc[0]['source_extraction_output'])
corr_path = output_source_extraction['meta']['corr']['main']
cn_filter = np.load(corr_path)

#coordinates = cm.utils.visualization.get_contours(cnm.estimates.A[:,cnm.estimates.idx_components], cn_filter.shape, 0.2, 'max')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape, 0.2, 'max')

temporal_mean = np.mean(np.mean(movie_corrected,axis=1),axis=1)

figure, axes = plt.subplots()
figure1 , axes1 = plt.subplots()
general_mask = np.zeros_like(cn_filter)
cell_counter = 0
#new_traces = np.zeros_like(cnm.estimates.C[cnm.estimates.idx_components,:])
new_traces = np.zeros_like(cnm.estimates.C)

for c in coordinates:
    if cell_counter >= 0:
        v = c['coordinates']
        y_pixel_nos = v[:,1].copy()
        x_pixel_nos = v[:,0].copy()
        temp_list = []
        for a, b in zip(x_pixel_nos, y_pixel_nos):
            if ~np.isnan(a) and ~np.isnan(b):
                temp_list.append([a, b])
        polygon = np.array(temp_list)
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0])+1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1])+1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))
        path = matplotlib.path.Path(polygon)
        mask = path.contains_points(points)
        mask.shape = xv.shape
        #print(mask.shape)
        complete_mask = np.zeros_like(cn_filter)

        complete_mask[yv[0,0]:yv[-1,0]+1,xv[0,0]:xv[0,-1]+1] = mask
        index = np.where(complete_mask == True)
        #print(index)
        axes1.imshow(general_mask + complete_mask)

        general_mask = general_mask + complete_mask
        mean_trace = np.zeros((movie_corrected.shape[0],))
        counter = 0
        for i in index[0]:
            for j in index[1]:
                mean_trace+=movie_corrected[:,i,j]
                counter+=1
        mean_trace/=counter
        mean_trace = mean_trace - temporal_mean
        mean_trace = (mean_trace - np.min(mean_trace)) / np.max( mean_trace - np.min(mean_trace))
        mean_trace_filtered = gaussian_filter1d(mean_trace,0.1)
        mean_trace_filtered = (mean_trace_filtered - np.min(mean_trace_filtered)) / (np.max(mean_trace_filtered - np.min(mean_trace_filtered)))
        axes.plot(mean_trace_filtered + cell_counter)
        new_traces[cell_counter,:] = mean_trace_filtered
        cell_counter = cell_counter + 1

        #axes.plot((movie_corrected[:,50,100]-np.min(movie_corrected[:,50,100]))/np.max(movie_corrected[:,50,100])+cell_counter)

#axes.set_xlim([20000,30000])
figure.set_size_inches([40, 25])
#figure.savefig(figure_path + 'source_extraction_havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces2_complete_raw_sigma_5.png')
#figure.savefig(figure_path + 'source_extraction_gSig_5_mincorr_0.5_minpnr_5_traces_Janek2022_raw_sigma_1.png')
figure.savefig(figure_path + 'source_extraction_tracesRAW_gSig_' + str(5) + '_corr_' + str(min_corr) + '_pnr_' + str(3.5) + '_' + str(mouse) + '_' + str(i) + '.png')

data_path = '/scratch/melisa/photon2_test/data/'
file_name = 'Janek2022_signma3_LEICA.npy'
np.save(data_path + file_name , new_traces)


figure, axes = plt.subplots()

for i in range(30):
    a = (new_traces[i,:]-np.min(new_traces[i,:]))/(np.max(new_traces[i,:])-np.min(new_traces[i,:]))
    b = (C_0[i,:]-np.min(C_0[i,:]))/(np.max(C_0[i,:])-np.min(C_0[i,:]))

    axes.plot(a + i, c = 'k')
    axes.plot(b + i, c='b')
figure.set_size_inches([40, 25])

figure.savefig(figure_path + 'source_extraction_tracesRAW-MODEL_gSig_' + str(5) + '_corr_' + str(min_corr) + '_pnr_' + str(3.5) + '_' + str(mouse) + '_' + str(i) + '.png')


############################################################################3
### RING MODEL FOR BACKGROUND SUSTRACTION

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = dist_from_center <= radius
    return mask

temporal_mean = np.mean(np.mean(movie_corrected,axis=1),axis=1)

figure, axes = plt.subplots()
figure1 , axes1 = plt.subplots()
general_mask = np.zeros_like(cn_filter)
general_mask_background = np.zeros_like(cn_filter)

cell_counter = 0
new_traces = np.zeros_like(cnm.estimates.C[cnm.estimates.idx_components,:])
for c in coordinates:
    if cell_counter <150:
        v = c['coordinates']

        ### for cell pixels
        y_pixel_nos = v[:,1].copy()
        x_pixel_nos = v[:,0].copy()
        temp_list = []
        for a, b in zip(x_pixel_nos, y_pixel_nos):
            if ~np.isnan(a) and ~np.isnan(b):
                temp_list.append([a, b])
        polygon = np.array(temp_list)
        left = np.min(polygon, axis=0)
        right = np.max(polygon, axis=0)
        x = np.arange(math.ceil(left[0]), math.floor(right[0])+1)
        y = np.arange(math.ceil(left[1]), math.floor(right[1])+1)
        xv, yv = np.meshgrid(x, y, indexing='xy')
        points = np.hstack((xv.reshape((-1,1)), yv.reshape((-1,1))))
        path = matplotlib.path.Path(polygon)
        mask = path.contains_points(points)
        mask.shape = xv.shape

        binary_mask = np.zeros((mask.shape[0],mask.shape[1]))
        binary_mask[np.where(mask == True)] = 1
        complete_mask = np.zeros_like(cn_filter)
        complete_mask[yv[0,0]:yv[-1,0]+1,xv[0,0]:xv[0,-1]+1] = binary_mask
        index = np.where(complete_mask == 1)
        #print(index)

        mean_trace = np.zeros((movie_corrected.shape[0],))
        counter = 0
        for i in index[0]:
            for j in index[1]:
                mean_trace+=movie_corrected[:,i,j]
                counter+=1
        mean_trace/=counter

        # ### for ring model
        center_x = np.nanmean(x_pixel_nos)
        center_y = np.nanmean(y_pixel_nos)
        diameter = np.max([np.nanmax(x_pixel_nos)-np.nanmin(x_pixel_nos),np.nanmax(y_pixel_nos)-np.nanmin(y_pixel_nos)])
        circular_mask = create_circular_mask(complete_mask.shape[0], complete_mask.shape[1], center=[center_x, center_y],
                             radius=diameter*1)
        binary_circular_mask =  np.zeros_like(cn_filter)
        binary_circular_mask[np.where(circular_mask == True)] = 2
        ring_mask = binary_circular_mask - complete_mask
        circular_coordinates = np.where(ring_mask)

        circular_mask2 = create_circular_mask(complete_mask.shape[0], complete_mask.shape[1], center=[center_x, center_y],
                             radius=diameter*1.5)
        binary_circular_mask2 =  np.zeros_like(cn_filter)
        binary_circular_mask2[np.where(circular_mask2 == True)] = 3
        ring_mask2 = binary_circular_mask2 - binary_circular_mask
        circular_coordinates = np.where(ring_mask2)

        general_mask = general_mask + ring_mask2

        mean_trace_ring = np.zeros((movie_corrected.shape[0],))
        counter = 0
        for i in circular_coordinates[0]:
            for j in circular_coordinates[1]:
                mean_trace_ring+=movie_corrected[:,i,j]
                counter+=1
        mean_trace_ring/=counter

        mean_trace = (mean_trace - temporal_mean) - (mean_trace_ring - temporal_mean)
        #mean_trace = mean_trace_ring - temporal_mean
        #mean_trace = mean_trace - temporal_mean
        #mean_trace = (mean_trace - np.min(mean_trace)) / np.max( mean_trace - np.min(mean_trace))
        mean_trace_filtered = gaussian_filter1d(mean_trace,5)
        mean_trace_filtered = (mean_trace_filtered - np.min(mean_trace_filtered)) / (np.max(mean_trace_filtered - np.min(mean_trace_filtered)))
        #axes.plot(mean_trace_filtered + cell_counter)
        new_traces[cell_counter,:] = mean_trace_filtered
    cell_counter = cell_counter + 1

        #axes.plot((movie_corrected[:,50,100]-np.min(movie_corrected[:,50,100]))/np.max(movie_corrected[:,50,100])+cell_counter)
#axes.set_xlim([20000,30000])
figure.set_size_inches([40, 25])
figure.savefig(figure_path + 'source_extraction_havaianna_SE_gSig_4_mincorr_0.312_minpnr_2.3_traces_CE_snr_4_pcc_0.5_traces2_complete_raw_sigma_5_ring.png')


data_path = '/scratch/melisa/photon2_test/data/'
file_name = 'havaianna_traces_sigma_5_ring_model_signal.npy'
np.save(data_path + file_name , new_traces)


 