
import caiman as cm
import numpy as np
import os
import psutil
import logging
import data_base as db
import matplotlib.pylab as plt
from preprocessing import run_cropper, run_motion_correction, run_alignment, run_source_extraction, cropping_interval, run_component_evaluation
from caiman.source_extraction.cnmf.cnmf import load_CNMF

n_processes = psutil.cpu_count()
#cm.cluster.stop_server()
# Start a new cluster
c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                 n_processes=n_processes,
                                                 single_thread=False)

figure_path = '/scratch/melisa/photon2_test/figures/'
file_path = '/ceph/imaging1/arie/unidirectional/20221020_433557/'
file_unidirectional = '433557_unidirectional_00001.tif'
file_bidirectional = '433557_bidirectional_00001.tif'
#
# movie0= cm.load(file_path + file_unidirectional)
# movie1 = cm.load(file_path + file_bidirectional)
#
# slices_title = ['MEAN Unidirectional','MEAN Bidirectional', ' One Frame Unidiretional', ' One Frame Bidirectional']
# figure, axes = plt.subplots(2,2)
# axes[0,0].imshow(np.mean(movie0,axis = 0), cmap = 'gray')
# axes[0,1].imshow(np.mean(movie1,axis = 0), cmap = 'gray')
#
# axes[1,0].imshow(movie0[0,:,:], cmap = 'gray')
# axes[1,1].imshow(movie1[0,:,:], cmap = 'gray')
#
# for i in range(2):
#     for j in range(2):
#         axes[i,j].set_xticklabels([])
#         axes[i,j].set_yticklabels([])
#         axes[i,j].set_aspect('equal')
#         axes[i,j].set_title(slices_title[i*2+j])
#
# figure.savefig(figure_path + 'unidirectional_vs_bidirectional.png')
# plt.show()


mouse = 433557
year = 2022
month = 10
date = 20
example = 0

states = db.open_data_base()
selected_row = db.select(states, mouse , year,month,date)

parameters_cropping = cropping_interval()  # check whether it is better to do it like this or to use the functions get
### run cropper
for i in range(2):
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
for i in range(2):
    row = selected_rows.iloc[i]
    updated_row = run_motion_correction(row, parameters_motion_correction, dview)
    states = db.update_data_base(states, updated_row)
    db.save_analysis_states_database(states)

parameters_alignment = {'make_template_from_trial': '1', 'gSig_filt': (5, 5), 'max_shifts': (5,5), 'niter_rig': 1,
                        'strides': (48, 48), 'overlaps': (96, 96), 'upsample_factor_grid': 2, 'num_frames_split': 80,
                        'max_deviation_rigid': 10, 'shifts_opencv': True, 'use_cuda': False, 'nonneg_movie': True,
                        'border_nan': 'copy'}

### run alignment
selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,0,0])
new_selected_rows = run_alignment(selected_rows, parameters_alignment, dview)

for i in range(2):
    new_name = db.replace_at_index1(new_selected_rows.iloc[i].name, 7, new_selected_rows.iloc[i].name[5])
    row_new = new_selected_rows.iloc[i].copy()
    row_new.name = new_name
    states = db.update_data_base(states, row_new)
    db.save_analysis_states_database(states)

figure, axes = plt.subplots(2,3)
mode_label = ['Unidirectional','Bidirectional']
for i in range(2):
    cropped_rows =  db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,0,0,0,0])
    movie_cropped = cm.load(eval(cropped_rows.iloc[i]['cropping_output'])['main'])
    selected_motion_corrected =  db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,0,0])
    movie_corrected = cm.load((eval(selected_motion_corrected.iloc[i]['motion_correction_output'])['main']))
    axes[i,0].imshow(np.mean(movie_cropped,axis=0), cmap = 'gray')
    axes[i,0].set_title('Mean' + mode_label[i], fontsize = 15)
    axes[i,1].imshow(np.mean(movie_corrected,axis=0), cmap = 'gray')
    axes[i,1].set_title('MeanCorrected' + mode_label[i], fontsize = 15)
    selected_alignment =  db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,1,0,0])
    movie_aligned = cm.load((eval(selected_motion_corrected.iloc[i]['motion_correction_output'])['main']))
    axes[i,2].imshow(np.mean(movie_aligned,axis=0), cmap = 'gray')
    axes[i,2].set_title('MeanAligned' + mode_label[i], fontsize = 15)
for i in range(2):
    for j in range(3):
        axes[i,j].set_xticklabels([])
        axes[i,j].set_yticklabels([])
        axes[i,j].set_aspect('equal')
figure.set_size_inches([10,10])
figure.savefig(figure_path + 'motion_correction_uni_vs_bi_aligned.png')



gSig = 5
gSiz = 2 * gSig + 1
min_corr = 0.5
min_pnr = 3
##trial_wise_parameters
parameters_source_extraction = {'fr': 15, 'decay_time': 0.1,
                                'min_corr': min_corr,
                                'min_pnr': min_pnr, 'p': 1, 'K': None, 'gSig': (gSig,gSig),
                                'gSiz': (gSiz,gSiz),
                                'merge_thr': 0.7, 'rf': 60, 'stride': 30, 'tsub': 1, 'ssub': 2, 'p_tsub': 1,
                                'p_ssub': 2, 'low_rank_background': None, 'nb': 0, 'nb_patch': 0,
                                'ssub_B': 2,
                                'init_iter': 2, 'ring_size_factor': 1.4, 'method_init': 'corr_pnr',
                                'method_deconvolution': 'oasis', 'update_background_components': True,
                                'center_psf': True, 'border_pix': 0, 'normalize_init': False,
                                'del_duplicates': True, 'only_init': True}

selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,1,0,0])
mouse_row_new = run_source_extraction(selected_rows.iloc[0], parameters_source_extraction, states, dview,
                                      multiple_files=True)
states = db.update_data_base(states, mouse_row_new)
db.save_analysis_states_database(states)

### run source extraction
selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,0,0])

for i in range(2):
    mouse_row_new =run_source_extraction(selected_rows.iloc[i], parameters_source_extraction, states, dview, multiple_files= False)
    states = db.update_data_base(states, mouse_row_new)
    db.save_analysis_states_database(states)

import caiman as cm

selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,1,0])

figure, axes = plt.subplots(1)
color_list = ['r','b']
for i in range(2):
    mouse_row_new = selected_rows.iloc[i]
    cnm = load_CNMF(eval(mouse_row_new['source_extraction_output'])['main'])
    output_source_extraction = eval(mouse_row_new.loc['source_extraction_output'])
    corr_path = output_source_extraction['meta']['corr']['main']
    cn_filter = np.load(corr_path)
    print(cn_filter.shape)

    selected_motion_corrected =  db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,0,0])
    movie_corrected = cm.load((eval(selected_motion_corrected.iloc[i]['motion_correction_output'])['main']))
    if i == 0:
        axes.imshow(np.mean(movie_corrected,axis = 0),cmap = 'gray')
    #axes[i].imshow(cn_filter,cmap = 'gray')

    coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape, 0.2, 'max')
    counter = 0
    for c in coordinates:
        if counter > 0:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            #axes.plot(*v.T, c='r')
            axes.plot(*v.T, c = color_list[i])
        counter = counter + 1
    #axes.set_title(mode_label[i] + ' count:'+ str(cnm.estimates.A.shape[1]), fontsize = 15)
    axes.set_xticklabels([])
    axes.set_yticklabels([])
    axes.set_aspect('equal')
figure.savefig(figure_path + 'source_extraction_gSig_'+str(5)+'_corr_'+str(min_corr)+'_pnr_'+str(min_pnr)+'_'+str(mouse)+'_'+str(mouse)+'_uni_vs_bi_together.png')


selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,1,1,0])
figure, axes = plt.subplots()
mouse_row_new = selected_rows.iloc[0]
cnm = load_CNMF(eval(mouse_row_new['source_extraction_output'])['main'])
output_source_extraction = eval(mouse_row_new.loc['source_extraction_output'])
corr_path = output_source_extraction['meta']['corr']['main']
cn_filter = np.load(corr_path)
print(cn_filter.shape)

selected_alignment =  db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,1,0,0])
movie_corrected = cm.load((eval(selected_alignment.iloc[0]['motion_correction_output'])['main']))
axes.imshow(np.mean(movie_corrected,axis = 0),cmap = 'gray')

coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape, 0.2, 'max')
counter = 0
for c in coordinates:
    v = c['coordinates']
    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                    np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
    axes.plot(*v.T, c = 'b')
axes.set_xticklabels([])
axes.set_yticklabels([])
axes.set_aspect('equal')
figure.savefig(figure_path + 'source_extraction_aligned_gSig_'+str(5)+'_corr_'+str(min_corr)+'_pnr_'+str(min_pnr)+'_'+str(mouse)+'_'+str(mouse)+'_uni_vs_bi_together.png')

time_uni = np.arange(0,movie_cropped.shape[0])/15
time_bi = np.arange(0,movie_cropped.shape[0])/30
time = []
time.append(time_uni)
time.append(time_bi)

figure, axes = plt.subplots(1,2)
mouse_row_new = selected_rows.iloc[0]
cnm = load_CNMF(eval(mouse_row_new['source_extraction_output'])['main'])
count = 0
for i in range(2):
    C_0 = cnm.estimates.C.copy()
    #C_0[0] += C_0[0].min()
    for j in range(0, len(C_0)):
        axes[i].plot(time[i],C_0[j,count : count + movie_cropped.shape[0]]/np.max(C_0[j,count : count + movie_cropped.shape[0]])+j, c = color_list[i])
    count = count + movie_cropped.shape[0]
    axes[i].set_title(mode_label[i], fontsize = 20)
figure.set_size_inches([15., 25])
figure.savefig(figure_path + 'source_extraction_traces_aligned_gSig_'+str(5)+'_corr_'+str(min_corr)+'_pnr_'+str(min_pnr)+'_'+str(mouse)+'_'+str(mouse)+'_uni_vs_bi2.png')


time_uni = np.arange(0,movie_cropped.shape[0])/15
time_bi = np.arange(0,movie_cropped.shape[0])/30
time = []
time.append(time_uni)
time.append(time_bi)

figure, axes = plt.subplots(1,2)
for i in range(2):
    mouse_row_new = selected_rows.iloc[i]
    cnm = load_CNMF(eval(mouse_row_new['source_extraction_output'])['main'])
    C_0 = cnm.estimates.C.copy()
    #C_0[0] += C_0[0].min()
    for j in range(0, len(C_0)):
        #C_0[i] += C_0[i].min() + C_0[:i].max()
        axes[i].plot(time[i],C_0[j,:]/np.max(C_0[j,:])+j)
        #axes.plot(C_0[i]/np.max(C_0[i])+i, c = 'k')
    axes[i].set_title(mode_label[i], fontsize = 20)
figure.set_size_inches([15., 25])
figure.savefig(figure_path + 'source_extraction_traces_gSig_'+str(5)+'_corr_'+str(min_corr)+'_pnr_'+str(min_pnr)+'_'+str(mouse)+'_'+str(mouse)+'_uni_vs_bi2.png')


#plot masked trace, no model, original data.. cell shapes taken from bidirectional data

import math
import matplotlib
from scipy.ndimage import gaussian_filter1d

selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,0,1,0])

cnm = load_CNMF(eval(selected_rows.iloc[0]['source_extraction_output'])['main'])
output_source_extraction = eval(selected_rows.iloc[0]['source_extraction_output'])
corr_path = output_source_extraction['meta']['corr']['main']
cn_filter = np.load(corr_path)

#coordinates = cm.utils.visualization.get_contours(cnm.estimates.A[:,cnm.estimates.idx_components], cn_filter.shape, 0.2, 'max')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape, 0.2, 'max')

selected_motion_corrected = db.select(states, mouse=mouse, year=year, month=month, date=date,
                                      analysis_version=[1, 1, 0, 0, 0])
movie_corrected_uni = cm.load((eval(selected_motion_corrected.iloc[0]['motion_correction_output'])['main']))
movie_corrected_bi = cm.load((eval(selected_motion_corrected.iloc[1]['motion_correction_output'])['main']))
temporal_mean_uni = np.mean(np.mean(movie_corrected_uni,axis=1),axis=1)
temporal_mean_bi = np.mean(np.mean(movie_corrected_bi,axis=1),axis=1)

movie_corrected_alignment = cm.load((eval(selected_alignment.iloc[0]['motion_correction_output'])['main']))
temporal_mean_alignment = np.mean(np.mean(movie_corrected_alignment,axis=1),axis=1)

general_mask = np.zeros_like(cn_filter)
cell_counter = 0
#new_traces = np.zeros_like(cnm.estimates.C[cnm.estimates.idx_components,:])
new_traces_uni = np.zeros_like(cnm.estimates.C)
new_traces_bi = np.zeros_like(cnm.estimates.C)
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
        general_mask = general_mask + complete_mask

        mean_trace_uni = np.zeros((movie_corrected_uni.shape[0],))
        mean_trace_bi = np.zeros((movie_corrected_uni.shape[0],))
        mean_trace_alignment =  np.zeros((movie_aligned.shape[0],))
        counter = 0
        for i in index[0]:
            for j in index[1]:
                mean_trace_uni+=movie_corrected_uni[:,i,j]
                mean_trace_bi+=movie_corrected_bi[:,i,j]
                mean_trace_alignment+=movie_aligned[:,i,j]

                counter+=1
        mean_trace_uni/=counter
        mean_trace_bi/=counter
        mean_trace_alignment/=counter

        mean_trace_uni = mean_trace_uni - temporal_mean_uni
        mean_trace_bi = mean_trace_bi - temporal_mean_bi
        mean_trace_alignment = mean_trace_alignment - temporal_mean_al

        mean_trace_uni = (mean_trace_uni - np.min(mean_trace_uni)) / np.max( mean_trace_uni - np.min(mean_trace_uni))
        mean_trace_bi = (mean_trace_bi - np.min(mean_trace_bi)) / np.max( mean_trace_bi - np.min(mean_trace_bi))

        mean_trace_filtered_uni = gaussian_filter1d(mean_trace_uni,0.1)
        mean_trace_filtered_bi = gaussian_filter1d(mean_trace_bi,0.1)

        mean_trace_filtered_uni = (mean_trace_filtered_uni - np.min(mean_trace_filtered_uni)) / (np.max(mean_trace_filtered_uni - np.min(mean_trace_filtered_uni)))
        mean_trace_filtered_bi = (mean_trace_filtered_bi- np.min(mean_trace_filtered_bi)) / (np.max(mean_trace_filtered_bi - np.min(mean_trace_filtered_bi)))

        new_traces_uni[cell_counter,:] = mean_trace_filtered_uni
        new_traces_bi[cell_counter,:] = mean_trace_filtered_bi
        cell_counter = cell_counter + 1


movie_corrected_alignment = cm.load((eval(selected_alignment.iloc[0]['alignment_output'])['main']))
temporal_mean_alignment = np.mean(np.mean(movie_corrected_alignment,axis=1),axis=1)

general_mask = np.zeros_like(cn_filter)
cell_counter = 0
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
        general_mask = general_mask + complete_mask

        mean_trace_alignment =  np.zeros((movie_corrected_alignment.shape[0],))
        counter = 0
        for i in index[0]:
            for j in index[1]:
                mean_trace_alignment+=movie_corrected_alignment[:,i,j]
                counter+=1

        mean_trace_alignment/=counter
        mean_trace_alignment = mean_trace_alignment - temporal_mean_alignment

        mean_trace_uni = (mean_trace_alignment - np.min(mean_trace_alignment)) / np.max( mean_trace_alignment - np.min(mean_trace_alignment))
        mean_trace_filtered_alignment = gaussian_filter1d(mean_trace_alignment,0.1)
        mean_trace_filtered_alignment = (mean_trace_filtered_alignment - np.min(mean_trace_filtered_alignment)) / (np.max(mean_trace_filtered_alignment - np.min(mean_trace_filtered_alignment)))

        new_traces[cell_counter,:] = mean_trace_filtered_alignment
        cell_counter = cell_counter + 1


figure, axes = plt.subplots()
for i in range(50):
    a = (new_traces_uni[i,:]-np.min(new_traces_uni[i,:]))/(np.max(new_traces_uni[i,:])-np.min(new_traces_uni[i,:]))
    b = (new_traces_bi[i,:]-np.min(new_traces_bi[i,:]))/(np.max(new_traces_bi[i,:])-np.min(new_traces_bi[i,:]))

    #b = (C_0[i,:]-np.min(C_0[i,:]))/(np.max(C_0[i,:])-np.min(C_0[i,:]))

    axes.plot(time_uni,a + i, c = 'r')
    axes.plot(time_bi,b + i, c='b')
figure.set_size_inches([40, 25])
figure.savefig(figure_path + 'source_extraction_tracesRAW-MODEL_gSig_' + str(5) + '_corr_' + str(min_corr) + '_pnr_' + str(min_pnr) + '_' + str(mouse) + '_uni_vs_bi2.png')

figure, axes = plt.subplots()
for i in range(50):
    a = (new_traces[i,:]-np.min(new_traces[i,:]))/(np.max(new_traces[i,:])-np.min(new_traces[i,:]))

    axes.plot(time_uni,a[0:9000] + i, c = 'r')
    axes.plot(time_bi,a[9000:18000] + i, c='b')
figure.set_size_inches([40, 25])
figure.savefig(figure_path + 'source_extraction_tracesRAW-MODEL_aligned_gSig_' + str(5) + '_corr_' + str(min_corr) + '_pnr_' + str(min_pnr) + '_' + str(mouse) + '_uni_vs_bi2.png')



mean_activity_uni = np.mean(new_traces_uni[:,0:4500], axis = 1)
std_activity_uni = np.std(new_traces_uni[:,0:4500] , axis = 1)#/np.sqrt(new_traces_uni.shape[1])

mean_activity_uni2 = np.mean(new_traces_uni[:,4500:9000], axis = 1)
std_activity_uni2 = np.std(new_traces_uni[:,4500:9000] , axis = 1)#/np.sqrt(new_traces_uni.shape[1])

mean_activity_bi = np.mean(new_traces_bi , axis = 1)
std_activity_bi = np.std(new_traces_bi , axis = 1)#/np.sqrt(new_traces_bi.shape[1])

figure, axes = plt.subplots()
axes.errorbar(mean_activity_bi, mean_activity_uni, xerr = std_activity_bi, yerr =std_activity_uni, fmt = 'o', c = 'k')
#axes.errorbar(mean_activity_bi, mean_activity_uni2, xerr = std_activity_bi, yerr =std_activity_uni2, fmt = 'o', c = 'b')
axes.errorbar(np.mean(mean_activity_bi),np.mean(mean_activity_uni), xerr = np.std(mean_activity_bi), yerr =np.std(mean_activity_uni), fmt = 'x', c = 'r')

axes.legend(['CellActivity','AllFrameActivity'])
axes.set_xlabel('Bidirectional Mean Cell Activity', fontsize = 15)
axes.set_ylabel('Unidirectional Mean Cell Activity', fontsize = 15)
axes.set_ylim([0,1])
axes.set_xlim([0,1])
figure.savefig(figure_path + 'source_extraction_mean_activity_gSig_' + str(5) + '_corr_' + str(min_corr) + '_pnr_' + str(min_pnr) + '_' + str(mouse) + '_uni_vs_bi2.png')

figure, axes = plt.subplots()
axes.hist(temporal_mean_bi, bins = 50, alpha = 0.5)
axes.hist(temporal_mean_uni, bins = 50, alpha = 0.5)
axes.legend(['Bidirectional','Unidirectional'],fontsize = 20)
axes.set_xlabel('Pixel Value', fontsize = 20)
axes.set_ylabel('Count', fontsize = 20)
figure.savefig(figure_path + 'source_extraction_mean_activity_histogram_gSig_' + str(5) + '_corr_' + str(min_corr) + '_pnr_' + str(min_pnr) + '_' + str(mouse) + '_uni_vs_bi2.png')


###################SNR 1###########################################################################

mean_activity_uni_ = np.mean(new_traces_uni[:,0:4500], axis = 1)
std_activity_uni_ = np.std(new_traces_uni[:,0:4500] , axis = 1)#/np.sqrt(new_traces_uni.shape[1])

mean_activity_uni2_ = np.mean(new_traces_uni[:,4500:9000], axis = 1)
std_activity_uni2_ = np.std(new_traces_uni[:,4500:9000] , axis = 1)#/np.sqrt(new_traces_uni.shape[1])

mean_activity_bi_ = np.mean(new_traces_bi , axis = 1)
std_activity_bi_ = np.std(new_traces_bi , axis = 1)#/np.sqrt(new_traces_bi.shape[1])


figure, axes = plt.subplots()
axes.scatter(mean_activity_bi/std_activity_bi, mean_activity_uni/std_activity_uni, facecolors='none', edgecolors='b')
axes.scatter(mean_activity_bi_/std_activity_bi_, mean_activity_uni_/std_activity_uni_,marker = 'x', c = 'r')
#axes.errorbar(mean_activity_bi, mean_activity_uni2, xerr = std_activity_bi, yerr =std_activity_uni2, fmt = 'o', c = 'b')
axes.scatter(np.mean(mean_activity_bi)/np.std(mean_activity_bi),np.mean(mean_activity_uni)/np.std(mean_activity_uni), marker = 'o', c = 'k')
axes.legend(['Templete BIDIRECTIONAL','Templete UNIDIRECTIONAL','AllFrameActivity'])
axes.set_xlabel('Bidirectional CELL SNR', fontsize = 15)
axes.set_ylabel('Unidirectional CELL SNR', fontsize = 15)
# axes.set_ylim([1.5,6])
# axes.set_xlim([1.5,6])
figure.savefig(figure_path + 'source_extraction_cell_SNR_gSig_' + str(5) + '_corr_' + str(min_corr) + '_pnr_' + str(min_pnr) + '_' + str(mouse) + '_uni_vs_bi_all.png')

####################### plot SNR######################################################################
def scatter_hist(x1, y1, x2,y2, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.scatter(x1, y1, facecolors='none', edgecolors='b')
    ax.scatter(x2, y2, marker = 'x', c = 'r')

    # now determine nice limits by hand:
    ax_histx.hist(x1, bins=20, alpha = 0.3, color = 'b', density = True)
    ax_histy.hist(y1, bins=20, orientation='horizontal',color = 'b', alpha = 0.3, density = True)
    ax_histx.hist(x2, bins=20, alpha = 0.3,color = 'r', density = True)
    ax_histy.hist(y2, bins=20, orientation='horizontal',color = 'r', alpha = 0.3, density = True)
    ax_histx.legend(['Bidirectional template', 'Unidirectional Templete'], fontsize=8)
    #ax_histy.legend(['Bidirectional template', 'Unidirectional Templete'], fontsize=5)
    ax.set_xlim([1.5,6])
    ax.set_ylim([1.5,6])

    # ax_histx.set_xlim([1.5,6])
    # ax_histy.set_xlim([1.5,6])

x1 = mean_activity_bi/std_activity_bi
y1 =  mean_activity_uni/std_activity_uni
x2 = mean_activity_bi_/std_activity_bi_
y2 =  mean_activity_uni_/std_activity_uni_
fig = plt.figure(figsize=(6, 6))
gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)
# Create the Axes.
ax = fig.add_subplot(gs[1, 0])
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
# Draw the scatter plot and marginals.
ax.set_ylabel('Unidirectional SNR', fontsize = 15)
ax.set_xlabel('Bidirectional SNR', fontsize = 15)
scatter_hist(x1, y1,x2,y2, ax, ax_histx, ax_histy)
fig.savefig(figure_path + 'source_extraction_cell_SNR_gSig_' + str(5) + '_corr_' + str(min_corr) + '_pnr_' + str(min_pnr) + '_' + str(mouse) + '_uni_vs_bi_all2.png')


