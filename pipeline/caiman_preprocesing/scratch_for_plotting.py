

import caiman as cm
import numpy as np
import os
import psutil
import logging
import data_base as db
import matplotlib.pylab as plt
from preprocessing import run_motion_correction, run_alignment, run_source_extraction
from caiman.source_extraction.cnmf.cnmf import load_CNMF

figure_path = '/scratch/melisa/photon2_test/figures/'

mouse = 3
year = 2022
month = 3
date = 23
example = 0


#########################################################################################################
### plot traces
import scipy.io as sio
file_path  = '/ceph/imaging1/melisa/photon2_test/data_processing/source_extraction/main/mouse_1_year_2022_month_1_day_20_example_1__v.1.0.1.hdf5'
cnm = load_CNMF(file_path)

test = sio.loadmat('/scratch/melisa/photon2_test/forMelisa.mat')
video = test['forMelisa']

onsets_load= sio.loadmat('/scratch/melisa/photon2_test/forMelisa2.mat')
onset_vide1 = onsets_load['vid1OnsetFrame'][0]
onset_vide2 = onsets_load['vid2OnsetFrame'][0]

C_0 = cnm.estimates.C.copy()
C_0[0] += C_0[0].min()
time = np.arange(0,C_0[0].shape[0]/30,1/30)

pos_aux1 = np.where(video[:,1]==0)[0] + 8609
pos_aux2 = np.where(video[:,2]==0)[0] + 8609
pos_aux3 = np.where(video[:,3]==1)[0] + 8609

color1 = [int(pos_aux1[i]*30/1000) for i in range(len(pos_aux1)) if int(pos_aux1[i]*30/1000) < C_0.shape[1]]
color2 = [int(pos_aux2[i]*30/1000) for i in range(len(pos_aux2)) if int(pos_aux2[i]*30/1000) < C_0.shape[1]]
color3 = [int(pos_aux3[i]*30/1000) for i in range(len(pos_aux3)) if int(pos_aux3[i]*30/1000) < C_0.shape[1]]

start_vid1_aux = np.diff(video[:,1])
start_vid2_aux = np.diff(video[:,2])
start_vid3_aux = np.diff(video[:,3])

start_vid1 = np.where(start_vid1_aux == 255)[0]+ 8609
start_vid2 = np.where(start_vid2_aux == 255)[0]+ 8609
start_vid3 = np.where(start_vid3_aux == 0)[0]+ 8609


figure, axes = plt.subplots(1)
for i in range(1, len(C_0)):
    C_0[i] += C_0[i].min() + C_0[:i].max()
    for j in range(0,len(start_vid1)-1):
        time1 = int(start_vid1[j]*30/1000)
        print(time1)
        time1 = onset_vide1[j]
        print(time1)
        time2 = int(time1  + 300)
        axes.plot(np.arange(time1,time2),C_0[i][time1:time2], c = 'r')
        time3 = int(time2 + 30*5)
        axes.plot(np.arange(time2,time3),C_0[i][time2:time3], c = 'k')

    for j in range(0, len(start_vid2)-1):
        time1 = int(start_vid2[j] * 30 / 1000)
        time1 = onset_vide2[j]
        time2 = int(time1 + 300)
        axes.plot(np.arange(time1, time2), C_0[i][time1:time2], c='b')
        time3 = int(time2 + 30*5)
        axes.plot(np.arange(time2,time3),C_0[i][time2:time3], c = 'k')

    #axes.plot(time[color2],C_0[i][color2], c = 'b')
    #axes.plot(time[color3],C_0[i][color3], c = 'k')

#axes.vlines(start_vid1/1000, 0, 1000000, color = 'k' , linewidth = 3)
axes.set_xlabel('t [s]', fontsize = 50)
axes.set_yticks([])
axes.set_ylabel('Actvivity', fontsize = 50)

# axes.vlines(timeline,0, 150000, color = 'k')
axes.set_ylabel('activity')
figure.set_size_inches([50., .5 * len(C_0)])
figure.savefig(figure_path + 'source_extraction_ma_2022_traces_vid1_vid2_iti_new_times.png')

#################################################################################
neurons = cnm.estimates.A.shape[1]

video1_matrix = np.zeros((neurons,300))
video1_iti_matrix = np.zeros((neurons,150))
video2_matrix = np.zeros((neurons,300))
video2_iti_matrix = np.zeros((neurons,150))

video1_evolution = np.zeros((neurons,len(start_vid1)))
video1_iti_evolution = np.zeros((neurons,len(start_vid1)))
video2_evolution = np.zeros((neurons,len(start_vid2)))
video2_iti_evolution = np.zeros((neurons,len(start_vid2)))



for n in range(neurons):
    #figure, axes = plt.subplots(1,4)
    figure = plt.figure()
    gs = plt.GridSpec(10,16)
    ### create matrix with new data
    matrix1 = np.zeros((len(start_vid1), 300))
    temporal_var = np.arange(0,10,1/30)
    for j in range(0, len(start_vid1) - 1):
        time1 = int(start_vid1[j] * 30 / 1000)
        time1 = onset_vide1[j]
        matrix1[j,:] = cnm.estimates.C[n][time1:time1+300]

    mean_activity = np.mean(matrix1,axis = 0)
    matrix1_zs = (matrix1 - np.mean(mean_activity) )/ np.std(mean_activity)
    video1_evolution[n,:] = np.mean(matrix1_zs,axis=1)
    matrix1_norm = (matrix1_zs - np.min(matrix1_zs)) / (np.max(matrix1_zs,axis=0) - np.min(matrix1_zs))
    matrix1 = matrix1_norm


    axes = figure.add_subplot(gs[8:10,0:5])
    mean_activity = np.mean(matrix1,axis = 0)
    std_activity = np.std(matrix1,axis = 0)#/np.sqrt(matrix1.shape[0])
    video1_matrix[n,:] = mean_activity
    axes.fill_between(temporal_var,mean_activity - std_activity,mean_activity + std_activity, alpha = 0.1, color = 'r')
    axes.plot(temporal_var,mean_activity, color = 'r', linewidth = 2)
    axes.set_yticks([])
    axes.set_ylabel('Mean Activity', fontsize = 10)
    axes.set_xlabel('t [s]', fontsize=10)

    axes = figure.add_subplot(gs[0:7,0:5])
    matrix1[0,:] += matrix1[0].min()
    for j in range(1,len(start_vid1)-1):
        matrix1[j] += matrix1[j].min() + matrix1[:j].max()
        axes.plot(temporal_var ,matrix1[j], c = 'r')
    axes.set_yticks([])
    axes.set_title('VIDE0 1', fontsize=12)
#    axes.set_xlabel('t [s]', fontsize=10)
    axes.set_ylabel('Trials', fontsize = 10)


    matrix = np.zeros((len(start_vid1), 150))
    temporal_var2 = np.arange(0,5,1/30)
    for j in range(0, len(start_vid1) - 1):
        time1 = int(start_vid1[j] * 30 / 1000) + 300
        time1 = onset_vide1[j] + 300
        matrix[j,:] = cnm.estimates.C[n][time1:time1+150]
    matrix[0,:] += matrix[0].min()
    mean_activity = np.mean(matrix,axis = 0)
    matrix_zs = (matrix - np.mean(mean_activity) )/ np.std(mean_activity)
    video1_iti_evolution[n,:] = np.mean(matrix_zs,axis=1)
    matrix_norm = (matrix_zs - np.min(matrix_zs)) / (np.max(matrix_zs,axis=0) - np.min(matrix_zs))
    matrix = matrix_norm

    axes = figure.add_subplot(gs[8:10,5:8])
    mean_activity = np.mean(matrix,axis = 0)
    std_activity = np.std(matrix,axis = 0)#/np.sqrt(matrix.shape[0])
    video1_iti_matrix[n,:] = mean_activity
    axes.fill_between(temporal_var2,mean_activity - std_activity,mean_activity + std_activity, alpha = 0.1, color = 'k')
    axes.plot(temporal_var2,mean_activity, color = 'k', linewidth = 2)
    axes.set_yticks([])
    axes.set_xlabel('t [s]', fontsize=10)

    axes = figure.add_subplot(gs[0:7,5:8])
    for j in range(1,len(start_vid1)-1):
        matrix[j] += matrix[j].min() + matrix[:j].max()
        axes.plot(temporal_var2 ,matrix[j], c = 'k')
    axes.set_yticks([])
    axes.set_title('ITI VIDE0 1', fontsize=12)
    axes.set_yticks([])


    ### create matrix with new data
    matrix2 = np.zeros((len(start_vid2), 300))
    for j in range(0, len(start_vid2) - 1):
        time1 = int(start_vid2[j] * 30 / 1000)
        time1 = onset_vide2[j]
        matrix2[j,:]= cnm.estimates.C[n][time1:time1+300]
    mean_activity = np.mean(matrix2,axis = 0)
    matrix2_zs = (matrix2 - np.mean(mean_activity) )/ np.std(mean_activity)
    video2_evolution[n,:] = np.mean(matrix2_zs,axis=1)

    matrix2_norm = (matrix2_zs - np.min(matrix2_zs)) / (np.max(matrix2_zs,axis=0) - np.min(matrix2_zs))
    matrix2 = matrix2_norm

    axes = figure.add_subplot(gs[8:10,9:13])
    mean_activity = np.mean(matrix2,axis = 0)
    std_activity = np.std(matrix2,axis = 0)#/np.sqrt(matrix1.shape[0])
    video2_matrix[n,:] = mean_activity
    axes.fill_between(temporal_var,mean_activity - std_activity,mean_activity + std_activity, alpha = 0.1, color = 'b')
    axes.plot(temporal_var,mean_activity, color = 'b', linewidth = 2)
    axes.set_yticks([])
    axes.set_xlabel('t [s]', fontsize=10)

    axes = figure.add_subplot(gs[0:7,9:13])
    matrix2[0] += matrix2[0].min()
    for j in range(1,len(start_vid1)-1):
        matrix2[j] += matrix2[j].min() + matrix2[:j].max()
        axes.plot(temporal_var,matrix2[j], c = 'b')
    axes.set_yticks([])
    axes.set_title('VIDE0 2', fontsize=12)

    matrix = np.zeros((len(start_vid2), 150))
    for j in range(0, len(start_vid2) - 1):
        time1 = int(start_vid2[j] * 30 / 1000) + 300
        time1 = onset_vide2[j] + 300
        matrix[j,:] = cnm.estimates.C[n][time1:time1+150]

    mean_activity = np.mean(matrix,axis = 0)
    matrix_zs = (matrix - np.mean(mean_activity) )/ np.std(mean_activity)
    video2_iti_evolution[n,:] = np.mean(matrix_zs,axis=1)
    matrix_norm = (matrix_zs - np.min(matrix_zs)) / (np.max(matrix_zs,axis=0) - np.min(matrix_zs))
    matrix = matrix_norm

    mean_activity = np.mean(matrix,axis = 0)
    std_activity = np.std(matrix,axis = 0)#/np.sqrt(matrix.shape[0])
    video2_iti_matrix[n,:] = mean_activity
    axes = figure.add_subplot(gs[8:10,13:16])
    axes.fill_between(temporal_var2,mean_activity - std_activity,mean_activity + std_activity, alpha = 0.1, color = 'k')
    axes.plot(temporal_var2,mean_activity, color = 'k', linewidth = 3)
    axes.set_yticks([])
    axes.set_xlabel('t [s]', fontsize=10)

    axes = figure.add_subplot(gs[0:7,13:16])
    for j in range(1,len(start_vid1)-1):
        matrix[j] += matrix[j].min() + matrix[:j].max()
        axes.plot(temporal_var2,matrix[j], c = 'k')
    axes.set_yticks([])
    axes.set_title('ITI VIDE0 2', fontsize=12)

    figure.savefig(figure_path + 'source_extraction_ma_2022_traces_normed_neuron_' + f' {n}' + '_new_times_.png')

plt.close('all')
#################################################################################

figure = plt.figure()
gs = plt.GridSpec(10,16)
axes = figure.add_subplot(gs[0:7, 0:5])
neuron_axes = np.arange(0,neurons)
mesh = axes.pcolormesh(temporal_var,neuron_axes,video1_matrix, cmap = 'Reds')
axes.set_title('VIDE0 1', fontsize=12)
axes.set_ylabel('Neurons', fontsize = 12)
axes.set_xticks([])
axes = figure.add_subplot(gs[7:10, 0:5])
mean_activity = np.mean(video1_matrix, axis = 0)
std_activity = np.std(video1_matrix,axis = 0)
axes.fill_between(temporal_var, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color='r')
axes.plot(temporal_var, mean_activity, c='r', linewidth=1)
axes.set_ylim([0.2,0.55])
axes.set_xlabel('Time [s]')
axes.set_ylabel('Mean Activity')

axes = figure.add_subplot(gs[0:7, 5:8])
neuron_axes = np.arange(0,neurons)
mesh = axes.pcolormesh(temporal_var2,neuron_axes,video1_iti_matrix, cmap = 'Greys')
axes.set_yticks([])
axes.set_title('ITI VIDE0 1', fontsize=12)
axes.set_xticks([])
axes = figure.add_subplot(gs[7:10, 5:8])
mean_activity = np.mean(video1_iti_matrix, axis = 0)
std_activity = np.std(video1_iti_matrix,axis = 0)
axes.fill_between(temporal_var2, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color='k')
axes.plot(temporal_var2, mean_activity, c='k', linewidth=1)
axes.set_ylim([0.2,0.55])
axes.set_yticks([])
axes.set_xlabel('Time [s]')


axes = figure.add_subplot(gs[0:7, 9:13])
neuron_axes = np.arange(0,neurons)
mesh = axes.pcolormesh(temporal_var,neuron_axes,video2_matrix, cmap = 'Blues')
axes.set_yticks([])
axes.set_title('VIDE0 2', fontsize=12)
axes.set_xticks([])
axes = figure.add_subplot(gs[7:10, 9:13])
mean_activity = np.mean(video2_matrix, axis = 0)
std_activity = np.std(video2_matrix,axis = 0)
axes.fill_between(temporal_var, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color='b')
axes.plot(temporal_var, mean_activity, c='b', linewidth=1)
axes.set_ylim([0.2,0.55])
axes.set_yticks([])
axes.set_xlabel('Time [s]')


axes = figure.add_subplot(gs[0:7, 13:16])
neuron_axes = np.arange(0,neurons)
mesh = axes.pcolormesh(temporal_var2,neuron_axes,video2_iti_matrix, cmap = 'Greys')
axes.set_yticks([])
axes.set_title('ITI VIDE0 2', fontsize=12)
axes.set_xticks([])
axes = figure.add_subplot(gs[7:10, 13:16])
mean_activity = np.mean(video2_iti_matrix, axis = 0)
std_activity = np.std(video2_iti_matrix,axis = 0)
axes.fill_between(temporal_var2, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color='k')
axes.plot(temporal_var2, mean_activity, c='k', linewidth=1)
axes.set_ylim([0.2,0.55])
axes.set_yticks([])
axes.set_xlabel('Time [s]')

figure.savefig(figure_path + 'source_extraction_ma_2022_traces_all_neuron_new_times_normed.png')
##########################################################################################3

figure, axes = plt.subplots()

mean_activity = np.mean(video1_matrix, axis = 0)
std_activity = np.std(video1_matrix,axis = 0)/np.sqrt(neurons)
axes.fill_between(temporal_var, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color='r')
axes.plot(temporal_var, mean_activity, c='r', linewidth=1)
std_activity = np.std(video2_matrix,axis = 0)/np.sqrt(neurons)
mean_activity = np.mean(video2_matrix, axis = 0)
axes.fill_between(temporal_var, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color='b')
axes.plot(temporal_var, mean_activity, c='b', linewidth=1)

axes.set_xlabel('Time [s]')
axes.set_ylabel('Mean Activity')
axes.set_ylim([0.3,0.4])
axes.legend(['VIDEO1','VIDEO2'])

figure.savefig(figure_path + 'source_extraction_ma_2022_traces_all_neuron_mean_new_times.png')

##############################################################################################

figure, axes = plt.subplots()

mean_activity = np.mean(video1_iti_matrix, axis = 0)
std_activity = np.std(video1_iti_matrix,axis = 0)/np.sqrt(neurons)
axes.fill_between(temporal_var2, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color='r')
axes.plot(temporal_var2, mean_activity, c='r', linewidth=1)
std_activity = np.std(video2_iti_matrix,axis = 0)/np.sqrt(neurons)
mean_activity = np.mean(video2_iti_matrix, axis = 0)
axes.fill_between(temporal_var2, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color='b')
axes.plot(temporal_var2, mean_activity, c='b', linewidth=1)

axes.set_xlabel('Time [s]')
axes.set_ylabel('Mean Activity')
#axes.set_ylim([0.3,0.4])
axes.legend(['VIDEO1','VIDEO2'])

figure.savefig(figure_path + 'source_extraction_ma_2022_traces_all_neuron_mean_iti_new_times.png')


############################################################################################3
trial = np.arange(0,video1_evolution.shape[1])

figure, axes = plt.subplots()
#axes = figure.add_subplot(gs[0:7, 0:5])
mesh = axes.pcolormesh(trial,neuron_axes,video1_evolution, cmap = 'Reds')


figure, axes = plt.subplots()
trial_id = np.arange(1,31)
axes.scatter(trial_id,np.mean(video1_iti_evolution,axis=0), c = 'r')
axes.plot(trial_id,np.mean(video1_iti_evolution,axis=0), c = 'r')
axes.scatter(trial_id,np.mean(video2_iti_evolution,axis=0), c = 'b')
axes.plot(trial_id,np.mean(video2_iti_evolution,axis=0), c = 'b')

axes.set_xlim([0,29])
axes.set_ylim([-2,2])

axes.set_xlabel('Trial', fontsize = 15)
axes.set_ylabel('Total Mean Activity', fontsize = 15)
axes.legend(['VIDEO1 ITI', 'VIDEO2 ITI'])

figure.savefig(figure_path + 'source_extraction_ma_2022_traces_all_neuron_mean_trial_iti_new_times.png')
