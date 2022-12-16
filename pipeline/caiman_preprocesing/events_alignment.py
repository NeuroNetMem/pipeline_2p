'''

author: Melisa
Data : March 24th 2022

This scripts requires three components: - cnm files of extracted cells
                                         - temporal alignmnet time stamps
                                         - log file information with offset and events time stamps


'''

import caiman as cm
import numpy as np
import os
import psutil
import logging
import data_base as db
import matplotlib.pylab as plt
from preprocessing import run_motion_correction, run_alignment, run_source_extraction
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import scipy.io as sio
import matplotlib.pylab as plt
import pickle


figure_path = '/scratch/melisa/photon2_test/figures/'
states = db.open_data_base()
mouse = 3
year = 2022
month = 3
date = 23
example = 0

### load cnm file
selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,1,1,1])
cnm_file_path  = eval(selected_rows.iloc[0]['source_extraction_output'])['main']
cnm = load_CNMF(cnm_file_path)
### load timline
timeline_file_path = eval(selected_rows.iloc[0]['alignment_output'])['meta']['timeline']
timeline = np.load(timeline_file_path)
### load lof file information
### (later this will be multiple log files)

loginfo_file_path = '/ceph/imaging1/melisa/photon2_test/data/log_information/20220323-143325_213_output.pickle'
log_pickle = open(loginfo_file_path, 'rb')
log_information = pickle.load(log_pickle)

srate = log_information['sampling_rate'] ## sampling rate in Hz
offset = log_information['offset'] ## offset in frames
stim_lenght = 2 ### in seconds
iti_lenght = 10

### create a list with the time stamps of all sounds
### remember that timestamps are already in frames
### now the thing to add is the offset related to multiple files aligments
sounds_list = []
sounds_list.append(log_information['sound1'])
sounds_list.append(log_information['sound2'])
sounds_list.append(log_information['sound3'])
sounds_list.append(log_information['sound4'])
sounds_list.append(log_information['sound5'])
sounds_list.append(log_information['sound6'])

#sounds_list = [sounds_list.append(log_information['sound' + f'{i+1}']) for i in range(6)]
iti = list(log_information['iti'])

C_0 = new_traces#cnm.estimates.C.copy()
time = np.arange(0,C_0[0].shape[0]/srate,1/srate)

#############################################################################################################3
##### plot traces coloring according to sound stimuli

colors = ['r','b','g','magenta','orange','cyan']
#### plot temporal traces with multiple colors depending on the sound
figure, axes = plt.subplots(1)
for cell_index in range(50, 51):
    C_0[cell_index] += C_0[cell_index].min() + C_0[:cell_index].max()
    axes.plot(np.arange(0, len(C_0[cell_index])) / srate, C_0[cell_index], c='k')
    for sound_index in range(len(sounds_list)):
        #for j in range(0,len(sounds_list[sound_index])):
        for j in range(0,sounds_list[sound_index].shape[0]):
            sound_onset = int(sounds_list[sound_index][j] + timeline[1])
            sound_end = int(sound_onset  + stim_lenght  * srate)
            axes.plot(np.arange(sound_onset,sound_end)/srate,C_0[cell_index][sound_onset:sound_end], c = 'k')
            iti_onset = sound_end
            iti_end = int(sound_end + iti_lenght*srate)
            axes.plot(np.arange(iti_onset,iti_end)/srate,C_0[cell_index][iti_onset:iti_end], c = colors[sound_index])

axes.set_xlabel('t [s]', fontsize = 25)
axes.set_yticks([])
axes.set_ylabel('Actvivity', fontsize = 25)

axes.set_ylabel('activity')
#figure.set_size_inches([50., .5 * len(C_0)])
figure.savefig(figure_path + 'source_extraction_2022_03_23_traces_audioseq_opposite_1_cell.png')

########################################################################################
#plot contours
output_source_extraction = eval(selected_rows.iloc[0]['source_extraction_output'])
corr_path = output_source_extraction['meta']['corr']['main']
cn_filter = np.load(corr_path)

figure, axes = plt.subplots()
axes.imshow(cn_filter,cmap = 'gray')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape, 0.2, 'max')
counter = 0
for c in coordinates:
    if counter > 0:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes.plot(*v.T, c='r')
    counter = counter + 1


#################################################################################################################
### plot individual neuron activity as a raster plot for different audio stimuli

n_neurons = cnm.estimates.A.shape[1]

audio_matrix_list = []
iti_matrix_list = []
mean_sound_activity_evolution_list = []
mean_iti_activity_evolution_list = []
C_0 = cnm.estimates.C.copy()

for sound_index in range(len(sounds_list)):
    #create matrix to contain mean activity of all neurons over trials
    sound_matrix = np.zeros((n_neurons,int(stim_lenght*srate)))
    iti_matrix = np.zeros((n_neurons,int(iti_lenght*srate)))
    audio_matrix_list.append(sound_matrix)
    iti_matrix_list.append(iti_matrix)

    #create matrix to contrain mean activity evolution over trials
    sound_mean = np.zeros((n_neurons,len(sounds_list[sound_index])))
    iti_mean = np.zeros((n_neurons,len(sounds_list[sound_index])))
    mean_sound_activity_evolution_list.append(sound_mean)
    mean_iti_activity_evolution_list.append(iti_mean)


traces = cnm.estimates.C.copy()
### zscored traces
traces_zscored = traces - traces.mean(axis = 1, keepdims = True) / traces.std(axis = 1, keepdims = True)
### normed traces
traces_normed = (traces_zscored - traces_zscored.min(axis=1,keepdims = True))/(traces_zscored.max(axis=1, keepdims = True) - traces_zscored.min(axis = 1, keepdims = True))
C_final = traces_normed

for n in range(n_neurons):
    print('cell number = ' , n)
    #figure, axes = plt.subplots(1,4)
    # create a figure for every neuron
    figure = plt.figure()
    gs = plt.GridSpec(15,48)
    axes = figure.add_subplot(gs[0:3, 0:12])
    axes.imshow(cn_filter, cmap='gray')
    counter = 0
    for c in coordinates:
        if counter == n:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                         np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes.plot(*v.T, c='r')
        counter = counter + 1
    axes.set_xlabel('Pixel',fontsize = 20)
    axes.set_ylabel('Pixel',fontsize = 20)


    axes = figure.add_subplot(gs[0:3, 10:32])
    axes.plot(np.arange(0, len(C_0[n])) / srate, C_final[n], c='k')
    axes.set_xlabel('time [s]', fontsize=25)
    axes.set_yticks([])
    axes.set_ylabel('Actvivity', fontsize=25)

    for sound_index in range(len(sounds_list)):
        for j in range(0,sounds_list[sound_index].shape[0]):
            sound_onset = int(sounds_list[sound_index][j] + timeline[1])
            sound_end = int(sound_onset  + stim_lenght  * srate)
            axes.plot(np.arange(sound_onset,sound_end)/srate,C_final[n][sound_onset:sound_end], c = 'k')
            iti_onset = sound_end
            iti_end = int(sound_end + iti_lenght*srate)
            axes.plot(np.arange(iti_onset,iti_end)/srate,C_final[n][iti_onset:iti_end], c = colors[sound_index])

    axes_0 = figure.add_subplot(gs[0:3, 37:39])
    axes_1 = figure.add_subplot(gs[0:3, 40:48])

    for sound_index in range(len(sounds_list)): # for everysound
        ### create matrix con trials
        aux_matrix_sound = np.zeros((len(sounds_list[sound_index]), int(stim_lenght*srate)))
        aux_time = np.arange(0,stim_lenght,1/srate)
        for trial in range(0, len(sounds_list[sound_index])):
            sound_onset = int(sounds_list[sound_index][trial] + timeline[1])
            aux_matrix_sound[trial,:] = C_final[n][sound_onset:sound_onset+int(stim_lenght*srate)]
        #mean_activity = np.mean(aux_matrix_sound,axis = 0)

        ### z-score and normalize traces
        # aux_matrix_zs = (aux_matrix_sound - aux_matrix_sound.mean(axis = 1, keepdims = True))/ aux_matrix_sound.std(axis = 1,keepdims=True)
        mean_sound_activity_evolution_list[sound_index][n,:] = np.mean(aux_matrix_sound,axis=1)
        # matrix1_norm = (aux_matrix_zs - aux_matrix_zs.min(axis = 1,keepdims = True)) / (aux_matrix_zs.max(axis=1, keepdims = True) - aux_matrix_zs.min(axis = 1,keepdims= True))
        final_matrix = aux_matrix_sound

        axes = figure.add_subplot(gs[13:15, sound_index * 8 : sound_index * 8 +1])
        mean_activity = np.mean(final_matrix,axis = 0)
        std_activity = np.std(final_matrix,axis = 0)#/np.sqrt(matrix1.shape[0])
        ###save the mean for population analysis
        audio_matrix_list[sound_index][n,:] = mean_activity
        temporal_var = np.arange(0,len(mean_activity))/srate
        axes.fill_between(temporal_var,mean_activity - std_activity,mean_activity + std_activity, alpha = 0.1, color = 'k')
        axes.plot(temporal_var,mean_activity, color = 'k', linewidth = 2)
        #axes.set_yticks([])
        axes.set_ylim([-0.1,0.1])
        if sound_index == 0:
            axes.set_ylabel('Mean Activity', fontsize = 15)
        axes.set_xlabel('time [s]', fontsize=15)

        axes_0.fill_between(temporal_var, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1,
                          color = colors[sound_index])
        axes_0.plot(temporal_var, mean_activity, color = colors[sound_index], linewidth=2)
        # axes.set_yticks([])
        axes_0.set_ylim([-0.15, 0.15])
        axes_0.set_ylabel('Mean Activity', fontsize=15)
        axes_0.set_xlabel('time [s]', fontsize=15)


        axes = figure.add_subplot(gs[5:12,sound_index * 8  : sound_index * 8 +1])
        final_matrix[0,:] += final_matrix[0].min()
        for j in range(1,len(sounds_list[sound_index])):
            final_matrix[j] += final_matrix[j].min() + 0.1 * j #+ final_matrix[:j].max()
            axes.plot(temporal_var,final_matrix[j], c = 'k')
        axes.set_yticks([])
        axes.set_title('SOUND = ' + f'{sound_index+1}' , fontsize=12)
    #    axes.set_xlabel('t [s]', fontsize=10)
        if sound_index == 0:
            axes.set_ylabel('Trials', fontsize = 20)

        ###same for iti
        aux_matrix_iti = np.zeros((len(sounds_list[sound_index]), int(iti_lenght*srate)))
        aux_time = np.arange(0,stim_lenght,1/srate)
        for trial in range(0, len(sounds_list[sound_index])):
            iti_onset = int(sounds_list[sound_index][trial] + timeline[1]) + int(stim_lenght*srate)
            aux_matrix_iti[trial,:] = C_final[n][iti_onset:iti_onset+int(iti_lenght*srate)]
        #mean_activity = np.mean(aux_matrix_iti,axis = 0)

        ### z-score and normalize traces
        #aux_matrix_zs = (aux_matrix_iti - np.mean(aux_matrix_iti) )/ np.std(aux_matrix_iti)
        # aux_matrix_zs = (aux_matrix_iti- aux_matrix_iti.mean(axis = 1, keepdims = True))/ aux_matrix_iti.std(axis = 1,keepdims=True)
        # matrix1_norm = (aux_matrix_zs - aux_matrix_zs.min(axis = 1,keepdims = True)) / (aux_matrix_zs.max(axis=1, keepdims = True) - aux_matrix_zs.min(axis = 1,keepdims= True))
        final_matrix_iti = aux_matrix_iti

        mean_iti_activity_evolution_list[sound_index][n,:] = np.mean(aux_matrix_iti,axis=1)
        # matrix1_norm = (aux_matrix_zs - np.min(aux_matrix_zs,axis = 1)) / (np.max( aux_matrix_zs,axis=1) - np.min(aux_matrix_zs,axis = 1 ))
        # final_matrix_iti =  matrix1_norm

        axes = figure.add_subplot(gs[13:15, sound_index * 8 + 1: (sound_index +1)* 8 -1])
        mean_activity = np.mean(final_matrix_iti,axis = 0)
        std_activity = np.std(final_matrix_iti,axis = 0)#/np.sqrt(matrix1.shape[0])
        ###save the mean for population analysis
        iti_matrix_list[sound_index][n,:] = mean_activity
        temporal_var_iti = np.arange(0,len(mean_activity))/srate
        axes.fill_between(temporal_var_iti,mean_activity - std_activity,mean_activity + std_activity, alpha = 0.1,
                          color = colors[sound_index])
        axes.plot(temporal_var_iti,mean_activity, color = colors[sound_index], linewidth = 2)
        axes.set_yticks([])
        axes.set_ylim([-0.1,0.1])
        #axes.set_ylabel('Mean Activity', fontsize = 10)
        axes.set_xlabel('time [s]', fontsize=20)


        axes_1.fill_between(temporal_var_iti, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1,
                          color = colors[sound_index])
        axes_1.plot(temporal_var_iti, mean_activity, color = colors[sound_index], linewidth = 2)
        axes_1.set_yticks([])
        axes_1.set_ylim([-0.15, 0.15])
        #axes_1.set_ylabel('Mean Activity', fontsize=10)
        axes_1.set_xlabel('t [s]', fontsize=10)

        axes = figure.add_subplot(gs[5:12,  sound_index * 8 + 1: (sound_index +1)* 8 -1])
        final_matrix_iti[0,:] += final_matrix_iti[0].min()
        for j in range(1,len(sounds_list[sound_index])):
            final_matrix_iti[j] += final_matrix_iti[j].min() + 0.1 * j #+ final_matrix_iti[:j].max()
            axes.plot(temporal_var_iti,final_matrix_iti[j], c = colors[sound_index])
        axes.set_yticks([])
        axes.set_title('ITI = ' + f'{sound_index+1}' , fontsize=12)
    #    axes.set_xlabel('t [s]', fontsize=10)
        #axes.set_ylabel('Trials', fontsize = 10)

    figure.set_size_inches([25,10])
    figure.savefig(figure_path + 'source_extraction_audiqseg_23_03_2022_traces_normed_neuron_' + f' {n}' + '.png')

    plt.close()


####################################################################


figure = plt.figure()
gs = plt.GridSpec(1, 8)

for sound_index in range(len(sounds_list)):

    axes = figure.add_subplot(gs[0, 0: 2])

    mean_activity = np.nanmean(audio_matrix_list[sound_index], axis = 0)
    std_activity = np.nanstd(audio_matrix_list[sound_index],axis = 0)/np.sqrt(n_neurons)
    temporal_var = np.arange(0, len(mean_activity)) / srate

    axes.fill_between(temporal_var, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color=colors[sound_index])
    axes.plot(temporal_var, mean_activity, c=colors[sound_index], linewidth=1)

    axes.set_xlabel('Time [s]',fontsize = 15)
    axes.set_ylabel('Mean Activity',fontsize = 15)
    axes.set_title('AUDIO',fontsize = 20)
    axes.set_ylim([0.012,0.03])

    axes = figure.add_subplot(gs[0, 3: 8])
    mean_activity = np.nanmean(iti_matrix_list[sound_index], axis = 0)
    std_activity = np.nanstd(iti_matrix_list[sound_index],axis = 0)/np.sqrt(n_neurons)

    temporal_var_iti = np.arange(0, len(mean_activity)) / srate
    axes.fill_between(temporal_var_iti, mean_activity - std_activity, mean_activity + std_activity, alpha=0.1, color=colors[sound_index])
    axes.plot(temporal_var_iti, mean_activity, c=colors[sound_index], linewidth=1)

    axes.set_xlabel('Time [s]',fontsize = 15)
    #axes.set_ylabel('Mean Activity',fontsize = 15)
    axes.set_title('ITI',fontsize = 20)
    axes.set_ylim([0.012,0.03])


#axes.set_ylim([0.3,0.4])
axes.legend(['audio1','audio2','audio3','audio4','audio5','audio6'], fontsize = 15)

figure.set_size_inches([15,7])
figure.savefig(figure_path + 'source_extraction_2022_23_03_traces_all_neuron_mean_normed.png')


################################



#axes = figure.add_subplot(gs[0:7, 0:5])
#mesh = axes.pcolormesh(trial,neuron_axes,video1_evolution, cmap = 'Reds')


figure, axes = plt.subplots(1,2)

for sound_index in range(len(mean_sound_activity_evolution_list)):
    trial_id = np.arange(0,mean_sound_activity_evolution_list[sound_index].shape[1])
    axes[0].scatter(trial_id,np.nanmean(mean_sound_activity_evolution_list[sound_index],axis=0), c = colors[sound_index])
    axes[0].plot(trial_id,np.nanmean(mean_sound_activity_evolution_list[sound_index],axis=0), c = colors[sound_index])

    axes[1].scatter(trial_id,np.nanmean(mean_iti_activity_evolution_list[sound_index],axis=0), c = colors[sound_index])
    axes[1].plot(trial_id,np.nanmean(mean_iti_activity_evolution_list[sound_index],axis=0), c = colors[sound_index])


axes[0].set_xlabel('Trial', fontsize = 15)
axes[0].set_ylabel('Total Mean Activity', fontsize = 15)
axes[0].legend(['audio1','audio2','audio3','audio4','audio5','audio6'], fontsize = 15)
axes[0].set_title('AUDIO')
axes[0].set_ylim([0,0.12])
axes[1].set_xlabel('Trial', fontsize = 15)
axes[1].set_ylabel('Total Mean Activity', fontsize = 15)
axes[1].legend(['audio1','audio2','audio3','audio4','audio5','audio6'], fontsize = 15)
axes[1].set_title('ITI')
axes[1].set_ylim([0,0.12])

figure.set_size_inches([15,7])
figure.savefig(figure_path + 'source_extraction_2022_23_03_traces_all_neuron_mean_trial_normed_evolution.png')
