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
import cv2
from matplotlib.patches import Rectangle
from matplotlib.colors import LogNorm



figure_path = '/ceph/imaging1/melisa/photon2_test/figures/video_plot/'
states = db.open_data_base()
mouse = 3
year = 2022
month = 3
date = 23
example = 0

### load cnm file
selected_rows = db.select(states, mouse = mouse, year = year,month=month,date = date, analysis_version = [1,1,1,1])
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

C_0 = cnm.estimates.C.copy()
time = np.arange(0,C_0[0].shape[0]/srate,1/srate)

#plot contours
input_video = eval(selected_rows.iloc[0]['alignment_output'])['main']
Yr, dims, T = cm.load_memmap(input_video)
#create a caiman movie with the mmap file
video = Yr.T.reshape((T,) + dims, order='F')
video = cm.movie(video)

output_source_extraction =  eval(selected_rows.iloc[0]['source_extraction_output'])
corr_path = output_source_extraction['meta']['corr']['main']
cn_filter = np.load(corr_path)

figure, axes = plt.subplots()
axes.imshow(video[0,:,:],cmap = 'gray')
coordinates = cm.utils.visualization.get_contours(cnm.estimates.A, cn_filter.shape, 0.2, 'max')
counter = 0
for c in coordinates:
    if counter > 0:
        v = c['coordinates']
        c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                     np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
        axes.plot(*v.T, c='r')
    counter = counter + 1

traces = cnm.estimates.C.copy()
### zscored traces
traces_zscored = traces - traces.mean(axis = 1, keepdims = True) / traces.std(axis = 1, keepdims = True)
### normed traces
traces_normed = (traces_zscored - traces_zscored.min(axis=1,keepdims = True))/(traces_zscored.max(axis=1, keepdims = True) - traces_zscored.min(axis = 1, keepdims = True))
C_final = traces_normed

n_neurons = cnm.estimates.A.shape[1]
colors = ['w','r','b','g','magenta','yellow','cyan']
color2 = ['k','r','b','g','magenta','yellow','cyan']


frame_range = slice(None, None, None)
Y_rec = cnm.estimates.A.dot(C_0[:, frame_range])
Y_rec = Y_rec.reshape(dims + (-1,), order='F')
Y_rec = Y_rec.transpose([2, 0, 1])
# convert the variable to a caiman movie type
Y_rec = cm.movie(Y_rec)

stimuli_vector_1 = np.zeros((video.shape[0],))
alpha_vector = np.ones((video.shape[0],))*0.2
for sound_index in range(6):
    for i in range(len(sounds_list[sound_index])):
        stimuli_vector_1[sounds_list[sound_index][i]+int(timeline[1]):sounds_list[sound_index][i]+int(srate * 12)+int(timeline[1])]=sound_index+1
        alpha_vector[sounds_list[sound_index][i]+int(timeline[1]):sounds_list[sound_index][i]+int(srate * 12)+int(timeline[1])]=0.2
        alpha_vector[sounds_list[sound_index][i]+int(timeline[1]):sounds_list[sound_index][i]+int(srate * 2)+int(timeline[1])]=0.5


temporal_variable = np.arange(0, len(C_0[0])) / srate

Y_rec_max =  int(np.nanmax( np.nanmax(Y_rec,axis=(1,2))))
movie_max = int(np.nanmax( np.nanmax(video,axis=(1,2))))
##############################
## complete plot
for n in range(74,75):
    print('cell number = ' , n)
    #figure, axes = plt.subplots(1,4)
    # create a figure for every neuron
    for time_index in range(0,video.shape[0]):
        if time_index > 750*33 and time_index < 850*33:
            figure = plt.figure()
            gs = plt.GridSpec(6,6)
            axes = figure.add_subplot(gs[0:3, 0:2])
            axes.set_title('Raw video', fontsize = 15)
            axes.imshow(video[time_index, :, :], cmap='gray',vmin=0.01, vmax=movie_max/2)
            counter = 0
            for c in coordinates:
                if counter == n:
                    v = c['coordinates']
                    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                    axes.plot(*v.T, c=colors[int(stimuli_vector_1[time_index])])
                counter = counter + 1
            #axes.set_xlabel('Pixel',fontsize = 12)
            axes.set_ylabel('Pixel',fontsize = 12)

            axes = figure.add_subplot(gs[3:6, 0:2])
            axes.set_title('CaImAn model', fontsize = 15)
            axes.imshow(Y_rec[time_index, :, :], cmap='gray',vmin=0.0, vmax=Y_rec_max/10)
            # pos = axes.imshow(Y_rec[time_index, :, :], cmap='gray',vmin=0, vmax=Y_rec_max)
            # figure.colorbar(pos, ax=axes)
            counter = 0
            for c in coordinates:
                if counter == n:
                    v = c['coordinates']
                    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                    axes.plot(*v.T, c=colors[int(stimuli_vector_1[time_index])])
                counter = counter + 1
            axes.set_xlabel('Pixel',fontsize = 12)
            axes.set_ylabel('Pixel',fontsize = 12)

            axes = figure.add_subplot(gs[0:2, 2:6])
            axes.set_title('Calcium Trace', fontsize = 15)
            axes.plot(temporal_variable, C_final[n], c='k')
            axes.set_xlabel('t [s]', fontsize=15)
            axes.set_yticks([])
            axes.set_ylabel('Actvivity', fontsize=15)

            for sound_index in range(len(sounds_list)):
                for j in range(0,sounds_list[sound_index].shape[0]):
                    sound_onset = int(sounds_list[sound_index][j] + timeline[1])
                    sound_end = int(sound_onset  + stim_lenght  * srate)
                    axes.plot(np.arange(sound_onset,sound_end)/srate,C_final[n][sound_onset:sound_end], c = 'k')
                    iti_onset = sound_end
                    iti_end = int(sound_end + iti_lenght*srate)
                    axes.plot(np.arange(iti_onset,iti_end)/srate,C_final[n][iti_onset:iti_end], c = colors[sound_index+1])
            axes.vlines(time_index/srate,'k','.')

            axes = figure.add_subplot(gs[3:6, 3:5])
            axes.plot(temporal_variable[time_index - int(srate * 2):time_index + int(srate * 2)], C_final[n,time_index - int(srate * 2):time_index + int(srate * 2)], c = 'k')
            axes.set_xlabel('t [s]', fontsize=12)
            #axes.set_ylim([0,1])
            axes.set_ylabel('Actvivity', fontsize=12)
            rect = Rectangle((temporal_variable[time_index - int(srate)],0),2,1, fill=True, color=color2[int(stimuli_vector_1[time_index])], linestyle='-', linewidth=2, alpha = alpha_vector[time_index])
            axes.vlines(time_index/srate ,0,1,linestyles ="dotted",color = 'k')
            axes.add_patch(rect)

            figure.set_size_inches([15, 10])

            figure.savefig(figure_path + 'movie_'+ f'{n}' + '_' + f'{100000+time_index}' + '.png')
            plt.close()
            if time_index % 3000 == 0:
                print(time_index)


##################################3

## all cells plots for movie

for time_index in range(0,video.shape[0]):
    if time_index % 30 == 0:
        figure = plt.figure()
        gs = plt.GridSpec(1,1)
        axes = figure.add_subplot(gs[0, 0])
        axes.set_title('Raw data', fontsize = 15)
        axes.imshow(video[time_index, :, :], cmap='gray',vmin=0.01, vmax=movie_max/2)
        for c in coordinates:
            v = c['coordinates']
            c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                        np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
            axes.plot(*v.T, c=colors[int(stimuli_vector_1[time_index])])
            #axes.set_xlabel('Pixel',fontsize = 12)
        axes.set_ylabel('Pixel',fontsize = 12)

        figure.savefig(figure_path + 'movie_raw_'+ '_' + f'{100000+time_index}' + '.png')
        plt.close()
        if time_index % 3000 == 0:
            print(time_index)


########################################################################
### crate the sum over all video
total_activity = np.nanmean(C_final,axis = 0)
total_activity_std = np.nanstd(C_final,axis = 0)

##############################
## complete plot
for n in range(1):
    print('cell number = ' , n)
    #figure, axes = plt.subplots(1,4)
    # create a figure for every neuron
    for time_index in range(0,video.shape[0]):
        if time_index > 800*30 and time_index < 1000*33:
            figure = plt.figure()
            gs = plt.GridSpec(6,6)
            axes = figure.add_subplot(gs[0:3, 0:2])
            axes.set_title('Raw video', fontsize = 15)
            axes.imshow(video[time_index, :, :], cmap='gray',vmin=0.01, vmax=movie_max/2)
            counter = 0
            for c in coordinates:
                if counter:
                    v = c['coordinates']
                    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                    axes.plot(*v.T, c=colors[int(stimuli_vector_1[time_index])])
                counter = counter + 1
            #axes.set_xlabel('Pixel',fontsize = 12)
            axes.set_ylabel('Pixel',fontsize = 12)

            axes = figure.add_subplot(gs[3:6, 0:2])
            axes.set_title('CaImAn model', fontsize = 15)
            axes.imshow(Y_rec[time_index, :, :], cmap='gray',vmin=0.0, vmax=Y_rec_max/5)
            # pos = axes.imshow(Y_rec[time_index, :, :], cmap='gray',vmin=0, vmax=Y_rec_max)
            # figure.colorbar(pos, ax=axes)
            counter = 0
            for c in coordinates:
                if counter:
                    v = c['coordinates']
                    c['bbox'] = [np.floor(np.nanmin(v[:, 1])), np.ceil(np.nanmax(v[:, 1])),
                                 np.floor(np.nanmin(v[:, 0])), np.ceil(np.nanmax(v[:, 0]))]
                    axes.plot(*v.T, c=colors[int(stimuli_vector_1[time_index])])
                counter = counter + 1
            axes.set_xlabel('Pixel',fontsize = 12)
            axes.set_ylabel('Pixel',fontsize = 12)

            axes = figure.add_subplot(gs[0:2, 2:6])
            axes.set_title('Calcium Trace', fontsize = 15)
            axes.plot(temporal_variable, total_activity, c='k')
            axes.set_xlabel('t [s]', fontsize=15)
            axes.set_yticks([])
            axes.set_ylabel('Actvivity', fontsize=15)

            for sound_index in range(len(sounds_list)):
                for j in range(0,sounds_list[sound_index].shape[0]):
                    sound_onset = int(sounds_list[sound_index][j] + timeline[1])
                    sound_end = int(sound_onset  + stim_lenght  * srate)
                    axes.plot(np.arange(sound_onset,sound_end)/srate,total_activity[sound_onset:sound_end], c = 'k')
                    iti_onset = sound_end
                    iti_end = int(sound_end + iti_lenght*srate)
                    axes.plot(np.arange(iti_onset,iti_end)/srate,total_activity[iti_onset:iti_end], c = colors[sound_index+1])
            axes.vlines(time_index/srate,0,0.2,'k', linestyles= 'dotted')

            axes = figure.add_subplot(gs[3:6, 3:5])
            axes.plot(temporal_variable[time_index - int(srate * 2):time_index + int(srate * 2)], total_activity[time_index - int(srate * 2):time_index + int(srate * 2)], c = 'k')
            axes.set_xlabel('t [s]', fontsize=12)
            #axes.set_ylim([0,1])
            axes.set_ylabel('Actvivity', fontsize=12)
            rect = Rectangle((temporal_variable[time_index - int(srate)],0),2,1, fill=True, color=color2[int(stimuli_vector_1[time_index])], linestyle='-', linewidth=2, alpha = alpha_vector[time_index])
            axes.vlines(time_index/srate ,0,0.2,linestyles ="dotted",color = 'k')
            axes.add_patch(rect)

            figure.set_size_inches([15, 10])

            figure.savefig(figure_path + 'movie_'+ f'{n}' + '_' + f'{100000+time_index}' + '.png')
            plt.close()
            if time_index % 330 == 0:
                print(time_index)

