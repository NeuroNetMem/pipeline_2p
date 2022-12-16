
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
file_path = '/ceph/imaging1/arie/zstack/20221205_zstack_toms/'
file_name = '20221205_429420_zstack_00004.tif'

# n_processes = psutil.cpu_count()
# #cm.cluster.stop_server()
# # Start a new cluster
# c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
#                                                  n_processes=n_processes,
#                                                  single_thread=False)

movie = cm.load(file_path + file_name)

frames_number = np.arange(0,movie.shape[0]-6,6)

figure, axes = plt.subplots(3,2)

for i in range(3):
    for j in range(2):
        zstack_index = i*2 + j
        print(frames_number + zstack_index)
        zstack = movie[ frames_number + zstack_index ,:,:]
        axes[i,j].imshow(np.mean(zstack,axis=0),cmap = 'gray')# vmin = -200, vmax = 3000)
        axes[i,j].set_xticklabels([])
        axes[i,j].set_yticklabels([])
        axes[i,j].set_aspect('equal')
        plt.subplots_adjust(wspace=None, hspace=None)
