
import caiman as cm
import numpy as np
import os
import psutil
import logging
import data_base as db
import matplotlib.pylab as plt
from preprocessing import run_cropper, run_motion_correction, run_alignment, run_source_extraction, cropping_interval, run_component_evaluation
from caiman.source_extraction.cnmf.cnmf import load_CNMF

# n_processes = psutil.cpu_count()
# #cm.cluster.stop_server()
# # Start a new cluster
# c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
#                                                  n_processes=n_processes,
#                                                  single_thread=False)

figure_path = '/scratch/melisa/photon2_test/figures/'
file_path = '/ceph/imaging1/melisa/photon2_test/data/CAMILO/slice/'
file_name0 = 'xx0_000_000.tif'
file_name1 = 'xx0_000_001.tif'

slices_title = ['MEAN Zoom1','MEAN Zoom2', ' One Frame Zoom1', ' One Frame Zoom2']

movie0 = cm.load(file_path + file_name0)
movie1 = cm.load(file_path + file_name1)

figure, axes = plt.subplots(2,2)
axes[0,0].imshow(np.mean(movie0,axis = 0), cmap = 'gray')
axes[0,1].imshow(np.mean(movie1,axis = 0), cmap = 'gray')

axes[1,0].imshow(movie0[0,:,:], cmap = 'gray')
axes[1,1].imshow(movie1[0,:,:], cmap = 'gray')

for i in range(2):
    for j in range(2):
        axes[i,j].set_xticklabels([])
        axes[i,j].set_yticklabels([])
        axes[i,j].set_aspect('equal')
        axes[i,j].set_title(slices_title[i*2+j])

figure.savefig(figure_path + 'camilo_slice.png')
plt.show()



