
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


dir_path = '/ceph/imaging1/melisa/photon2_test/'
file_name_original = 'recording_20211112_00001.tif'
file_name_mode1 = 'recording_20211112_00001_01.tif'
file_name_mode2 = 'recording_20211112_00001_02.tif'

#load tif file
m_orig = cm.load(dir_path + file_name_original)
m_mode1 = cm.load(dir_path + file_name_mode1)
m_mode2 = cm.load(dir_path + file_name_mode2)

figure, axes = plt.subplots(1,3)
axes[0].imshow(np.mean(m_orig,axis = 0),cmap = 'gray')
axes[1].imshow(np.mean(m_mode1,axis = 0),cmap = 'gray')
axes[2].imshow(np.mean(m_mode2,axis = 0),cmap = 'gray')


corr_0 = m_orig.local_correlations(eight_neighbours=True, swap_dim=False)
corr_1 = m_mode1.local_correlations(eight_neighbours=True, swap_dim=False)
corr_2 = m_mode2.local_correlations(eight_neighbours=True, swap_dim=False)

figure, axes = plt.subplots(1,3)

axes[0].imshow(corr_0, cmap = 'gray')
axes[1].imshow(corr_1, cmap = 'gray')
axes[2].imshow(corr_2, cmap = 'gray')

