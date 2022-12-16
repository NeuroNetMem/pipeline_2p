from random import randrange
import matplotlib.pylab as plt
import caiman as cm
from matplotlib.patches import Rectangle
import numpy as np

def temporal_evolution(file_name = None, output_file_name = None):
    '''

    '''

    movie_original = cm.load(file_name)

    figure = plt.figure(constrained_layout=True)
    gs = figure.add_gridspec(5, 6)

    figure_ax1 = figure.add_subplot(gs[0:2, 0:3])
    figure_ax1.set_title('ROI', fontsize = 15)
    figure_ax1.set_yticks([])
    figure_ax1.set_xticks([])

    figure_ax2 = figure.add_subplot(gs[2:5, 0:3])
    figure_ax2.set_xlabel('Time [s]', fontsize = 15)
    figure_ax2.set_ylabel('Pixel value', fontsize = 15)
    figure_ax2.set_title('Temporal Evolution', fontsize = 15)
    #figure_ax2.set_ylim((300,1000))

    figure_ax1.imshow(np.mean(movie_original,axis=0), cmap = 'gray')
    color = ['b', 'r' , 'g', 'c', 'm']
    for i in range(5):
        x = randrange(movie_original.shape[1]-5)+5
        y = randrange(movie_original.shape[2]-5)+5
        [x_, _x, y_, _y] = [x-5,x+5,y-5,y+5]
        rect = Rectangle((y_, x_), _y - y_, _x - x_, fill=False, color=color[i], linestyle='-', linewidth=2)
        figure_ax1.add_patch(rect)
        figure_ax2.plot(np.arange(0,movie_original.shape[0],)/10, movie_original[:,x,y], color = color[i])

        figure_ax_i = figure.add_subplot(gs[i, 4:])
        figure_ax_i.hist(movie_original[:,x,y],50, color = color[i])
        #figure_ax_i.set_xlim((300,1000))
        figure_ax_i.set_ylabel('#')
        figure_ax_i.set_xlabel('Pixel value')
        figure_ax_i.set_yscale('log')
        figure_ax_i.set_xscale('log')

    figure.set_size_inches([5.,5.])
    figure.savefig(output_file_name)

    return

