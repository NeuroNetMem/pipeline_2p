import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import ipywidgets as widgets
from ipywidgets import interact,fixed
import pandas as pd
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import caiman as cm


def plot_summary_images(data_path,animal,date):
    data_path = Path(data_path)
    
    try:
    
        with open(data_path.joinpath(f'{animal}/{date}/metrics.pickle'),'rb') as pfile:
            metrics = pickle.load(pfile)
        
        plt.figure(figsize=(10,5))

        plt.subplot(1,2,1)
        plt.title('mean image')
        x = np.asarray(metrics['mean_image'])
        plt.imshow(x,vmin=np.percentile(x.ravel(),3),vmax=np.percentile(x.ravel(),97),cmap=plt.cm.gnuplot2)
        plt.colorbar()

        plt.subplot(1,2,2)
        plt.title('corr image')
        x = np.asarray(metrics['corr_image'])
        plt.imshow(x,vmin=np.percentile(x.ravel(),3),vmax=np.percentile(x.ravel(),80),cmap=plt.cm.gnuplot2)
        plt.colorbar()

        plt.figure(figsize=(10,5))
        plt.title('Average luminance')
        x = np.asarray(metrics['frame_average'])
        plt.xlabel('time (frames)')
        plt.plot(x)

        plt.tight_layout()
    
    except FileNotFoundError:
        # Create a figure and axis with the desired image size
        fig, ax = plt.subplots(figsize=(10,10))
    
    # Set the background color
        
        error_text = f'Metrics files not found \n the session is either not preprocessed or \n saved elsewere'

        # Hide the axis ticks and labels
        ax.axis('off')

        # Set the text properties
        text_props = {'ha': 'center', 'va': 'center', 'fontsize': 40, 'color': 'k'}

        # Add the text in the center
        ax.text(0.5, 0.5, error_text, **text_props)
        
        


def interactive_summary_images(data_path,sessions):
    
    animals_list = sessions.keys()  
    dates_list = np.unique([item for key in sessions.keys() for item in sessions[key]])

    animal_dropdown = widgets.Dropdown(options=animals_list, description='Animal:')
    date_dropdown = widgets.Dropdown(options=dates_list, description='Date:')
    
    interact(plot_summary_images,data_path = fixed(str(data_path)),animal=animal_dropdown,date=date_dropdown);
    
    
def plot_contours_images(data_path,animal,session):
    data_path = Path(data_path)
    
    try:
    
        cnmf_file = data_path.joinpath(f'{animal}/{session}/cnmf.hdf5')
        cnmf = load_CNMF(cnmf_file)
        ests = cnmf.estimates

        metrics_file = data_path.joinpath(f'{animal}/{session}/metrics.pickle')
        with open(metrics_file,'rb') as pfile:
                    metrics = pickle.load(pfile)
                
        plt.figure(figsize=(20,10))
        ests.plot_contours(idx=ests.idx_components,cmap=plt.cm.gnuplot2)
        
        print(f'total # of components:{len(ests.idx_components)+len(ests.idx_components_bad)}')
        print(f'GOOD components:{len(ests.idx_components)}')
        print(f'BAD components:{len(ests.idx_components_bad)}')

        
                
        
        

    except FileNotFoundError as e:
        # Create a figure and axis with the desired image size
        fig, ax = plt.subplots(figsize=(10,10))
    
    # Set the background color
        
        error_text = str(e)

        # Hide the axis ticks and labels
        ax.axis('off')

        # Set the text properties
        text_props = {'ha': 'center', 'va': 'center', 'fontsize': 20, 'color': 'k'}

        # Add the text in the center
        ax.text(0.5, 0.5, error_text, **text_props)
        
def interactive_contours_images(data_path,sessions):
    
    animals_list = sessions.keys()  
    dates_list = np.unique([item for key in sessions.keys() for item in sessions[key]])

    animal_dropdown = widgets.Dropdown(options=animals_list, description='Animal:')
    date_dropdown = widgets.Dropdown(options=dates_list, description='Date:')
    
    interact(plot_contours_images,data_path = fixed(str(data_path)),animal=animal_dropdown,session=date_dropdown);
    

def compute_cnmf_metrics(estimates):
    
    out_dict = {}
    
    good_components = estimates.idx_components
    
    out_dict['n_neurons'] = len(good_components)
    out_dict['SNR_components'] = np.mean(estimates.SNR_comp[good_components])
    out_dict['r_values'] = np.mean(estimates.r_values[good_components])
    out_dict['cnn_preds'] = np.mean(estimates.cnn_preds[good_components])
    out_dict['pixels_sn'] = np.mean(estimates.sn[good_components])
    out_dict['neurons_sn'] = np.mean(estimates.neurons_sn[good_components])

    
    return out_dict




def compute_batch_summary(data_path,sessions,verbose=False):

    df = pd.DataFrame()

    for animal in sessions.keys():
        for date in sessions[animal]:
            try:
                cnmf_file = data_path.joinpath(f'{animal}/{date}/cnmf.hdf5')
                if verbose:
                    print(f'Computing metrics for: \n {cnmf_file}')
                cnmf = load_CNMF(cnmf_file)
                
                metrics = compute_cnmf_metrics(cnmf.estimates)
            
                metrics['animal'] = animal
                metrics['date'] = date
                
                if len(df) == 0:
                    df = pd.DataFrame.from_dict(metrics, orient='index').T
                    
                    
                    
                else:
                    metrics_df = pd.DataFrame.from_dict(metrics, orient='index').T
                    df = pd.concat([df,metrics_df],ignore_index=True)
                
                    
            except FileNotFoundError:
                # Handle the case where the file is not found. Prints error and includes nans in df
                print(f"File not found for {animal} - {date}, skipping")
                continue
                
                
    
    return df