from pathlib import Path
import numpy as np
import caiman as cm
from caiman.source_extraction.cnmf.cnmf import load_CNMF
import pickle

def make_output_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+'/cropping').mkdir(parents=True, exist_ok=True)
    Path(path+'/motion_correction').mkdir(parents=True, exist_ok=True)
    Path(path+'/alignment').mkdir(parents=True, exist_ok=True)
    Path(path+'/source_extraction').mkdir(parents=True, exist_ok=True)
    Path(path+'/component_evaluation').mkdir(parents=True, exist_ok=True)
    print(f'built output directory @ {path}')
    return


def compute_metrics(movie,metrics_list):
    '''
    inputs: caiman movie, list of metric keywords
    '''
    
    metric_func = {'mean_image': lambda x: np.mean(x, axis=0),
                   'corr_pnr_images': lambda x: cm.summary_images.correlation_pnr(x,swap_dim=False),
                   'frame_average': lambda x: np.mean(x, axis=(1,2)),
                   
                  }
    
    computed_metrics = {}
    
    for m in metrics_list:
        if m not in metric_func.keys():
            print(f'{m} is not a valid metric keyword. Skipping.')
            continue
        else:
            computed_metrics[m] = metric_func[m](movie)

    
    return computed_metrics


def save_preprocessed_data(cnmf_file,output_path):
    '''
    Takes a cnmf.hdf5 file with the results of the preprocessing and saves it to the specified path as 'neural_data.pickle'
    '''
    
    output_path = Path(output_path)
    
    cnmf = load_CNMF(cnmf_file)
    ests = cnmf.estimates

    
    neural_data = {}
    neural_data['traces'] = ests.C[ests.idx_components]
    neural_data['footprints'] = ests.A[:,ests.idx_components].toarray().reshape(ests.dims[0],ests.dims[1],len(ests.idx_components))
    neural_data['df_f'] = ests.F_dff[ests.idx_components]
    
    spikes = []
    for trace in ests.S[ests.idx_components]:
        spikes.append(np.where(trace>0)[0])
    neural_data['deconvolved'] =  spikes
    
    # fake-plot the components to estimate coordinates. Is there a direct function? No trace of it in the documentation
    ests.plot_contours(idx=ests.idx_components)
    
    neural_data['positions'] = np.asarray([i['CoM'] for i in ests.coordinates])[ests.idx_components]
    neural_data['contour'] = [l['coordinates'] for i,l in enumerate(ests.coordinates) if i in ests.idx_components]
    

    save_path = output_path.joinpath('neural_data.pickle')
    filehandler = open(save_path, 'wb') 
    pickle.dump(neural_data, filehandler)
    filehandler.close()
    
    return







