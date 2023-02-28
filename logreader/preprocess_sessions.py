import logreader as lr
from pathlib import Path
import pickle
import tifffile
import glob
import numpy as np
from scipy.io import savemat

animals = []
dates = []

raw_data_path = Path('/ceph/imaging1/arie')
preprocessed_data_path = Path('/ceph/imaging1/davide/2p_data')


# SESSION TO PREPROCESS
sessions ={'429420_toms': ['20230202','20230203','20230210','20230211'],
           '429419_croc': ['20230202','20230203','20230210','20230211']}


for animal in sessions.keys():
    animal_num = animal.split('_')[0]
    
    print(f'PROCESSING {animal} ...')
    
    for date in sessions[animal]:
        print(f'session date: {date}')
        
        
        
        session_path = raw_data_path.joinpath(f'{animal}/{date}_{animal_num}')
        print(session_path)
        
        try:
            tif_file = glob.glob(str(session_path)+'/*.tif')[0]
            log_file = glob.glob(str(session_path)+'/*.b64')[0]
        
        except IndexError:
            print('Files not found, skipping session')
        
        
        print(f'log file: {log_file}')
        print(f'tif file: {tif_file}')
        
        #make output dir
        save_path = preprocessed_data_path.joinpath(f'{animal}/{date}')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        decoded_log = lr.create_bp_structure(log_file)
        tif_header = lr.read_tif_header(tif_file)
        frames = tif_header['frame_ts']
        
        #save decoded log
        savemat(preprocessed_data_path.joinpath('decoded_log.mat'),decoded_log)
        
        #save frame timestamps
        with open(preprocessed_data_path.joinpath('tif_header.pickle'),'wb') as file:
                  pickle.dump(tif_header,file, protocol=pickle.HIGHEST_PROTOCOL)
    
        
        digital_in = decoded_log['digitalIn'].astype(int)
        digital_out = decoded_log['digitalOut'].astype(int)
        digital_scan_signal = digital_in[:,6]
        log_times = decoded_log['startTS']
        
        
        sync_times = lr.compute_sync_times(digital_scan_signal,log_times,frames)

        tm = lr.build_trial_matrix(digital_in,digital_out)
        
        position = decoded_log['longVar'][:,1].astype(int)
        lick_onsets = lr.compute_onsets(digital_out[:,-2])
        lick_offsets = lr.compute_offsets(digital_out[:,-2])
        reward_onsets = lr.compute_onsets(digital_out[:,0])
        reward_offsets = lr.compute_offsets(digital_out[:,0])

        data = {'time':sync_times,'position':position.astype('int'),
                'lick_onsets':lick_onsets,'lick_offsets':lick_offsets,
                'reward_onsets':reward_onsets,'reward_offsets':reward_offsets}
        
        # save preprocessed behaviour
        filename = save_path.joinpath('behaviour_data.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
        # save trial matrix
        filename = save_path.joinpath('trial_data.csv')
        tm.to_csv(filename,index=False)
            
        