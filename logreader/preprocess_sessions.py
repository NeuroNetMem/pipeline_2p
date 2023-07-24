'''
This scrips lets the user define a batch of session to analyze, and preprocesses the VR data
of this list.
Note: this is old, please use `preprocess_batch.py` for a new version of the same functionality.
'''

from pathlib import Path
import pickle
import tifffile
import glob
import numpy as np
from logreader import preprocess_vr_data, save_processed_vr_data

animals = []
dates = []

raw_data_path = Path('/ceph/imaging1/arie')
preprocessed_data_path = Path('/ceph/imaging1/davide/2p_data')


# SESSION TO PREPROCESS
sessions = {'429420_toms': ['20230202', '20230203', '20230210', '20230211'],
            '429419_croc': ['20230202', '20230203', '20230210', '20230211']}


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

        # preprocess vr data
        vr_data = preprocess_vr_data(tif_file=tif_file, log_file=log_file)

        # save preprocessed data
        output_path = preprocessed_data_path.joinpath(f'{animal}/{date}')
        Path(output_path).mkdir(parents=True, exist_ok=True)
        save_processed_vr_data(output_path, vr_data)
