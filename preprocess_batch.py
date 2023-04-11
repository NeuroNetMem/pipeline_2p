

raw_data_path = Path('/ceph/imaging1/arie')
temp_output_path = Path('/scratch/dspalla/2p_data')
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
        
        #make temp dir
        save_path = temp_output_path.joinpath(f'{animal}/{date}')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        #make output dir
        save_path = preprocessed_data_path.joinpath(f'{animal}/{date}')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        #Decode log and process behaviour
        
        #Run caiman pipeline 
        
        



#####






