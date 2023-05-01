from shutil import rmtree
from pathlib import Path
import glob
import logreader.logreader as lr
import pipeline.functions as fs


raw_data_path = Path('/ceph/imaging1/arie')
params_folder = raw_data_path.joinpath('preprocess_params')
temp_output_path = Path('/scratch/dspalla/2p_data')
preprocessed_data_path = Path('/ceph/imaging1/davide/2p_data')


# SESSION TO PREPROCESS
sessions = {'441406_fiano':['20230301','20230306','20230307','20230308','20230309','20230315','20230316','20230317','20230320','20230321'],
            '441394_ribolla':['20230301','20230306','20230307','20230308','20230309','20230315','20230316','20230317','20230320','20230321']
           }

# PREPROCESSING STEPS
preprocess_vr_data = False
preprocess_2p_video = True



for animal in sessions.keys():
    animal_num = animal.split('_')[0]
    
    print(f'PROCESSING {animal} ...')
    
    for date in sessions[animal]:
        print(f'session date: {date}')
        
        #load preprocesssing parameters for that session
        parameters = fs.load_session_parameters(animal = animal, date = date,params_folder=params_folder)
        
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
        temp_path = temp_output_path.joinpath(f'{animal}/{date}')
        Path(temp_path).mkdir(parents=True, exist_ok=True)
        
        #make output dir
        output_path = preprocessed_data_path.joinpath(f'{animal}/{date}')
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        #Decode log and process behaviour
        if preprocess_vr_data:
            try:
                vr_data = lr.preprocess_vr_data(tif_file = tif_file, log_file=log_file)
                lr.save_processed_vr_data(output_path,vr_data)
            except:
                print(f'Error in session {animal}_{date}, skipping ...')
                continue
        
        #Run caiman pipeline 
        if preprocess_2p_video:
            
            #make temp dir
            temp_path = temp_output_path.joinpath(f'{animal}/{date}')
            Path(temp_path).mkdir(parents=True, exist_ok=True)
            
            fs.preprocess_video(input_video=tif_file,
                            output_folder=output_path,
                            parameters=parameters,
                            temp_folder=temp_path)
        
        
        
        
        






