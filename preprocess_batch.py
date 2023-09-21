from shutil import rmtree
from pathlib import Path
import glob
import logreader.logreader as lr
import pipeline.functions as fs
import shutil


raw_data_path = Path('/ceph/imaging1/arie')
params_folder = raw_data_path.joinpath('preprocess_params')
temp_output_path = Path('/scratch/dspalla/2p_data')
preprocessed_data_path = Path('/ceph/imaging1/davide/2p_data_nb10')


# SESSION TO PREPROCESS
# 
sessions = {'429420_toms': ['20221208', '20221003', '20221014', '20230213', '20230210', '20221130', 
                            '20220928', '20230201', '20221117', '20230203', '20221205', '20230211', 
                            '20230214', '20221207', '20221206', '20221209','20221210', '20221118', 
                            '20221026', '20221202', '20230202', '20221122', '20221201', '20221027', 
                            '20221115', '20221030']
           }

# PREPROCESSING STEPS
preprocess_vr_data = True
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
            except Exception as e:
                print(f'Error in VR data for session {animal}/{date}')
                print(e)
                print('skipping ...')
        
        #Run caiman pipeline 
        if preprocess_2p_video:
            
            #make temp dir
            temp_path = temp_output_path.joinpath(f'{animal}/{date}')
            Path(temp_path).mkdir(parents=True, exist_ok=True)
            
            try:
                fs.preprocess_video(input_video=tif_file,
                                output_folder=output_path,
                                parameters=parameters,
                                temp_folder=temp_path)
            
            except Exception as e:
                print(f'Error in neural data for session {animal}/{date}')
                print(e)
                print(f'Cleaning temporary output directory: {str(temp_path)}')
                shutil.rmtree(str(temp_path))
                
                print('skipping ...')
        
        
        
        
        






