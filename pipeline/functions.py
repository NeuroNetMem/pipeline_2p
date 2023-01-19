from pathlib import Path

def make_output_folder(path):
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+'/cropping').mkdir(parents=True, exist_ok=True)
    Path(path+'/motion_correction').mkdir(parents=True, exist_ok=True)
    Path(path+'/alignment').mkdir(parents=True, exist_ok=True)
    Path(path+'/source_extraction').mkdir(parents=True, exist_ok=True)
    Path(path+'/component_evaluation').mkdir(parents=True, exist_ok=True)
    print(f'built output directory @ {path}')
    return





