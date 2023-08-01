import pandas as pd
import numpy as np
import tifffile


def read_tiff_slice(tif_file,start_frame=0,end_frame=None):
    """
    Read a slice of frames from a TIFF video file and return it as a 3D NumPy array (time x pixels x pixels).

    Parameters:
        tif_file (str): The path to the TIFF video file.
        start_frame (int, optional): The index of the first frame to read (default is 0).
        end_frame (int, optional): The index of the last frame to read (default is None).
                                   If not provided, the function will read all frames until the end of the video.

    Returns:
        numpy.ndarray: A 3D NumPy array representing the slice of frames from the TIFF video.
                       The array has dimensions (num_frames, height, width).

    Note:
        - The function uses the 'tifffile' library to read the TIFF video.
        - If 'end_frame' is not provided, all frames from 'start_frame' to the end of the video will be read.
    """

    
    with tifffile.TiffFile(tif_file) as tif:
        
        if end_frame == None:
            print('No value provided for last frame, using video end, this can take a while ...')
            end_page = len(tif.pages)
            
    
        movie = [tif.pages[page].asarray() for page in range(start_frame,end_frame)]
        movie = np.dstack(movie).transpose(2,0,1)
    
    return movie


def make_tif_list(xls_file,animal_names):
    '''list all the tif files present in the xls info filefor each of the given animals'''

    data = pd.read_excel(xls_file)
    root_folder = '/ceph/imaging1/arie'
    tif_file_list = []

    for animal in animal_names:
        subset = data[data['name']==animal]
        print (f'{len(subset)} files found for mouse: {animal}')
        for i in range(len(subset)):
            animal_num = subset['mouse'].iloc[i]
            year = subset['year'].iloc[i]
            month = str(subset['month'].iloc[i]).zfill(2)
            day = str(subset['day'].iloc[i]).zfill(2)
            rec = subset['rec'].iloc[i]
            tif_file = subset['filename.tif'].iloc[i]
            session = f'{year}{month}{day}'
            tif_path = root_folder+f'/{animal_num}_{animal}/{session}_{animal_num}/{tif_file}.tif'
            tif_file_list.append(tif_path)
    
    return tif_file_list

