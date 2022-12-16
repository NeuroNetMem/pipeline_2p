'''
Created on Wed Jan 19 2022
@Melisa

Create a set of function to save the processed data using Caiman
'''

import os
import logging
import data_base as db
import numpy as np
import math
import psutil

import caiman as cm
from caiman.motion_correction import MotionCorrect, high_pass_filter_space
from caiman.source_extraction.cnmf import params as params
from caiman.source_extraction import cnmf
from caiman.source_extraction.cnmf.cnmf import load_CNMF


def cropping_interval():
    '''
    This function ask the user for cropping paramenters
    :param None:
    :return: dictionary with a new assignment to cropping_paramenters
    '''
    x1 = int(input("Limit X1 : "))
    x2 = int(input("Limit X2 : "))
    y1 = int(input("Limit Y1 : "))
    y2 = int(input("Limit Y2 : "))
    parameters_cropping = {'crop_spatial': True, 'cropping_points_spatial': [y1, y2, x1, x2], 'segmentation': False,
                           'crop_temporal': False, 'cropping_points_temporal': []}
    #print(parameters_cropping)
    return parameters_cropping

def run_cropper(row, parameters):
    '''
    This function takes in a decoded analysis state and crops it according to
    specified cropping points.

    Args:
        index: tuple
            The index of the analysis state to be cropped.
        row: pd.DataFrame object
            The row corresponding to the analysis state to be cropped.

    Returns
        row: pd.DataFrame object
            The row corresponding to the cropped analysis state.
    '''

    row_local = row.copy()
    name = row_local.name
    states_db = db.open_data_base()

    input_tif_file_path = eval(row_local.loc['raw_output'])['main']
    if not os.path.isfile(input_tif_file_path):
        logging.error('File not found. Cancelling motion correction.')
        return row_local

    ### write used parameters in the data base
    row_local.loc['cropping_parameters'] = str(parameters)
    ### update motion correction version in the data base
    row_local = db.modify_data_base_row_name(row_local, states_db, 0)
    name = row_local.name

    ### create output data for data base
    data_dir = '/ceph/imaging1/melisa/photon2_test/data_processing/cropping/main/'  ### replace this in OS
    file_name = db.create_file_name(0, name)
    output_tif_file_path = data_dir + file_name +  '.tif'

    # Create a dictionary with the output
    output = {
        'main': output_tif_file_path,
        }

    # Spatial copping
    m = cm.load(input_tif_file_path)
    [x_, _x, y_, _y] = parameters['cropping_points_spatial']
    m = m[:, x_:_x, y_:_y]

    # Save the movie
    m.save(output_tif_file_path)
    # Write necessary variables to the trial index and row_local
    row_local.loc['cropping_parameters'] = str(parameters)
    row_local.loc['cropping_output'] = str(output)

    return row_local


def run_motion_correction(row,parameters,dview):

    '''
    Runs motion correction and returns the modified entry for the data base
    :param entry:
    :param parameters:
    :param dview:
    :return:
    '''

    row_local = row.copy()
    name = row_local.name
    states_db = db.open_data_base()

    if not parameters['pw_rigid']:
        parameters['save_movie_rig'] = True

    input_tif_file_path = eval(row_local.loc['cropping_output'])['main']

    if not os.path.isfile(input_tif_file_path):
        logging.error('File not found. Cancelling motion correction.')
        return row_local

    ### write used parameters in the data base
    row_local.loc['motion_correction_parameters'] = str(parameters)
    ### update motion correction version in the data base
    row_local = db.modify_data_base_row_name(row_local, states_db, 1)
    name = row_local.name

    ### create output data for data base
    data_dir = '/ceph/imaging1/melisa/photon2_test/data_processing/motion_correction/'  ### replace this in OS
    file_name = db.create_file_name(1, name)
    output_meta_pkl_file_path = data_dir + db.create_file_path('meta', name, 1, '.pkl')

    # Create a dictionary with the output
    output = { 'meta': {'other':output_meta_pkl_file_path} }

    # Calculate movie minimum to subtract from movie
    min_mov = np.min(cm.load(input_tif_file_path))
    # Apply the parameters to the CaImAn algorithm
    caiman_parameters = parameters.copy()
    caiman_parameters['min_mov'] = min_mov
    opts = params.CNMFParams(params_dict = caiman_parameters)

    # Rigid motion correction (in both cases)
    logging.info(f'{name} Performing rigid motion correction')
    # Create a MotionCorrect object
    mc = MotionCorrect([input_tif_file_path], dview=dview, **opts.get_group('motion'))
    # Perform rigid motion correction
    mc.motion_correct_rigid(save_movie=parameters['save_movie_rig'], template=None)
    # Obtain template, rigid shifts and border pixels
    total_template_rig = mc.total_template_rig
    shifts_rig = mc.shifts_rig
    output['meta']['cropping_points'] = [0,0,0,0]

    if parameters['save_movie_rig']:
        # Load the movie saved by CaImAn, which is in the wrong
        # directory and is not yet cropped
        logging.info(f'{name} Loading rigid movie for cropping')
        m_rig = cm.load(mc.fname_tot_rig[0])
        logging.info(f'{name} Loaded rigid movie for cropping')
        # Get the cropping points determined by the maximal rigid shifts
        x_, _x, y_, _y = get_crop_from_rigid_shifts(shifts_rig)
        output['meta']['cropping_points'] = [x_, _x, y_, _y]
        # Crop the movie
        logging.info(f'{name} Cropping and saving rigid movie with cropping points: [x_, _x, y_, _y] = {[x_, _x, y_, _y]}')
        m_rig = m_rig.crop(x_, _x, y_, _y, 0, 0)
        # Save the movie
        fname_tot_rig = m_rig.save(data_dir + 'main/' + file_name + '_rig' + '.mmap', order='C')
        logging.info(f'{name} Cropped and saved rigid movie as {fname_tot_rig}')
        # Remove the remaining non-cropped movie
        os.remove(mc.fname_tot_rig[0])

    if parameters['pw_rigid']:
        logging.info(f'{name} Performing piecewise-rigid motion correction')
        # Perform non-rigid (piecewise rigid) motion correction. Use the rigid result as a template.
        mc.motion_correct_pwrigid(save_movie=True, template=total_template_rig)
        # Obtain template and filename
        total_template_els = mc.total_template_els
        fname_tot_els = mc.fname_tot_els[0]

        # Load the movie saved by CaImAn, which is in the wrong
        # directory and is not yet cropped
        logging.info(f'{name} Loading pw-rigid movie for cropping')
        m_els = cm.load(fname_tot_els)
        logging.info(f'{name} Loaded pw-rigid movie for cropping')
        # Get the cropping points determined by the maximal rigid shifts
        x_, _x, y_, _y = get_crop_from_pw_rigid_shifts(np.array(mc.x_shifts_els),
                                                       np.array(mc.y_shifts_els))

        output['meta']['cropping_points'] = [x_, _x, y_, _y]
        # Crop the movie
        logging.info(f'{name} Cropping and saving pw-rigid movie with cropping points: [x_, _x, y_, _y] = {[x_, _x, y_, _y]}')

        #m_els = m_els.crop(x_, _x, y_, _y, 0, 0)
        x1, x2, x3 = m_els.shape
        m_els = m_els[:, x_+5:x2 - (_x+5), y_+5:x3 - (_y+5)]
        # Save the movie
        fname_tot_els = m_els.save(data_dir + 'main/' + file_name + '_els' + '.mmap', order='C')
        logging.info(f'{name} Cropped and saved rigid movie as {fname_tot_els}')
        # Remove the remaining non-cropped movie
        os.remove(mc.fname_tot_els[0])

        # Store the total path in output
        output['main'] = fname_tot_els
    else:
        output['main'] = fname_tot_rig



    # Write necessary variables to the trial index and row
    row_local.loc['motion_correction_output'] = str(output)
    row_local.loc['motion_correction_parameters'] = str(parameters)

    return row_local

def get_crop_from_rigid_shifts(shifts_rig):
    x_ = int(round(abs(np.array(shifts_rig)[:, 1].max()) if np.array(shifts_rig)[:, 1].max() > 0 else 0))
    _x = int(round(abs(np.array(shifts_rig)[:, 1].min()) if np.array(shifts_rig)[:, 1].min() < 0 else 0))
    y_ = int(round(abs(np.array(shifts_rig)[:, 0].max()) if np.array(shifts_rig)[:, 0].max() > 0 else 0))
    _y = int(round(abs(np.array(shifts_rig)[:, 0].min()) if np.array(shifts_rig)[:, 0].min() < 0 else 0))
    return x_, _x, y_, _y


def get_crop_from_pw_rigid_shifts(x_shifts_els, y_shifts_els):
    x_ = int(round(abs(x_shifts_els.max()) if x_shifts_els.max() > 0 else 0))
    _x = int(round(abs(x_shifts_els.min()) if x_shifts_els.min() < 0 else 0))
    y_ = int(round(abs(y_shifts_els.max()) if y_shifts_els.max() > 0 else 0))
    _y = int(round(abs(x_shifts_els.min()) if x_shifts_els.min() < 0 else 0))
    return x_, _x, y_, _y


def run_alignment(rows,parameters,dview):
    '''
    This is the main function for the alignment step. It applies methods
    from the CaImAn package used originally in motion correction
    to do alignment.
    :param row:
    :param parameters:
    :param dview:
    :return:
    '''
    df= rows.copy()
    df = df.sort_values(by=db.multi_index_structure)

    # Determine the mouse and date
    name = df.iloc[0].name
    mouse, year, month, date, *r = name
    alignment_v = name[5]
    # Define the output .mmap file name
    file_name = f'mouse_{mouse}_year_{year}_month_{month}_date_{date}_v{alignment_v}'
    output_mmap_file_path = f'/ceph/imaging1/melisa/photon2_test/data_processing/alignment/main/{file_name}.mmap'

    output = { 'meta': {'other': output_mmap_file_path} }

    # Get necessary parameters from all motion corrected files
    motion_correction_parameters_list = []
    motion_correction_output_list = []
    input_mmap_file_list = []
    x_ = []
    _x = []
    y_ = []
    _y = []
    for idx, row in df.iterrows():
        motion_correction_parameters_list.append(eval(row.loc['motion_correction_parameters']))
        motion_correction_output = eval(row.loc['motion_correction_output'])
        motion_correction_output_list.append(motion_correction_output)
        input_mmap_file_list.append(motion_correction_output['main'])
        [x1,x2,y1,y2] = motion_correction_output['meta']['cropping_points']
        x_.append(x1)
        _x.append(x2)
        y_.append(y1)
        _y.append(y2)

    new_x1 = max(x_)
    new_x2 = max(_x)
    new_y1 = max(y_)
    new_y2 = max(_y)
    m_list = []
    for i in range(len(input_mmap_file_list)):
        m = cm.load(input_mmap_file_list[i])
        motion_correction_output = eval(df.iloc[i].loc['motion_correction_output'])
        [x1,x2,y1,y2] = motion_correction_output['meta']['cropping_points']
        m = m.crop(new_x1 - x1, new_x2 - x2, new_y1 - y1, new_y2 - y2, 0, 0)
        m_list.append(m)

    # Concatenate them using the concat function
    m_concat = cm.concatenate(m_list, axis=0)
    data_dir = f'/ceph/imaging1/melisa/photon2_test/data_processing/alignment/main/'
    file_name = db.create_file_name(2, name)
    fname= m_concat.save(data_dir + file_name + '.mmap', order='C')

    ### crop first video to create a template
    m0 = cm.load(input_mmap_file_list[0])
    [x1, x2, y1, y2] = motion_correction_output_list[0]['meta']['cropping_points']
    m0 = m0.crop(new_x1 - x1, new_x2 - x2, new_y1 - y1, new_y2 - y2, 0, 0)
    m0_filt = cm.movie(
        np.array([high_pass_filter_space(m_, parameters['gSig_filt']) for m_ in m0]))
    template0 = cm.motion_correction.bin_median(
        m0_filt.motion_correct(5, 5, template=None)[0])  # may be improved in the future

    # Setting the parameters
    opts = params.CNMFParams(params_dict=parameters)

    # Create a motion correction object for the aligned movie
    mc = MotionCorrect(fname, dview=dview, **opts.get_group('motion'))
    # run motion correction
    mc.motion_correct(template=template0, save_movie=True)

    # Cropping borders
    x_ = math.ceil(abs(np.array(mc.shifts_rig)[:, 1].max()) if np.array(mc.shifts_rig)[:, 1].max() > 0 else 0)
    _x = math.ceil(abs(np.array(mc.shifts_rig)[:, 1].min()) if np.array(mc.shifts_rig)[:, 1].min() < 0 else 0)
    y_ = math.ceil(abs(np.array(mc.shifts_rig)[:, 0].max()) if np.array(mc.shifts_rig)[:, 0].max() > 0 else 0)
    _y = math.ceil(abs(np.array(mc.shifts_rig)[:, 0].min()) if np.array(mc.shifts_rig)[:, 0].min() < 0 else 0)

    # Load the motion corrected movie into memory
    movie= cm.load(mc.fname_tot_rig[0])
    # Crop all movies to those border pixels
    x1, x2 ,x3 = movie.shape
    movie = movie[:, x_+5:x2-(_x+5), y_+5:x3-(_y+5)]
    #movie.crop(x_, _x, y_, _y, 0, 0)
    output['meta']['cropping_points'] = [x_, _x, y_, _y]

    #save motion corrected and cropped movie
    output_mmap_file_path_tot = movie.save(data_dir + file_name  + '.mmap', order='C')
    logging.info(f'{name} Cropped and saved rigid movie as {output_mmap_file_path_tot}')
    # Save the path in teh output dictionary
    output['main'] = output_mmap_file_path_tot
    # Remove the remaining non-cropped movie
    os.remove(mc.fname_tot_rig[0])

    # Create a timeline and store it
    timeline = np.zeros((len(m_list)+1),)
    meta_data_dir = f'/ceph/imaging1/melisa/photon2_test/data_processing/alignment/meta/'
    timeline_np_file_path =  meta_data_dir  + file_name + '_' + f'{len(m_list)}' + '.npy'
    for i in range(0, len(m_list)):
        m = m_list[i]
        timeline[i+1] = timeline[i-1]+ m.shape[0]
    np.save(timeline_np_file_path,timeline)
    output['meta']['timeline'] = timeline_np_file_path

    for idx, row in df.iterrows():
        df.loc[idx, 'alignment_output'] = str(output)
        df.loc[idx, 'alignment_parameters'] = str(parameters)

    return df


def run_source_extraction(row, parameters, states_db, dview, multiple_files = False):
    '''
    This is the function for source extraction.
    Its goal is to take in a .mmap file,
    perform source extraction on it using cnmf-e and save the cnmf object as a .hdm5 file.
    :param row:
    :param parameters:
    :param dview:
    :return:
    '''

    row_local = row.copy()
    row_local.loc['source_extraction_parameters'] = str(parameters)
    ### update motion correction version in the data base
    row_local = db.modify_data_base_row_name(row_local, states_db, 3)
    name = row_local.name

    # Determine input path
    input_mmap_file_path =  eval(row_local.loc['motion_correction_output'])['main']
    if multiple_files:
        input_mmap_file_path = eval(row_local.loc['alignment_output'])['main']
    #input_mmap_file_path = eval(row_local.loc['motion_correction_output'])['main']

    # Determine output paths
    file_name = db.create_file_name(3, name)
    data_dir = f'/ceph/imaging1/melisa/photon2_test/data_processing/source_extraction/'
    output_file_path = data_dir + f'main/{file_name}.hdf5'

    # Create a dictionary with parameters
    output = {
        'main': output_file_path,
        'meta': { }
    }

    # Load memmory mappable input file
    if os.path.isfile(input_mmap_file_path):
        Yr, dims, T = cm.load_memmap(input_mmap_file_path)
        #        logging.debug(f'{index} Loaded movie. dims = {dims}, T = {T}.')
        images = Yr.T.reshape((T,) + dims, order='F')
    else:
        logging.warning(f'{name} .mmap file does not exist. Cancelling')
        return row_local

    cn_filter, pnr = cm.summary_images.correlation_pnr(images[::1], gSig=parameters['gSig'][0], swap_dim=False)
    gSig = parameters['gSig'][0]
    corr_npy_file_path = data_dir + f'/meta/corr/{db.create_file_name(3, name)}_gSig_{gSig}.npy'
    pnr_npy_file_path = data_dir + f'/meta/pnr/{db.create_file_name(3, name)}_gSig_{gSig}.npy'
    with open(corr_npy_file_path, 'wb') as f:
        np.save(f, cn_filter)
    with open(pnr_npy_file_path, 'wb') as f:
        np.save(f, pnr)

    # Store the paths in the meta dictionary
    output['meta']['corr'] = {'main': corr_npy_file_path, 'meta': {}}
    output['meta']['pnr'] = {'main': pnr_npy_file_path, 'meta': {}}

    # SOURCE EXTRACTION
    logging.info(f'{name} Performing source extraction')
    n_processes = psutil.cpu_count()
    logging.info(f'{name} n_processes: {n_processes}')
    opts = params.CNMFParams(params_dict=parameters)
    cnm = cnmf.CNMF(n_processes=n_processes, dview=dview, params=opts)
    cnm.fit(images)

    # Save the cnmf object as a hdf5 file
    logging.info(f'{name} Saving cnmf object')
    cnm.save(output_file_path)

    # Write necessary variables in row and return
    row_local.loc['source_extraction_parameters'] = str(parameters)
    row_local.loc['source_extraction_output'] = str(output)

    return row_local

def run_component_evaluation(row, parameters, states_db, multiple_files = False):


    row_local = row.copy()
    row_local.loc['component_evaluation_parameters'] = str(parameters)
    ### update motion correction version in the data base
    row_local = db.modify_data_base_row_name(row_local, states_db, 4)
    name = row_local.name

    if multiple_files:
        motion_correction_output = eval(row_local.loc['alignment_output'])
    else:
        motion_correction_output = eval(row_local.loc['motion_correction_output'])

    source_extraction_output = eval(row_local.loc['source_extraction_output'])
    input_hdf5_file_path = source_extraction_output['main']
    input_mmap_file_path = motion_correction_output['main']

    # Determine output paths
    file_name = db.create_file_name(4, name)
    data_dir = f'/ceph/imaging1/melisa/photon2_test/data_processing/component_evaluation/'
    output_file_path = data_dir + f'main/{file_name}.hdf5'

    # Create a dictionary with parameters
    output = {
        'main': output_file_path,
        'meta': {},
    }

    # Load CNMF object (contains source extracted cells)
    cnm = load_CNMF(input_hdf5_file_path)

    # Load the original movie
    Yr, dims, T = cm.load_memmap(input_mmap_file_path)
    images = Yr.T.reshape((T,) + dims, order='F')

    # Set the parmeters
    cnm.params.set('quality', parameters)

    # Stop the cluster if one exists
    n_processes = psutil.cpu_count()
    try:
        cm.cluster.stop_server()
    except:
        pass

    # Start a new cluster
    c, dview, n_processes = cm.cluster.setup_cluster(backend='local',
                                                     n_processes=n_processes,
                                                     # number of process to use, if you go out of memory try to reduce this one
                                                     single_thread=False)
    # Evaluate components
    cnm.estimates.evaluate_components(images, cnm.params, dview=dview)

    logging.debug('Number of total components: ', len(cnm.estimates.C))
    logging.debug('Number of accepted components: ', len(cnm.estimates.idx_components))

    # Stop the cluster
    dview.terminate()

    # Save CNMF object
    cnm.save(output_file_path)

    # Write necessary variables to the trial index and row
    row_local.loc['component_evaluation_parameters'] = str(parameters)
    row_local.loc['component_evaluation_output'] = str(output)

    return row_local

