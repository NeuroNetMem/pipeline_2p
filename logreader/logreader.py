'''
This file contains function for the handling of the log files outputted by VR acquisition system, 
as well as logic for reading the metadata of the miscroscope tiff files.

'''

import numpy as np
import base64
import struct
from cobs import cobs
from tqdm.notebook import tqdm
from collections import namedtuple
import tifffile
from ScanImageTiffReader import ScanImageTiffReader
import pandas as pd
import pickle
from scipy.io import savemat


def extract_frame_timestamps(tif_file):
    '''
    Reads the TIF header and extracts frame timestamps (in seconds).

    Parameters:
        tif_file (str): Path to the TIF file to read.

    Returns:
        list: A list containing the frame timestamps in seconds.
    '''
    tag_structure = {'image_description': 5,
                     'frame_timestamp': 3}  # position of required information in tif header
    frame_ts = []

    with tifffile.TiffFile(tif_file) as tif:
        for page in tqdm(tif.pages):
            # extract image description string
            description = page.tags.values(
            )[tag_structure['image_description']].value
            timestamp = float(description.split('\n')[tag_structure['frame_timestamp']].split(
                '=')[-1])  # fetch timestamp in image description
            frame_ts.append(timestamp)

    return frame_ts


def print_header_content(tif_file):
    '''
    Displays the header of the first page in the TIF file.

    Parameters:
        tif_file (str): Path to the TIF file to read.

    Returns:
        None

    Side Effects:
        Prints the header content of the first page in the TIF file.
    '''
    print('Header content of first page:')
    with tifffile.TiffFile(tif_file) as tif:
        for tag in tif.pages[0].tags.values():
            name, value = tag.name, tag.value
            print(f'{name} --> {value}')
    return


def read_tif_header(tif_file):
    '''
    Reads the TIF header and extracts frame timestamps (in seconds) and scanner synchronization data.

    Parameters:
        tif_file (str): Path to the TIF file to read.

    Returns:
        dict: A dictionary with the following keys:
            - 'frame_ts': A list containing frame timestamps in seconds.
            - 'i2c_data': A dictionary containing 'ts', 'value', and 'frame_n' keys with respective data.
    '''
    tag_structure = {'image_description': 5,
                     'frame_timestamp': 3,
                     'i2c_data': 14}  # position of required information in tif header
    frame_ts = []
    i2c_data = {'ts': [], 'value': [], 'frame_n': []}

    with tifffile.TiffFile(tif_file) as tif:
        for i, page in tqdm(enumerate(tif.pages)):
            # extract image description string
            description = page.tags.values(
            )[tag_structure['image_description']].value
            timestamp = float(description.split('\n')[tag_structure['frame_timestamp']].split(
                '=')[-1])  # fetch timestamp in image description
            frame_ts.append(timestamp)

            i2c_line = description.split('\n')[tag_structure['i2c_data']]
            data = i2c_line.split('=')[-1].strip(' {').strip('}').strip(' }')
            if len(data) > 0:
                i2c_data['ts'].append(float(data.split(',')[0]))
                i2c_data['value'].append(
                    int(data.split(',')[-1].strip(' [').strip(']')))
                i2c_data['frame_n'].append(i)

    return {'frame_ts': frame_ts, 'i2c_data': i2c_data}


def create_bp_structure(bp):
    '''
    Decode the log file and create a VR channel data structure.

    Parameters:
        bp (str): Path to the log file to read.

    Returns:
        dict: A dictionary containing the decoded log data with the following keys:
            - 'analog': Analog data.
            - 'digitalIn': Digital input data (flipped left to right).
            - 'digitalOut': Digital output data (flipped left to right).
            - 'startTS': Start timestamps in microseconds.
            - 'transmitTS': Transmit timestamps in microseconds.
            - 'longVar': States data.
            - 'packetNums': Packet IDs.

    Note:
        This function decodes will soon be updated with a better name and a more flexible logic.
    '''

    print('Decoding log file')
    # Format package
    DataPacketDesc = {'type': 'B',
                      'size': 'B',
                      'crc16': 'H',
                      'packetID': 'I',
                      'us_start': 'I',
                      'us_end': 'I',
                      'analog': '8H',
                      'states': '8l',
                      'digitalIn': 'H',
                      'digitalOut': 'B',
                      'padding': 'x'}

    DataPacket = namedtuple('DataPacket', DataPacketDesc.keys())
    DataPacketStruct = '<' + ''.join(DataPacketDesc.values())
    DataPacketSize = struct.calcsize(DataPacketStruct)

    # package with non-digital data
    dtype_no_digital = [
        ('type', np.uint8),
        ('size', np.uint8),
        ('crc16', np.uint16),
        ('packetID', np.uint32),
        ('us_start', np.uint32),
        ('us_end', np.uint32),
        ('analog', np.uint16, (8, )),
        ('states', np.uint32, (8, ))]

    # DigitalIn and DigitalOut
    dtype_w_digital = dtype_no_digital + \
        [('digital_in', np.uint16, (16, )), ('digital_out', np.uint8, (8, ))]

    # Creating array with all the data (differenciation digital/non digital)
    np_DataPacketType_noDigital = np.dtype(dtype_no_digital)
    np_DataPacketType_withDigital = np.dtype(dtype_w_digital)
    # Unpack the data as done on the teensy commander code
    num_lines = count_lines(bp)
    log_duration = num_lines/1000/60

    # Decode and create new dataset
    data = np.zeros(num_lines, dtype=np_DataPacketType_withDigital)
    non_digital_names = list(np_DataPacketType_noDigital.names)

    with open(bp, 'rb') as bf:
        for nline, line in enumerate(tqdm(bf, total=num_lines)):
            bl = cobs.decode(base64.b64decode(line[:-1])[:-1])
            dp = unpack_data_packet(bl, DataPacketStruct, DataPacket)

            data[non_digital_names][nline] = np.frombuffer(
                bl[:-4], dtype=np_DataPacketType_noDigital)
            digital_arr = np.frombuffer(bl[-4:], dtype=np.uint8)
            data[nline]['digital_in'] = np.hstack(
                [np.unpackbits(digital_arr[1]), np.unpackbits(digital_arr[0])])
            data[nline]['digital_out'] = np.unpackbits(
                np.array(digital_arr[2], dtype=np.uint8))
        # Check for packetID jumps
    jumps = np.unique(np.diff(data['packetID']))
    decoded = {"analog": data['analog'], "digitalIn": data['digital_in'][:, ::-1], "digitalOut": data['digital_out'][:, ::-1],
               "startTS": data['us_start'], "transmitTS": data['us_end'], "longVar": data['states'], "packetNums": data['packetID']}

    return decoded


def decode_logfile(logfile):
    '''
    Yet to be implemented
    '''

    raise NotImplementedError(
        'This function is yet to be implemented, use create_bp_structure instead')
    # print('Decoding log file')

    # digital_in_labels = []  # labels for digital input channels
    # digital_out_labels = []  # labels for digital output channels
    # analog_labels = []  # labels for analog channels

    # # Format package
    # DataPacketDesc = {'type': 'B',
    #                   'size': 'B',
    #                   'crc16': 'H',
    #                   'packetID': 'I',
    #                   'us_start': 'I',
    #                   'us_end': 'I',
    #                   'analog': '8H',
    #                   'states': '8l',
    #                   'digitalIn': 'H',
    #                   'digitalOut': 'B',
    #                   'padding': 'x'}

    # DataPacket = namedtuple('DataPacket', DataPacketDesc.keys())
    # DataPacketStruct = '<' + ''.join(DataPacketDesc.values())
    # DataPacketSize = struct.calcsize(DataPacketStruct)

    # # package with non-digital data
    # dtype_no_digital = [
    #     ('type', np.uint8),
    #     ('size', np.uint8),
    #     ('crc16', np.uint16),
    #     ('packetID', np.uint32),
    #     ('us_start', np.uint32),
    #     ('us_end', np.uint32),
    #     ('analog', np.uint16, (8, )),
    #     ('states', np.uint32, (8, ))]

    # # DigitalIn and DigitalOut
    # dtype_w_digital = dtype_no_digital + \
    #     [('digital_in', np.uint16, (16, )), ('digital_out', np.uint8, (8, ))]

    # # Creating array with all the data (differenciation digital/non digital)
    # np_DataPacketType_noDigital = np.dtype(dtype_no_digital)
    # np_DataPacketType_withDigital = np.dtype(dtype_w_digital)
    # # Unpack the data as done on the teensy commander code
    # num_lines = count_lines(logfile)
    # log_duration = num_lines/1000/60

    # # Decode and create new dataset
    # data = np.zeros(num_lines, dtype=np_DataPacketType_withDigital)
    # non_digital_names = list(np_DataPacketType_noDigital.names)

    # with open(logfile, 'rb') as bf:
    #     for nline, line in enumerate(tqdm(bf, total=num_lines)):
    #         bl = cobs.decode(base64.b64decode(line[:-1])[:-1])
    #         dp = unpack_data_packet(bl, DataPacketStruct, DataPacket)

    #         data[non_digital_names][nline] = np.frombuffer(
    #             bl[:-4], dtype=np_DataPacketType_noDigital)
    #         digital_arr = np.frombuffer(bl[-4:], dtype=np.uint8)
    #         data[nline]['digital_in'] = np.hstack(
    #             [np.unpackbits(digital_arr[1]), np.unpackbits(digital_arr[0])])
    #         data[nline]['digital_out'] = np.unpackbits(
    #             np.array(digital_arr[2], dtype=np.uint8))
    #     # Check for packetID jumps
    # jumps = np.unique(np.diff(data['packetID']))
    # decoded = {"analog": data['analog'],
    #            "digital_in": data['digital_in'][:, ::-1],
    #            "digital_out": data['digital_out'][:, ::-1],
    #            "scanner_start_ts": data['us_start'],
    #            "scanner_transmit_ts": data['us_end'],
    #            "longVar": data['states'],
    #            "packetNums": data['packetID']}

    # return decoded


def unpack_data_packet(dp, DataPacketStruct, DataPacket):
    '''
    Unpacks the data packet from b64 file stream.

    Parameters:
        dp (bytes): Data packet to unpack.
        DataPacketStruct (str): Structure of the data packet.
        DataPacket (namedtuple): Namedtuple representing the data packet structure.

    Returns:
        namedtuple: Unpacked data packet.
    '''
    s = struct.unpack(DataPacketStruct, dp)
    up = DataPacket(type=s[0], size=s[1], crc16=s[2], packetID=s[3], us_start=s[4], us_end=s[5],
                    analog=s[6:14], states=s[14:22], digitalIn=s[22], digitalOut=s[23], padding=None)
    return up


def count_lines(fp):
    '''
    Counts the number of lines in a b64 file.

    Parameters:
        fp (str): Path to the file.

    Returns:
        int: Number of lines in the file.
    '''
    def _make_gen(reader):
        b = reader(2**16)
        while b:
            yield b
            b = reader(2**16)
    with open(fp, 'rb') as f:
        count = sum(buf.count(b'\n') for buf in _make_gen(f.raw.read))
    return count


def compute_sync_shift(scanner_digital, log_ts, frame_ts):
    '''
    Computes time shift between log timestamps and video timestamps
    PARAMETERS:
    scanner_digital: log digital signal for scanner acquisition events
    start_ts: log timestamps (in us, as read in decoded log file)
    frame_ts: video timestamps (in s, as read from .tif header)

    RETURNS:
    sync_shift: log_ts - frame_ts shift, in seconds
    '''
    # extract indexes of scanner acquisition
    scanner_idxs = np.where(np.diff(scanner_digital.astype(int)) == -1)[0]
    # convert indexes to time in seconds
    scanner_ts = log_ts[scanner_idxs]/pow(10, 6)
    # get rid of spurious signals using inter-signal interval
    scanner_ts = scanner_ts[:-1][np.where(np.diff(scanner_ts) < 0.4)]
    # compute mean shift of first 1000 points
    sync_shift = np.mean(scanner_ts[:1000] - frame_ts[:1000])
    return sync_shift


def compute_sync_times(scanner_digital, log_ts, frame_ts):
    '''
    Computes the timestamps corresponding to each data point in the time-series
    extracted from the VR logs, in the scanner time frame.

    Parameters:
        scanner_digital (array-like): digital channel with scanner on signal.
        log_ts (array-like): timestamps of log channels acquisitions.
        frame_ts (array-like): timestamps of the scanner frames.

    Returns:
        array-like: Computed log times adjusted to the scanner timeframe.
    '''
    sync_shift = compute_sync_shift(
        scanner_digital=scanner_digital, log_ts=log_ts, frame_ts=frame_ts)

    # convert indexes to time in seconds and subtracts shift
    times = log_ts/pow(10, 6)-sync_shift
    return times


def invert_polarity(digital_channel):
    '''
    Inverts channel polarity.

    Parameters:
        digital_channel (array-like): Digital channel data.

    Returns:
        array-like: Digital channel data with inverted polarity.
    '''
    digital_channel = np.logical_not(digital_channel).astype(int)
    return digital_channel


def compute_onsets(digital_channel):
    '''
    Compute transitions 0->1 in digital channel.

    Parameters:
        digital_channel (array-like): Digital channel data.

    Returns:
        array-like: Indices of transitions from 0 to 1 in the digital channel.
    '''
    digital_channel = digital_channel.astype(int)
    onsets = np.where(np.diff(digital_channel) == 1)[0]+1
    return onsets


def compute_offsets(digital_channel):
    '''
    Compute transitions 1->0 in digital channel.

    Parameters:
        digital_channel (array-like): Digital channel data.

    Returns:
        array-like: Indices of transitions from 1 to 0 in the digital channel.
    '''
    digital_channel = digital_channel.astype(int)
    onsets = np.where(np.diff(digital_channel) == -1)[0]
    return onsets


def compute_switch(digital_channel):
    '''
    Compute all transitions in the digital channel.

    Parameters:
        digital_channel (array-like): Digital channel data.

    Returns:
        array-like: Indices of all transitions in the digital channel.
    '''
    digital_channel = digital_channel.astype(int)
    onsets = np.where(np.diff(digital_channel) != 0)[0]
    return onsets


def clean_switches(switch_list, time_th=500):
    '''
    Clean switch list by removing transitions that are too close.

    Parameters:
        switch_list (array-like): List of indices representing transitions.
        time_th (int, optional): Threshold to remove close transitions, in indexes. Default is 500.

    Returns:
        array-like: Cleaned switch list with transitions further apart than time_th.
    '''
    clean_switch = []
    for i in range(0, len(switch_list), 2):
        if i+1 < len(switch_list):
            if abs(switch_list[i+1]-switch_list[i]) > time_th:
                clean_switch.append(switch_list[i])
                clean_switch.append(switch_list[i+1])
        else:
            clean_switch.append(switch_list[i])
    return np.asarray(clean_switch)


def is_sound(sound_onsets, t1, t2):
    '''
    Check if sound onsets occur between t1 and t2.

    Parameters:
        sound_onsets (array-like): Onset timestamps of sound events.
        t1 (float): Start time.
        t2 (float): End time.

    Returns:
        bool: True if sound events occur between t1 and t2, False otherwise.
    '''
    return np.any(np.logical_and((t1 < sound_onsets), (t2 > sound_onsets)))


def good_reward_zones(reward_onsets, rz_onsets, rz_offsets):
    '''
    Check if there are reward presentations in the reward zone.

    Parameters:
        reward_onsets (array-like): Onset timestamps of rewards.
        rz_onsets (array-like): Onset timestamps of reward zones.
        rz_offsets (array-like): Offset timestamps of reward zones.

    Returns:
        list: List of indices of reward zones with reward presentations.
    '''
    good_idxs = []
    for i in range(len(rz_offsets)):
        if np.any(np.logical_and((rz_onsets[i] < reward_onsets), (rz_offsets[i] > reward_onsets))):
            good_idxs.append(i)
    return good_idxs


def build_trial_matrix(digital_in, digital_out):
    '''Builds trial matrix from digital channels.

       Trials are defined with reward zones: each reward zone that has a reward in it is used to definie a trial.
       - env_onset is given by the first envinronment channel switch after previous revard zone.
       - tunnel1_onset is given by last environent channel switch before the reward zone of the current trial.
       - reward_zone_onset is given by onset of reward zone in current trial
       - tunnel2_onset is given by offset of reward zone in current trial
       - tunnel2 offset is given by first env switch after current reward zone offset (equals next env_onset by definition)
       - reward_onset is given by first reward onset after reward zone onset
       - reward offset is given by first reward offset reward onset
       - sound onset is given by first sound onset after previous reward zone onset (if present)
       - sound offset is given by first sound offset after sound onset (if present)

    '''

    # builds trial matrix columns
    trial_matrix = {'env_onset': [], 'tunnel1_onset': [], 'reward_zone_onset': [],
                    'tunnel2_onset': [], 'tunnel2_offset': [], 'trial_duration': [],
                    'env_label': [], 'sound_onset': [], 'sound_offset': [],
                    'sound_presented': [], 'reward_onset': [], 'reward_offset': []}

    # columns that need to be converted in timestamps
    timestamp_keys = ['env_onset', 'tunnel1_onset', 'reward_zone_onset',
                      'tunnel2_onset', 'tunnel2_offset', 'sound_onset', 'sound_offset',
                      'sound_presented', 'reward_onset', 'reward_offset']

    # channel mapping
    channels_in = {'env1': 10, 'env2': 11, 'env3': 15, 'sound': 7,
                   'tunnel1': 13, 'tunnel2': 14, 'reward_zone': 9}
    channels_out = {'reward': 0}

    # extract digital signals
    reward_zone = digital_in[:, channels_in['reward_zone']]

    # the first env onsets is always missing, so the channel polarity has to be inverted
    env1 = invert_polarity(digital_in[:, channels_in['env1']])
    env2 = digital_in[:, channels_in['env2']]
    env3 = digital_in[:, channels_in['env3']]
    sound = digital_in[:, channels_in['sound']]
    # check id sound is witched, if True, switch it back
    if sum(sound)/len(sound) > 0.5:
        sound = invert_polarity(sound)
    reward = digital_out[:, channels_out['reward']]

    # compute environment onsets and offsets
    # adds first env1 onset at 0
    env1_switches = np.hstack([np.asarray([0.0]), compute_switch(env1)])
    env2_switches = compute_switch(env2)
    env3_switches = compute_switch(env3)

    env1_switches = clean_switches(env1_switches)
    env2_switches = clean_switches(env2_switches)
    env3_switches = clean_switches(env3_switches)

    # concatenate environments
    env_switches = np.hstack([env1_switches, env2_switches, env3_switches])

    # build env labels
    env_labels = np.hstack([np.full_like(env1_switches, 1),
                            np.full_like(env2_switches, 2),
                            np.full_like(env3_switches, 3)])

    # sort envrionments onsets, offsets and labels
    sorted_idxs = np.argsort(env_switches)
    env_switches, env_labels = env_switches[sorted_idxs], env_labels[sorted_idxs]

    # reward zone
    rz_onsets = compute_onsets(reward_zone)
    rz_offsets = compute_offsets(reward_zone)

    # reward presentation
    reward_onsets = compute_onsets(reward)
    reward_offsets = compute_offsets(reward)

    good_rz = good_reward_zones(reward_onsets, rz_onsets, rz_offsets)

    rz_onsets = rz_onsets[good_rz]
    rz_offsets = rz_offsets[good_rz]

    # sound presentation
    sound_onsets = compute_onsets(sound)
    sound_offsets = compute_offsets(sound)

    # reward presentation
    reward_onsets = compute_onsets(reward)
    reward_offsets = compute_offsets(reward)

    # first_trial
    trial_matrix['env_onset'].append(env_switches[0])
    trial_matrix['tunnel1_onset'].append(
        int(np.max(env_switches[env_switches < rz_onsets[0]])))
    trial_matrix['reward_zone_onset'].append(rz_onsets[0])
    trial_matrix['tunnel2_onset'].append(rz_offsets[0])
    trial_matrix['tunnel2_offset'].append(
        int(np.min(env_switches[env_switches > rz_offsets[0]])))

    trial_matrix['trial_duration'].append(np.nan)
    trial_matrix['env_label'].append(int(env_labels[0]))

    if is_sound(sound_onsets, env_switches[0], rz_onsets[0]):
        sound_onset = int(np.min(sound_onsets[sound_onsets > env_switches[0]]))
        sound_offset = int(np.min(sound_offsets[sound_offsets > sound_onset]))
        trial_matrix['sound_onset'].append(sound_onset)
        trial_matrix['sound_offset'].append(sound_offset)
        trial_matrix['sound_presented'].append(True)
    else:
        trial_matrix['sound_onset'].append(np.nan)
        trial_matrix['sound_offset'].append(np.nan)
        trial_matrix['sound_presented'].append(False)

    reward_onset = int(np.min(reward_onsets[reward_onsets > rz_onsets[0]]))
    reward_offset = int(np.min(reward_offsets[reward_offsets > reward_onset]))
    trial_matrix['reward_onset'].append(reward_onset)
    trial_matrix['reward_offset'].append(reward_offset)

    # loops reward zones, used for trial definition
    for i in range(1, len(rz_offsets)):
        trial_matrix['env_onset'].append(
            int(np.min(env_switches[env_switches > rz_offsets[i-1]])))
        trial_matrix['tunnel1_onset'].append(
            int(np.max(env_switches[env_switches < rz_onsets[i]])))
        trial_matrix['reward_zone_onset'].append(rz_onsets[i])
        trial_matrix['tunnel2_onset'].append(rz_offsets[i])

        # if the experiment does not end
        if len(env_switches[env_switches > rz_offsets[i]]) > 0:
            trial_matrix['tunnel2_offset'].append(
                int(np.min(env_switches[env_switches > rz_offsets[i]])))
        else:
            trial_matrix['tunnel2_offset'].append(np.nan)

        trial_matrix['trial_duration'].append(np.nan)
        trial_matrix['env_label'].append(
            int(env_labels[np.where(env_switches < rz_offsets[i])[0][-1]]))
        # trial_matrix['env_label'].append(int(env_labels[np.argmax(env_switches[env_switches<rz_onsets[i]])]))

        if is_sound(sound_onsets, rz_offsets[i-1], rz_onsets[i]):
            sound_onset = int(
                np.min(sound_onsets[sound_onsets > rz_offsets[i-1]]))
            sound_offset = int(
                np.min(sound_offsets[sound_offsets > sound_onset]))
            trial_matrix['sound_onset'].append(sound_onset)
            trial_matrix['sound_offset'].append(sound_offset)
            trial_matrix['sound_presented'].append(True)
        else:
            trial_matrix['sound_onset'].append(np.nan)
            trial_matrix['sound_offset'].append(np.nan)
            trial_matrix['sound_presented'].append(False)

        reward_onset = int(np.min(reward_onsets[reward_onsets > rz_onsets[i]]))
        reward_offset = int(
            np.min(reward_offsets[reward_offsets > reward_onset]))
        trial_matrix['reward_onset'].append(reward_onset)
        trial_matrix['reward_offset'].append(reward_offset)

    trial_matrix = pd.DataFrame(trial_matrix)

    return trial_matrix


def build_trial_matrix_old(digital_in, digital_out):
    '''Builds trial matrix from digital channels and synchorized time axis'''

    trial_matrix = {'env_onset': [], 'tunnel1_onset': [], 'reward_zone_onset': [],
                    'tunnel2_onset': [], 'tunnel2_offset': [], 'trial_duration': [],
                    'env_label': [], 'sound_onset': [], 'sound_offset': [],
                    'sound_presented': [], 'reward_onset': [], 'reward_offset': [],
                    'clean_trial': []}

    # columns that need to be converted in timestamps
    timestamp_keys = ['env_onset', 'tunnel1_onset', 'reward_zone_onset',
                      'tunnel2_onset', 'tunnel2_offset', 'sound_onset', 'sound_offset',
                      'sound_presented', 'reward_onset', 'reward_offset']

    # channel mapping
    channels_in = {'env1': 10, 'env2': 11, 'env3': 15, 'sound': 7,
                   'tunnel1': 13, 'tunnel2': 14, 'reward_zone': 9}
    channels_out = {'reward': 0}

    # extract digital signals
    reward_zone = digital_in[:, channels_in['reward_zone']]
    # the first env onsets is always missing, so the channel polarity has to be inverted
    env1 = invert_polarity(digital_in[:, channels_in['env1']])
    env2 = digital_in[:, channels_in['env2']]
    env3 = digital_in[:, channels_in['env3']]
    sound = digital_in[:, channels_in['sound']]
    reward = digital_out[:, channels_out['reward']]

    # compute environment onsets and offsets
    # adds first env1 onset at 0
    env1_onsets = np.hstack([np.asarray([0.0]), compute_onsets(env1)])
    env2_onsets = compute_onsets(env2)
    env3_onsets = compute_onsets(env3)
    env1_offsets = compute_offsets(env1)
    env2_offsets = compute_offsets(env2)
    env3_offsets = compute_offsets(env3)

    # concatenate environments
    env_onsets = np.hstack([env1_onsets, env2_onsets, env3_onsets])
    env_offsets = np.hstack([env1_offsets, env2_offsets, env3_offsets])

    # build env labels
    env_labels = np.hstack([np.full_like(env1_onsets, 1),
                            np.full_like(env2_onsets, 2),
                            np.full_like(env3_onsets, 3)])

    # sort envrionments onsets, offsets and labels
    sorted_idxs = np.argsort(env_onsets)
    env_onsets, env_labels = env_onsets[sorted_idxs], env_labels[sorted_idxs]
    sorted_idxs = np.argsort(env_offsets)
    env_offsets = env_offsets[sorted_idxs]

    # reward zone
    rz_onsets = compute_onsets(reward_zone)
    rz_offsets = compute_offsets(reward_zone)
    # sound presentation
    sound_onsets = compute_onsets(sound)
    sound_offsets = compute_offsets(sound)
    # reward presentation
    reward_onsets = compute_onsets(reward)
    reward_offsets = compute_offsets(reward)

    trial_starts = []
    corrected_trial = []
    # handles first trial
    trial_starts.append(int(env1_onsets[0]))
    trial_matrix['env_label'].append(int(env_labels[0]))
    trial_matrix['tunnel1_onset'].append(int(env_offsets[0]))
    # loops over env onsets
    for i in range(1, len(env_onsets)-1):
        if np.any(np.logical_and(rz_onsets > env_onsets[i-1], rz_onsets < env_onsets[i])):
            # if rz is present, appends trial starts
            trial_starts.append(int(env_onsets[i]))
            trial_matrix['env_label'].append(int(env_labels[i]))
        else:
            # if rz is not present, trial start is not appended,
            # i is stored in corrected trial indexes
            corrected_trial.append(i)

        if np.any(np.logical_and(rz_onsets > env_onsets[i], rz_onsets < env_onsets[i+1])):
            trial_matrix['tunnel1_onset'].append(int(env_offsets[i]))

        else:
            # if rz is not present, trial start is not appended,
            # i is stored in corrected trial indexes
            corrected_trial.append(i)

    # handles last trial: if reward zone was reached it is considered
    if np.any(rz_onsets > env_onsets[-1]):
        trial_starts.append(int(env_onsets[-1]))
        trial_matrix['env_label'].append(int(env_labels[-1]))
        trial_matrix['tunnel1_onset'].append(int(env_offsets[-1]))

    # marks as corrected all trial that come AFTER any number of rejected trials
    # this allows to handle multiple jitters at the env start
    corrected_trial = np.unique(corrected_trial)
    trial_matrix['clean_trial'] += [True for i in trial_starts]
    corrected_idxs = [
        i+1 for i in corrected_trial if not((i+1) in corrected_trial)]
    for i in corrected_idxs:
        trial_matrix['clean_trial'][i] = False

    # stores correct env onsets
    trial_matrix['env_onset'] += trial_starts

    # reward_zone
    # loops over trial starts
    for i in range(len(trial_starts)-1):
        # takes all rz onsets between two consecutive trial starts
        in_trial_idx = np.where(np.logical_and(
            rz_onsets > trial_starts[i], rz_onsets < trial_starts[i+1]))[0]
        # stores the first onset
        rz_on = min(rz_onsets[in_trial_idx])
        # takes all rz offsets between two consecutive trial starts
        in_trial_idx = np.where(np.logical_and(
            rz_offsets > trial_starts[i], rz_offsets < trial_starts[i+1]))[0]
        # stores the last offset
        rz_off = max(rz_offsets[in_trial_idx])
        # append in trial matrix
        trial_matrix['reward_zone_onset'].append(int(rz_on))
        trial_matrix['tunnel2_onset'].append(int(rz_off))
        # if more than one reward zone, sign trial as not clean
        if len(in_trial_idx) > 1:
            trial_matrix['clean_trial'][i] = False

    # handles last trial
    if len(np.where(rz_onsets > trial_starts[-1])[0]) > 0:
        rz_on = min(rz_onsets[np.where(rz_onsets > trial_starts[-1])[0]])
        trial_matrix['reward_zone_onset'].append(int(rz_on))
        if len(np.where(rz_onsets > trial_starts[-1])[0]) > 1:
            trial_matrix['clean_trial'][i] = False
    else:
        trial_matrix['reward_zone_onset'].append(np.nan)

    if len(np.where(rz_offsets > trial_starts[-1])[0]) > 0:
        rz_off = max(rz_offsets[np.where(rz_offsets > trial_starts[-1])[0]])
        trial_matrix['tunnel2_onset'].append(int(rz_off))
    else:
        trial_matrix['tunnel2_onset'].append(np.nan)

    # saves beginning of next trial as tunnel2 offset for convenience
    trial_matrix['tunnel2_offset'] += trial_matrix['env_onset'][1:]
    # no endpoint to last trial
    trial_matrix['tunnel2_offset'].append(np.nan)

    # saves sound onsets and offsets, and sound presented flag
    for i in range(len(trial_starts)-1):
        in_trial_idx = np.where(np.logical_and(
            sound_onsets > trial_starts[i], sound_offsets < trial_starts[i+1]))[0]
        if len(in_trial_idx) > 0:
            trial_matrix['sound_onset'].append(
                int(sound_onsets[in_trial_idx][0]))
            trial_matrix['sound_offset'].append(
                int(sound_offsets[in_trial_idx][0]))
            trial_matrix['sound_presented'].append(True)

        else:
            trial_matrix['sound_onset'].append(np.nan)
            trial_matrix['sound_offset'].append(np.nan)
            trial_matrix['sound_presented'].append(False)

    if len(np.where(sound_onsets > trial_starts[-1])[0]) > 0:
        trial_matrix['sound_onset'].append(
            int(sound_onsets[np.where(sound_onsets > trial_starts[-1])[0][0]]))
        trial_matrix['sound_offset'].append(
            int(sound_offsets[np.where(sound_offsets > trial_starts[-1])[0][0]]))
        trial_matrix['sound_presented'].append(True)
    else:
        trial_matrix['sound_onset'].append(np.nan)
        trial_matrix['sound_offset'].append(np.nan)
        trial_matrix['sound_presented'].append(False)

    # saves sound onsets and offsets
    trial_matrix['reward_onset'] = []
    trial_matrix['reward_offset'] = []
    for i in range(len(trial_starts)-1):
        in_trial_idx = np.where(np.logical_and(
            reward_onsets > trial_starts[i], reward_offsets < trial_starts[i+1]))[0]
        if len(in_trial_idx) > 0:
            trial_matrix['reward_onset'].append(
                int(reward_onsets[in_trial_idx][0]))
            trial_matrix['reward_offset'].append(
                int(reward_offsets[in_trial_idx][0]))

        else:
            trial_matrix['reward_onset'].append(np.nan)
            trial_matrix['reward_offset'].append(np.nan)

    if len(np.where(reward_onsets > trial_starts[-1])[0]) > 0:
        trial_matrix['reward_onset'].append(
            int(reward_onsets[np.where(reward_onsets > trial_starts[-1])[0][0]]))
        trial_matrix['reward_offset'].append(
            int(reward_offsets[np.where(reward_offsets > trial_starts[-1])[0][0]]))
    else:
        trial_matrix['reward_onset'].append(np.nan)
        trial_matrix['reward_offset'].append(np.nan)

    # convert in times:
    # for k in timestamp_keys:
    #    for i in range(len(trial_matrix[k])):
    #        if not np.isnan(trial_matrix[k][i]):
    #            trial_matrix[k][i] = sync_times[int(trial_matrix[k][i])]

    # computes and store trial duration
    trial_matrix['trial_duration'] += list(np.asarray(
        trial_matrix['tunnel2_offset'])-np.asarray(trial_matrix['env_onset']))

    trial_matrix = pd.DataFrame(trial_matrix)

    return trial_matrix


# preprocess session function

def preprocess_vr_data(tif_file=None, log_file=None):
    '''
    Preprocesses the data in the given TIF and log files and returns a tuple
    containing the data structure needed for further analysis.

    Parameters:
        tif_file (str): Path to the TIF file to read.
        log_file (str): Path to the log file to read.

    Returns:
        dict: A dictionary containing the preprocessed VR data with the following keys:
            - 'decoded_log': Decoded log data.
            - 'tif_header': Timestamps from the TIF header.
            - 'trial_matrix': Trial matrix as a pandas DataFrame.
            - 'behavioural_data': Dictionary containing behavioral data with the following keys:
                - 'time': Synchronized timestamps.
                - 'position': Position data as integers.
                - 'lick_onsets': Lick onsets indices in the digital_out data.
                - 'lick_offsets': Lick offsets indices in the digital_out data.
                - 'reward_onsets': Reward onsets indices in the digital_out data.
                - 'reward_offsets': Reward offsets indices in the digital_out data.

    Note:
        The function reads the log file and TIF header to extract relevant data. It performs type
        conversion and computes scanner-synchronized timestamps based on the digital scanner signal. 
        It then builds a trial matrix from the digital_in and digital_out data. 
        The function also extracts behavioral data such as position, lick onsets, lick offsets, reward onsets, and reward offsets.
    '''

    # reads log file and tif header
    decoded_log = create_bp_structure(log_file)
    tif_header = read_tif_header(tif_file)
    frames = tif_header['frame_ts']

    # type conversion
    digital_in = decoded_log['digitalIn'].astype(int)
    digital_out = decoded_log['digitalOut'].astype(int)
    digital_scan_signal = digital_in[:, 6]
    log_times = decoded_log['startTS']

    # compute synchronized timestamps
    sync_times = compute_sync_times(digital_scan_signal, log_times, frames)

    # build trial matrix
    tm = build_trial_matrix(digital_in, digital_out)

    # extract behavioural data
    position = decoded_log['longVar'][:, 1].astype(int)
    lick_onsets = compute_onsets(digital_out[:, -2])
    lick_offsets = compute_offsets(digital_out[:, -2])
    reward_onsets = compute_onsets(digital_out[:, 0])
    reward_offsets = compute_offsets(digital_out[:, 0])

    data = {'time': sync_times, 'position': position.astype('int'),
            'lick_onsets': lick_onsets, 'lick_offsets': lick_offsets,
            'reward_onsets': reward_onsets, 'reward_offsets': reward_offsets}

    vr_data = {'decoded_log': decoded_log, 'tif_header': tif_header,
               'trial_matrix': tm, 'behavioural_data': data}

    return vr_data


def save_processed_vr_data(output_path, vr_data):
    '''
    Saves processed virtual reality (VR) data to the specified output directory.

    This function takes the processed VR data, which includes the decoded log, tif header,
    trial matrix, and behavioral data, and saves each component into separate files.

    Parameters:
        output_path (str or Path): The path to the output directory where the data will be saved.
        vr_data (dict): A dictionary containing the processed VR data with the following keys:
            - 'decoded_log': Decoded log data.
            - 'tif_header': TIF header data (frame timestamps).
            - 'trial_matrix': Trial matrix data.
            - 'behavioural_data': Preprocessed behavioral data.

    Returns:
        None

    Side Effects:
        - Saves 'decoded_log.mat': Decoded log data in MATLAB .mat format.
        - Saves 'tif_header.pickle': TIF header data (frame timestamps) in a Python pickle file.
        - Saves 'behaviour_data.pickle': Preprocessed behavioral data in a Python pickle file.
        - Saves 'trial_data.csv': Trial matrix data in a CSV file format.
    '''

    decoded_log = vr_data['decoded_log']
    tif_header = vr_data['tif_header']
    tm = vr_data['trial_matrix']
    data = vr_data['behavioural_data']

    # save decoded log
    savemat(output_path.joinpath('decoded_log.mat'), decoded_log)
    # save frame timestamps
    with open(output_path.joinpath('tif_header.pickle'), 'wb') as file:
        pickle.dump(tif_header, file, protocol=pickle.HIGHEST_PROTOCOL)

    # save preprocessed behaviour
    filename = output_path.joinpath('behaviour_data.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # save trial matrix
    filename = output_path.joinpath('trial_data.csv')
    tm.to_csv(filename, index=False)

    return
