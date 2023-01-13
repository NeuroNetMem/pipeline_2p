import numpy as np
import base64
import struct
from cobs import cobs
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.signal
from scipy.io import savemat
import rawpy
import os
import pickle
from collections import namedtuple


from ScanImageTiffReader import ScanImageTiffReader


def read_ScanImageTiffHeader(file_path):
    '''
    Reads header description information of ScanImageTiff File
    input: (str) tiff file path
    output: (list) Dictionary and list containing I2C events information and frame timestamps 

    '''
    frameTs = []
    i2c_timestamp = []
    i2c_values = []
    i2c_frameN = []

    with ScanImageTiffReader(file_path) as reader:
        time = reader.shape()[0]
        for frame in range(30):
            x = reader.description(frame)
            #print(x)
            description = x.split('\n')
            frameTs.append(float(description[3].split('=')[1]))
            i2c = description[14].split('=')
            if len(i2c[1]) > 3:
                y = i2c[1].split('{{')[1].split(']}')
                if len(y) == 1:
                    i2c_timestamp.append(float(y[events].split(',')[0]))
                    i2c_values.append(int(y[events].split('[')[1]))
                    i2c_frameN.append(frame)
                else:
                    i2c_timestamp.append(float(y[0].split(',')[0]))
                    i2c_values.append(int(y[0].split('[')[1].split(',')[0]))
                    i2c_frameN.append(frame)
                    for events in range(1, len(y)-1):
                        i2c_timestamp.append(
                            float(y[events].split(',')[0].split('{')[1]))
                        i2c_values.append(
                            int(y[events].split('[')[1].split(',')[0]))
                        i2c_frameN.append(frame)

    I2C = {"ts": i2c_timestamp, "val": i2c_values, "frameNum": i2c_frameN}
    return I2C, frameTs


def unpack_data_packet(dp, DataPacketStruct, DataPacket):
    s = struct.unpack(DataPacketStruct, dp)
    up = DataPacket(type=s[0], size=s[1], crc16=s[2], packetID=s[3], us_start=s[4], us_end=s[5],
                    analog=s[6:14], states=s[14:22], digitalIn=s[22], digitalOut=s[23], padding=None)
    return up


def count_lines(fp):
    # function to count the packet number
    def _make_gen(reader):
        b = reader(2**16)
        while b:
            yield b
            b = reader(2**16)
    with open(fp, 'rb') as f:
        count = sum(buf.count(b'\n') for buf in _make_gen(f.raw.read))
    return count


def create_bp_structure(bp):
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
    decoded = {"analog": data['analog'], "digitalIn": data['digital_in'], "digitalOut": data['digital_out'],
               "startTS": data['us_start'], "transmitTS": data['us_end'], "longVar": data['states'], "packetNums": data['packetID']}

    return decoded


def find_sync_shift():
    pass