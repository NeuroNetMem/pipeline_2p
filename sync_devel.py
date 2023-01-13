from logreader import logreader
from os import path


tif_file = '/ceph/imaging1/arie/429420_toms/20221026_429420/20221026_429420_00001.tif'
log_file = '/ceph/imaging1/arie/429420_toms/20221026_429420/20221026-200818_249.b64'

log = logreader.create_bp_structure(log_file)

for k in log.keys():
    print(f'{k}: {log[k].shape}')

if path.exists(tif_file):
    print('reading .tif')
    I2C, frameTs = logreader.read_ScanImageTiffHeader(tif_file)
else:
    print('file does not exist')

print(f"Transmit frameTs: {frameTs}")
print(f"Transmit Ts: {log['transmitTS'][:30]}")
#print(frameTs)






