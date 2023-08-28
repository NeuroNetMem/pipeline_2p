'''
This module contains a tweaked implementation of CaImAn's save_memmap functions. It manually sets the 'in_memory' argument in the cm.load() function that reads the file (line 140) to 'False', in order to let the function handle big files. The flag is currently not exposed to the user, a better fix would be rewriting the caiman function to allow for that.
'''


from past.builtins import basestring
from past.utils import old_div

import ipyparallel as parallel
from itertools import chain
import logging
import numpy as np
import os
import pickle
import sys
import tifffile
from typing import Any, Dict, List, Optional, Tuple, Union
import pathlib

import caiman as cm
import caiman.paths



def save_memmap(filenames: List[str],
                base_name: str = 'Yr',
                resize_fact: Tuple = (1, 1, 1),
                remove_init: int = 0,
                idx_xy: Tuple = None,
                order: str = 'F',
                var_name_hdf5: str = 'mov',
                xy_shifts: Optional[List] = None,
                is_3D: bool = False,
                add_to_movie: float = 0,
                border_to_0=0,
                dview=None,
                n_chunks: int = 100,
                slices=None) -> str:
    """ Efficiently write data from a list of tif files into a memory mappable file
    Args:
        filenames: list
            list of tif files or list of numpy arrays
        base_name: str
            the base used to build the file name. WARNING: Names containing underscores may collide with internal semantics.
        resize_fact: tuple
            x,y, and z downsampling factors (0.5 means downsampled by a factor 2)
        remove_init: int
            number of frames to remove at the begining of each tif file
            (used for resonant scanning images if laser in rutned on trial by trial)
        idx_xy: tuple size 2 [or 3 for 3D data]
            for selecting slices of the original FOV, for instance
            idx_xy = (slice(150,350,None), slice(150,350,None))
        order: string
            whether to save the file in 'C' or 'F' order
        xy_shifts: list
            x and y shifts computed by a motion correction algorithm to be applied before memory mapping
        is_3D: boolean
            whether it is 3D data
        add_to_movie: floating-point
            value to add to each image point, typically to keep negative values out.
        border_to_0: (undocumented)
        dview:       (undocumented)
        n_chunks:    (undocumented)
        slices: slice object or list of slice objects
            slice can be used to select portion of the movies in time and x,y
            directions. For instance
            slices = [slice(0,200),slice(0,100),slice(0,100)] will take
            the first 200 frames and the 100 pixels along x and y dimensions.
    Returns:
        fname_new: the name of the mapped file, the format is such that
            the name will contain the frame dimensions and the number of frames
    """
    if not isinstance(filenames, list):
        raise Exception('save_memmap: input should be a list of filenames')

    if slices is not None:
        slices = [slice(0, None) if sl is None else sl for sl in slices]

    if len(filenames) > 1:
        recompute_each_memmap = False
        for file__ in filenames:
            if ('order_' + order not in file__) or ('.mmap' not in file__):
                recompute_each_memmap = True


        if recompute_each_memmap or (remove_init>0) or (idx_xy is not None)\
                or (xy_shifts is not None) or (add_to_movie != 0) or (border_to_0>0)\
                or slices is not None:

            logging.debug('Distributing memory map over many files')
            # Here we make a bunch of memmap files in the right order. Same parameters
            fname_parts = cm.save_memmap_each(filenames,
                                              base_name=base_name,
                                              order=order,
                                              border_to_0=border_to_0,
                                              dview=dview,
                                              var_name_hdf5=var_name_hdf5,
                                              resize_fact=resize_fact,
                                              remove_init=remove_init,
                                              idx_xy=idx_xy,
                                              xy_shifts=xy_shifts,
                                              is_3D=is_3D,
                                              slices=slices,
                                              add_to_movie=add_to_movie)
        else:
            fname_parts = filenames

        # The goal is to make a single large memmap file, which we do here
        if order == 'F':
            raise Exception('You cannot merge files in F order, they must be in C order')

        fname_new = cm.save_memmap_join(fname_parts, base_name=base_name,
                                        dview=dview, n_chunks=n_chunks)

    else:
        # TODO: can be done online
        Ttot = 0
        for idx, f in enumerate(filenames):
            if isinstance(f, str):     # Might not always be filenames.
                logging.debug(f)

            if is_3D:
                Yr = f if not (isinstance(f, basestring)) else tifffile.imread(f)
                if Yr.ndim == 3:
                    Yr = Yr[None, ...]
                if slices is not None:
                    Yr = Yr[tuple(slices)]
                else:
                    if idx_xy is None:         #todo remove if not used, superceded by the slices parameter
                        Yr = Yr[remove_init:]
                    elif len(idx_xy) == 2:     #todo remove if not used, superceded by the slices parameter
                        Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
                    else:                      #todo remove if not used, superceded by the slices parameter
                        Yr = Yr[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

            else:
                if isinstance(f, (basestring, list)):
                    # the in-memory option is changed with respect to caiman base function, to allow handling
                    # of large files
                    Yr = cm.load(caiman.paths.fn_relocated(f), fr=1, in_memory=False, var_name_hdf5=var_name_hdf5)
                else:
                    Yr = cm.movie(f)
                if xy_shifts is not None:
                    Yr = Yr.apply_shifts(xy_shifts, interpolation='cubic', remove_blanks=False)

                if slices is not None:
                    Yr = Yr[tuple(slices)]
                else:
                    if idx_xy is None:
                        if remove_init > 0:
                            Yr = Yr[remove_init:]
                    elif len(idx_xy) == 2:
                        Yr = Yr[remove_init:, idx_xy[0], idx_xy[1]]
                    else:
                        raise Exception('You need to set is_3D=True for 3D data)')
                        Yr = np.array(Yr)[remove_init:, idx_xy[0], idx_xy[1], idx_xy[2]]

            if border_to_0 > 0:
                if slices is not None:
                    if isinstance(slices, list):
                        raise Exception(
                            'You cannot slice in x and y and then use add_to_movie: if you only want to slice in time do not pass in a list but just a slice object'
                        )

                min_mov = Yr.calc_min()
                Yr[:, :border_to_0, :] = min_mov
                Yr[:, :, :border_to_0] = min_mov
                Yr[:, :, -border_to_0:] = min_mov
                Yr[:, -border_to_0:, :] = min_mov

            fx, fy, fz = resize_fact
            if fx != 1 or fy != 1 or fz != 1:
                if 'movie' not in str(type(Yr)):
                    Yr = cm.movie(Yr, fr=1)
                Yr = Yr.resize(fx=fx, fy=fy, fz=fz)

            T, dims = Yr.shape[0], Yr.shape[1:]
            Yr = np.transpose(Yr, list(range(1, len(dims) + 1)) + [0])
            Yr = np.reshape(Yr, (np.prod(dims), T), order='F')
            Yr = np.ascontiguousarray(Yr, dtype=np.float32) + np.float32(0.0001) + np.float32(add_to_movie)

            if idx == 0:
                fname_tot = cm.paths.generate_fname_tot(base_name, dims, order)
                if isinstance(f, str):
                    fname_tot = caiman.paths.fn_relocated(os.path.join(os.path.split(f)[0], fname_tot))
                if len(filenames) > 1:
                    big_mov = np.memmap(caiman.paths.fn_relocated(fname_tot),
                                        mode='w+',
                                        dtype=np.float32,
                                        shape=prepare_shape((np.prod(dims), T)),
                                        order=order)
                    big_mov[:, Ttot:Ttot + T] = Yr
                    del big_mov
                else:
                    logging.debug('SAVING WITH numpy.tofile()')
                    Yr.tofile(fname_tot)
            else:
                big_mov = np.memmap(fname_tot,
                                    dtype=np.float32,
                                    mode='r+',
                                    shape=prepare_shape((np.prod(dims), Ttot + T)),
                                    order=order)

                big_mov[:, Ttot:Ttot + T] = Yr
                del big_mov

            sys.stdout.flush()
            Ttot = Ttot + T

        fname_new = caiman.paths.fn_relocated(fname_tot + f'_frames_{Ttot}.mmap')
        try:
            # need to explicitly remove destination on windows
            os.unlink(fname_new)
        except OSError:
            pass
        os.rename(fname_tot, fname_new)

    return fname_new