import os

import numpy as np

import fklab.io.neuralynx as nlx
import fklab.io.mwl as mwl
from fklab.utilities.general import blocks


def convert_to_pxyabw( filename ):
    """Save spike amplitudes from ntt file to pxyabw file.
    
    The output file is saved in the same directory and with the same base name
    as the input file.
    
    Parameters
    ----------
    filename: str
        path to Neuralynx ntt file
    
    
    """
    
    invert = 1.
    
    try:
        fid = nlx.NlxOpen( filename )
        fid.correct_inversion = False
        if not fid.header['InputInverted']:
            invert = -1. #we want to invert, so we can use max to find spike peaks
    except nlx.NeuralynxIOError:
        fid = mwl.MwlOpen( filename )
    
    nrec = fid.nrecords
    
    dtype = [('id',np.int32),('time',np.float64)]
    for k in range(fid.nchannels):
        dtype.append( ('peak'+str(k),np.float32) )
    
    dtype = np.dtype(dtype)
    
    root, name = os.path.split( fid.fullpath )
    name, ext = os.path.splitext( name )
    
    outfile = mwl.MwlFileFeature.create( os.path.join(root,name), dtype=dtype)
    
    blocksize=1000
    data = np.zeros( blocksize, dtype=dtype )
    
    for start,n in blocks( nitems=nrec, blocksize=blocksize ):
        t = fid.data.time[start:(start+n)]
        w = fid.data.waveform[start:(start+n)]
        
        peak = np.amax(invert*w, axis=1)
        
        data['id'][0:n] = np.arange( start, start+n )
        data['time'][0:n] = t
        for k in range(fid.nchannels):
            data['peak'+str(k)][0:n] = peak[:,k]
        
        outfile.append_data( data[0:n] )


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Extract spike waveform features from Neuralynx ntt file and save as pxyabw file for use with xclust.')
    parser.add_argument('infile')
    args = parser.parse_args()
    
    convert_to_pxyabw( args.infile )
    
    
    
