import tarfile
import tempfile
import os, os.path

import numpy as np

def convert_dtype_to_fmt(dtype, quote=True):
    '''
    Converts dtype from record array to output format
    Uses %d for integers, %g for floats, and %s for strings
    '''
    # Get kinds
    kinds = [dtype.fields[key][0].kind for key in dtype.names]
    
    # Iterate through kinds, assigning format as needed
    fmt = []
    for i in range(len(kinds)):
        if kinds[i] in ('b','i','u'):
            fmt.append('%d')
        elif kinds[i] in ('c', 'f'):
            fmt.append('%g')
        elif kinds[i] in ('S',):
            if quote:
                fmt.append('"%s"')
            else:
                fmt.append('%s')
        else:
            fmt.append('%s')
    
    return fmt

def write_recarray_to_file(fname, data, header=True, sep=' ', quote=False,
                           fmt=None):
    '''
    Write numpy record array to file as delimited text.
    
    fname can be either a file name or a file object.
    
    Works only for numeric data in current form; it will not format strings
    correctly.
    '''
    # Get field names
    fieldnames = data.dtype.names
    
    # Build header
    if header: header_str = sep.join(fieldnames) + '\n'
    
    # Build format string for numeric data
    if fmt is None:
        fmt = sep.join( convert_dtype_to_fmt(data.dtype, quote) ) + '\n'
    else:
        fmt = sep.join(fmt) + '\n'
    
    # Setup output file object
    if type(fname) is file:
        out_file = fname
    else:
        out_file = open(fname, "wb")
    
    # Write output
    if header: out_file.write(header_str)
    
    for rec in data:
        out_file.write(fmt % rec.tolist())
    
    # Close output file
    out_file.close()

def write_arrays_to_tarball(fname, compress='bz2', scratch=None, **kwargs):
    '''
    Write arrays (from **kwargs) to tarball at fname with given compression.

    Works be first writing each array to an npy file in a scratch directory,
    then archiving these files to a (compressed) tarball.

    This handles arrays of sizes that np.savez and np.save with non-file objects
    cannot handle. It relies upon using pure file objects for np.save that force
    use of the np.ndarray.tofile function, whcih intelligently chunks binary
    output to avoid integer overflows.
    
    Parameters
    ----------
    - fname : string
        Path for output archive
    - compress : string
        Compression to use. Can be 'bz2', 'gz', or ''.
    - scratch : string
        Scratch directory for intermediate npy files. Created via
        tempfile.mkdtemp with default arguments if scratch is None.
    - **kwargs
        Arrays to archive.

    Returns
    -------
    None
    '''
    # Make scratch directory if needed
    if scratch is None:
        scratch = tempfile.mkdtemp()
    if not os.path.exists(scratch):
        os.makedirs(scratch)

    # Save each array to its own npy file and archive npy files into compressed
    # tarball

    # Setup tarfile for writing
    archive = tarfile.open(name=fname, mode='w:' + compress)

    for name, array in kwargs.iteritems():
        path_npy = os.path.join(scratch, name + '.npy')
        np.save(path_npy, array)
        archive.add(path_npy, arcname=name + '.npy')
        os.remove(path_npy)

    archive.close()


