import numpy as np

# Define functions
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

def write_recarray_to_file(fname, data, header=True, sep=' ', quote=False):
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
    fmt = sep.join( convert_dtype_to_fmt(data.dtype, quote) ) + '\n'
    
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

