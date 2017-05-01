# Created by David P. Grote, March 6, 1998

import cfinterface

# Declare a class to hold fortran variable info
class Fvars:
    name = ''
    type = ''
    dims = []
    args = []
    dynamic = 0
    data = ''
    unit = ''
    comment = ''
    group = ''
    attr = ''
    limit = ''
    dimstring = ''
    array = 0
    function = 0
    derivedtype = 0
    parameter = 0
    setaction = None
    getaction = None

class Fargs:
    name = ''
    type = ''
    dimstring = ''
    dims = []

class Fdims:
    low = ''
    high = ''

class Ftype:
    def __init__(s, name, attr):
        s.name = name
        s.attr = attr
        s.vlist = []
    def addvar(s, v):
        s.vlist.append(v)
    def display(s):
        print 'name = ', s.name
        print 'variables:'
        for v in s.vlist:
            print '  ', v.name, '  ', v.type

ftoc_dict = {'integer':'long', 'logical':'long',
             'real':{'8':'double', '4':'float'}[cfinterface.realsize],
             'double':'double', 'float':'float',
             'character':'FSTRING', 'string':'FSTRING',
             'void':'void', 'Filedes':'long', 'complex':'Py_complex'}
ftop_dict = {'integer':'LONG', 'logical':'LONG',
             'real':{'8':'DOUBLE', '4':'FLOAT'}[cfinterface.realsize],
             'double':'DOUBLE', 'float':'FLOAT',
             'character':'STRING', 'string':'STRING', 'void':'VOID',
             'Filedes':'LONG', 'complex':'CDOUBLE'}
fto1 = {'integer':'l', 'logical':'l',
        'real':{'8':'d', '4':'f'}[cfinterface.realsize],
        'double':'d', 'float':'f',
        'character':'s',
        'string':'s', 'Filedes':'l', 'complex':'D'}
ftof_dict = {'integer':'integer('+cfinterface.isz+')',
             'real':'real(kind=%s)'%cfinterface.realsize,
             'double':'real(kind=8)',
             'float':'real(kind=4)',
             'logical':'logical('+cfinterface.isz+')',
             'character':'character',
             'string':'character',
             'void':'void',
             'Filedes':'integer('+cfinterface.isz+')',
             'complex':'complex(kind=8)'}

def isderivedtype(arg):
    if arg.type in ftoc_dict: return 0
    else: return 1

def ftoc(type):
    if type in ftoc_dict: return ftoc_dict[type]
    else: return 'char'

def ftop(type):
    # --- Returns the numpy type associated with the fortran type.
    try:
        result = ftop_dict[type]
    except KeyError:
        result = 'OBJECT'
    return result

def ftof(type):
    if type in ftof_dict: return ftof_dict[type]
    else: return 'TYPE(%s)'%type

