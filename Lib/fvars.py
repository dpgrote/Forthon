# Created by David P. Grote, March 6, 1998
# $Id: fvars.py,v 1.2 2004/05/11 00:50:45 dave Exp $

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

class Fargs:
  name = ''
  type = ''

class Fdims:
  low = ''
  high = ''

class Ftype:
  def __init__(s,name,attr):
    s.name = name
    s.attr = attr
    s.vlist = []
  def addvar(s,v):
    s.vlist.append(v)
  def display(s):
    print 'name = ',s.name
    print 'variables:'
    for v in s.vlist:
      print '  ',v.name,'  ',v.type

ftoc_dict = {'integer':'long', 'real':'double', 'logical':'long',
             'character':'FSTRING', 'string':'FSTRING',
             'void':'void','Filedes':'long','complex':'Py_complex'}
ftop_dict = {'integer':'LONG', 'real':'DOUBLE', 'logical':'LONG',
             'character':'CHAR','string':'CHAR','void':'VOID',
             'Filedes':'LONG','complex':'CDOUBLE'}
fto1 = {'integer':'i',   'real':'d',      'logical':'i',   'character':'s',
        'string':'s','Filedes':'i','complex':'D'}
ftof_dict = {'integer':'integer('+cfinterface.isz+')',
             'real':'real(kind=8)',
             'logical':'logical('+cfinterface.isz+')',
             'character':'character',
             'string':'character',
             'void':'void',
             'Filedes':'integer('+cfinterface.isz+')',
             'complex':'complex(kind=8)'}

def isderivedtype(arg):
  if arg.type in ftoc_dict.keys(): return 0
  else: return 1

def ftoc(type):
  if type in ftoc_dict.keys(): return ftoc_dict[type]
  else: return 'char'

def ftop(type):
  if type in ftop_dict.keys(): return ftop_dict[type]
  else: return 'OBJECT'

def ftof(type):
  if type in ftof_dict.keys(): return ftof_dict[type]
  else: return 'TYPE(%s)'%type

