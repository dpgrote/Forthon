# Created by David P. Grote, March 6, 1998
# $Id: cfinterface.py,v 1.3 2005/04/02 00:12:36 dave Exp $

# Routines which allows c functions to be callable by fortran
import sys
import getopt
import string
import re
import struct

# Set default values of inputs
machine = sys.platform
f90 = 1
f90f = 0
twounderscores = 0 # When true, names with underscores in them have an extra
                 # underscore appedend to the fortran name

# Get system name from the command line
try:
  optlist,args = getopt.getopt(sys.argv[1:],'ad:t:F:',
                     ['f90','f90f','2underscores','nowritemodules','macros='])
  for o in optlist:
    if o[0] == '-t':
      machine = o[1]
    elif o[0] == '--f90':
      f90 = 1
    elif o[0] == '--f90f':
      f90f = 1
    elif o[0] == '--2underscores':
      twounderscores = 1
except (getopt.error,IndexError):
  pass

#----------------------------------------------------------------------------
# Set size of fortran integers and logicals. This is almost alway 4.
isz = 'kind=4'
if machine in ['AXP','T3E','J90']:
  isz = 'kind=8'
if struct.calcsize('l') == 8:
  isz = 'kind=8'

#----------------------------------------------------------------------------
# Creates a function which converts a C name into a Fortran name

if machine in ['hp-uxB','aix4','win32','MAC']:
  def fname(n):
    return string.lower(n)
elif machine in ['linux2','darwin','SOL','sunos5','AXP','osf1V4','DOS']:
  if not twounderscores:
    def fname(n):
      return string.lower(n+'_')
  else:
    def fname(n):
      m = re.search('_',n)
      if m == None: return string.lower(n+'_')
      else:         return string.lower(n+'__')
elif machine in ['T3E','sn67112','C90','J90','SGI','irix646']:
  def fname(n):
    return string.upper(n)
else:
  raise 'Machine %s not supported'%machine

#----------------------------------------------------------------------------
# Creates a function which returns the Fortran name of an object

if not f90:
  def fnameofobj(f):
    return fname(f.name)
else:
  def fnameofobj(f):
    return fname(f.name)
# if machine in ['hp-uxB','linux2', \
#                'SOL','sunos5','AXP','osf1V4','DOS','MAC']:
#   def fnameofobj(f):
#     return f.name+'_in_'+f.group
# elif machine in ['T3E','sn67112','C90','J90']:
#   def fnameofobj(f):
#     return string.upper(f.name)+'_in_'+string.upper(f.group)
# elif machine in ['SGI','irix646']:
#   def fnameofobj(f):
#     return string.upper(f.name)+'.in.'+string.upper(f.group)
# elif machine in ['aix4']:
#   def fnameofobj(f):
#     return '__'+f.group+'_MOD_'+f.name
# else:
#   raise 'Machine %s not supported'%machine

#----------------------------------------------------------------------------
# Sets up C macros which are used to take the place of the length
# of a string passed from Fortran to C.

if machine in ['hp-uxB','linux2','darwin','SOL','sunos5','DOS','aix4','win32']:
  charlen_at_end = 1
  forthonf2c = """
#define FSTRING char*
#define SL1 ,int sl1
#define SL2 ,int sl2
#define SL3 ,int sl3
#define SL4 ,int sl4
#define SL5 ,int sl5
#define FSTRLEN1(S) sl1
#define FSTRLEN2(S) sl2
#define FSTRLEN3(S) sl3
#define FSTRLEN4(S) sl4
#define FSTRLEN5(S) sl5
#define FSTRPTR(S) S
#define FSETSTRING(S,P,L) S = P
#define FSETSTRLEN1(S,L) sl1 = L
#define FSETSTRLEN2(S,L) sl2 = L
#define FSETSTRLEN3(S,L) sl3 = L
#define FSETSTRLEN4(S,L) sl4 = L
#define FSETSTRLEN5(S,L) sl5 = L
"""

elif machine in ['T3E','sn67112','SGI','irix646']:
  charlen_at_end = 0
  forthonf2c = """
typedef struct {char* ptr;int len;} fstring;
#define FSTRING fstring
#define SL1 
#define SL2 
#define SL3 
#define SL4 
#define SL5 
#define FSTRLEN1(S) S.len
#define FSTRLEN2(S) S.len
#define FSTRLEN3(S) S.len
#define FSTRLEN4(S) S.len
#define FSTRLEN5(S) S.len
#define FSTRPTR(S) S.ptr
#define FSETSTRING(S,P,L) {S.ptr = P;S.len = L;}
"""

elif machine in ['C90','J90']:
  charlen_at_end = 0
  forthonf2c = """
#include <fortran.h>
#define FSTRING _fcd
#define SL1 
#define SL2 
#define SL3 
#define SL4 
#define SL5 
#define FSTRLEN1(S) _fcdlen(S)
#define FSTRLEN2(S) _fcdlen(S)
#define FSTRLEN3(S) _fcdlen(S)
#define FSTRLEN4(S) _fcdlen(S)
#define FSTRLEN5(S) _fcdlen(S)
#define FSTRPTR(S) _fcdtocp(S)
#define FSETSTRING(S,P,L) S = _cptofcd(P,L);
"""

elif machine in ['MAC']:
  charlen_at_end = 0
  forthonf2c = """
#define FSTRING char*
#define SL1 
#define SL2 
#define SL3 
#define SL4 
#define SL5 
#define FSTRLEN1(S) sl1
#define FSTRLEN2(S) sl2
#define FSTRLEN3(S) sl3
#define FSTRLEN4(S) sl4
#define FSTRLEN5(S) sl5
#define FSTRPTR(S) S
#define FSETSTRING(S,P,L) S = P
#define FSETSTRLEN1(S,L) sl1 = L
#define FSETSTRLEN2(S,L) sl2 = L
#define FSETSTRLEN3(S,L) sl3 = L
#define FSETSTRLEN4(S,L) sl4 = L
#define FSETSTRLEN5(S,L) sl5 = L
"""

else:
  raise 'Machine %s not supported'%machine

# --- Create the forthonf2c.h file
ff = open('forthonf2c.h','w')
ff.write(forthonf2c)
ff.close()


