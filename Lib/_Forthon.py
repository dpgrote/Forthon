"""Python utilities for the Forthon fortran wrapper.
"""
"""
But flies an eagle flight, bold, and forthon, Leaving no tract behind.
"""
# import all of the neccesary packages
from Numeric import *
from types import *
import string
import re
import os
import copy
try:
  import PW
  import PR
except ImportError:
  pass
try:
  import PWpyt
  import PRpyt
except ImportError:
  pass
import __main__
import sys
import cPickle
try:
  import inspect
except ImportError:
  pass
# --- Add line completion capability
try:
  import readline
except ImportError:
  pass
else:
  import rlcompleter
  readline.parse_and_bind("tab: complete")

Forthon_version = "$Id: _Forthon.py,v 1.10 2004/09/03 21:06:24 dave Exp $"

##############################################################################
# --- Functions needed for object pickling
def forthonobject_constructor(typename,dict):
  import __main__
  type = __main__.__dict__[typename]
  obj = type()
  obj.setdict(dict)
  return obj
def pickle_forthonobject(o):
  return (forthonobject_constructor, (o.gettypename(),o.getdict()))


# --- The following routines deal with multiple packages. The ones setting
# --- up or changing the allocation of groups will be called from fortran.

# --- The currently active package defaults to none. Stick this in the
# --- main dictionary. This is somewhat kludgy but I think this is the only
# --- way. The value is put in main so it can be saved in a restart dump
# --- and have the code restarted with the correct package active.
# --- Also, since it is put directly in main, it does not appear in the
# --- list of globals in warp.py and so is not include in 
# --- initial_global_dict_keys which would prevent it from being written
# --- to the dump file.
__main__.__dict__["currpkg"] = ' '
def getcurrpkg():
  return __main__.__dict__["currpkg"]
def setcurrpkg(pkg):
  __main__.__dict__["currpkg"] = pkg

_pkg_dict = {}
_pkg_list = []
def registerpackage(pkg,name):
  """Registers a package so it can be accessed using various global routines."""

  # --- For each package, which has its own type, the pickling functions
  # --- must be registered for that type.
  import copy_reg
  copy_reg.pickle(type(pkg),pickle_forthonobject,forthonobject_constructor)

  _pkg_dict[name] = pkg
  _pkg_list.append(name)

def package(name=None):
  """Sets the default package - sets global currpkg variable.
If name is not given, then returns a copy of the list of all registered
packages.
  """
  if name is None: return copy.deepcopy(_pkg_list)
  setcurrpkg(name)
  _pkg_list.remove(name)
  _pkg_list.insert(0,name)

def packageobject(name):
  """Returns the package object, rather than the package name"""
  return _pkg_dict[name]

def gallot(group='*',iverbose=0):
  """Allocates all dynamic arrays in the specified group.
If the group is not given or is '*', then all groups are allocated.
When optional argument iverbose is true, the variables allocated and their new size is printed.
  """
  for pkg in _pkg_dict.values():
    r = pkg.gallot(group,iverbose)
    if r and (group != '*'): return
  if group != '*': raise "No such group"

def gchange(group='*',iverbose=0):
  """Changes the allocation of all dynamic arrays in the specified group if it is needed.
If the group is not given or is '*', then all groups are changed.
When optional argument iverbose is true, the variables changed and their new size is printed.
  """
  for pkg in _pkg_dict.values():
    r = pkg.gchange(group,iverbose)
    if r and (group != '*'): return
  if group != '*': raise "No such group"

def gfree(group='*'):
  """Frees the allocated memory of all dynamic arrays in the specified group.
If the group is not given or is '*', then all groups are freed.
  """
  for pkg in _pkg_dict.values():
    r = pkg.gfree(group)
    if r and (group != '*'): return
  if group != '*': raise "No such group"

def gsetdims(group='*'):
  """Sets the size in the python database of all dynamic arrays in the specified group.
If the group is not given or is '*', then it is done for all groups.
  """
  for pkg in _pkg_dict.values():
    r = pkg.gsetdims(group)
    if r and (group != '*'): return
  if group != '*': raise "No such group"

def forceassign(name,v):
  """Forces the assignment to an array, resizing the array is necessary.
  """
  for pkg in _pkg_dict.values():
    pkg.forceassign(name,v)

def listvar(name):
  """Lists information about a variable, given the name as a string.
  """
  for pkg in _pkg_dict.values():
    r = pkg.listvar(name)
    if r is not None: return r
  raise NameError

def deprefix():
  """For each variable in the package, a python object is created which has the same name and
same value. For arrays, the new objects points to the same memory location.
  """
  pkglist = package()
  pkglist.reverse()
  for pname in pkglist:
    packageobject(pkg).deprefix()

def reprefix():
  """For each variable in the main dictionary, if there is a package variable with the same name
it is assigned to that value. For arrays, the data is copied.
  """
  for pkg in _pkg_dict.values():
    pkg.reprefix()

def totmembytes():
  """Prints the total amount of memory dynamically allocated for all groups in all packages.
  """
  tot = 0.
  for pkg in _pkg_dict.values():
    tot = tot + pkg.totmembytes()
  return tot

def IsForthonType(v):
  t = repr(type(v))
  if re.search("Forthon",t): return 1
  else: return 0


# --- Some platforms have a different value of .true. in fortran.
if sys.platform in ['sn960510']:
  true = -1
  false = 0
else:
  true = 1
  false = 0

# --- Converts an array of characters into a string.
def arraytostr(a,strip=true):
  a = array(a)
  if len(shape(a)) == 1:
    result = ''
    for c in a:
      result = result + c
    if strip: result = string.strip(result)
  elif len(shape(a)) == 2:
    result = []
    for i in xrange(shape(a)[1]):
      result.append(arraytostr(a[:,i]))
  return result

# --- Allows int operation on arrrays
builtinint = int
def int(x):
  if type(x) == ArrayType:
    return x.astype(Int)
  else:
    return builtinint(x)

# --- Return the nearest integer
def nint(x):
  if type(x) == ArrayType:
    return where(greater(x,0),int(x+0.5),-int(abs(x)+0.5))
  else:
    if x >= 0: return int(x+0.5)
    else: return -int(abs(x)+0.5)

# --- These are replacements for array creation routines which create
# --- arrays which have the proper ordering for fortran. When arrays created
# --- with these commands are passed to a fortran subroutine, no copies are
# --- needed to get the data into the proper order for fortran.
def fones(shape,typecode=Int):
  try:
    s = list(shape)
  except TypeError:
    s = list([shape])
  s.reverse()
  return transpose(ones(s,typecode))
def fzeros(shape,typecode=Int):
  try:
    s = list(shape)
  except TypeError:
    s = list([shape])
  s.reverse()
  return transpose(zeros(s,typecode))

# --- Prints out the documentation of the subroutine or variable.
def doc(f,printit=1):
  # --- The for loop only gives the code something to break out of. There's
  # --- probably a better way of doing this.
  for i in range(1):
    if type(f) == StringType:
        # --- Check if it is a WARP variable
        try:
          d = listvar(f)
          break
        except NameError:
          pass
        # --- Check if it is a module name
        try:
          m = __import__(f)
          try:
            d = m.__dict__[f+'doc']()
            if d is None: d = ''
          except KeyError:
            d = m.__doc__
          break
        except ImportError:
          pass
        # --- Try to get the actual value of the object
        try:
          v = __main__.__dict__[f]
          d = v.__doc__
          break
        except KeyError:
          d = "Name not found"
        except AttributeError:
          d = "No documentation found"
    else:
      # --- Check if it has a doc string
      try:
        d = f.__doc__
      except AttributeError:
        d = "No documentation found"
  if printit: print d
  else:       return d

# --- Get size of all variables in a group
def getgroupsize(pkg,grp):
  ll = pkg.varlist(grp)
  ss = 0
  for v in ll:
    vv = pkg.getpyobject(v)
    if type(vv) == type(array([1])):
      ss = ss + product(array(shape(vv)))
    else:
      ss = ss + 1
  return ss

# --- Print out all variables in a group
def printgroup(pkg,group='',maxelements=10):
  """
Print out all variables in a group or with an attribute
  - pkg: package name
  - group: group name
  - maxelements=10: only up to this many elements of arrays are printed
  """
  if type(pkg) == StringType: pkg = __main__.__dict__[pkg]
  vlist = pkg.varlist(group)
  if not vlist:
    print "Unknown group name "+group
    return
  for vname in vlist:
    v = pkg.getpyobject(vname)
    if v is None:
      print vname+' is not allocated'
    elif type(v) != ArrayType:
      print vname+' = '+str(v)
    else:
      if v.typecode() == 'c':
        print vname+' = "'+str(arraytostr(v))+'"'
      elif size(v) <= maxelements:
        print vname+' = '+str(v)
      else:
        if rank(v) == 1:
          print vname+' = '+str(v[:maxelements])[:-1]+" ..."
        else:
          if shape(v)[0] <= maxelements:
            if rank(v) == 2:
              print vname+' = ['+str(v[:,0])+"] ..."
            elif rank(v) == 3:
              print vname+' = [['+str(v[:,0,0])+"]] ..."
            elif rank(v) == 4:
              print vname+' = [[['+str(v[:,0,0,0])+"]]] ..."
            elif rank(v) == 5:
              print vname+' = [[[['+str(v[:,0,0,0,0])+"]]]] ..."
            elif rank(v) == 6:
              print vname+' = [[[[['+str(v[:,0,0,0,0,0])+"]]]]] ..."
          else:
            if rank(v) == 2:
              print vname+' = ['+str(v[:maxelements,0])[:-1]+" ..."
            elif rank(v) == 3:
              print vname+' = [['+str(v[:maxelements,0,0])[:-1]+" ..."
            elif rank(v) == 4:
              print vname+' = [[['+str(v[:maxelements,0,0,0])[:-1]+" ..."
            elif rank(v) == 5:
              print vname+' = [[[['+str(v[:maxelements,0,0,0,0])[:-1]+" ..."
            elif rank(v) == 6:
              print vname+' = [[[[['+str(v[:maxelements,0,0,0,0,0])[:-1]+" ..."
  
##############################################################################
##############################################################################
def pydumpforthonobject(ff,attr,objname,obj,varsuffix,writtenvars,fobjlist,
                        serial,verbose,lonlymakespace=0):
  # --- General work of this object
  if verbose: print "object "+objname+" being written"
  # --- Write out the value of fobj so that in restore, any links to this
  # --- object can be restored. Only do this if fobj != 0, which means that
  # --- it is not a top level package, but a variable of fortran derived type.
  fobj = obj.getfobject()
  if fobj != 0:
    ff.write('FOBJ'+varsuffix,fobj)
    ff.write('TYPENAME'+varsuffix,obj.gettypename())
    # --- If this object has already be written out, then return.
    if fobj in fobjlist: return
    # --- Add this object to the list of object already written out.
    fobjlist.append(fobj)
  # --- Get variables in this package which have attribute attr.
  vlist = []
  for a in attr:
    if type(a) == StringType: vlist = vlist + obj.varlist(a)
  # --- Loop over list of variables
  for vname in vlist:
    # --- Check if object is available (i.e. check if dynamic array is
    # --- allocated).
    v = obj.getpyobject(vname)
    if v is None: continue
    # --- If serial flag is set, get attributes and if has the parallel
    # --- attribute, don't write it.
    if serial:
      a = obj.getvarattr(vname)
      if re.search('parallel',a):
        if verbose: print "variable "+vname+varsuffix+" skipped since it is a parallel variable"
        continue
    # --- Check if variable is a complex array. Currently, these
    # --- can not be written out.
    if type(v) == ArrayType and v.typecode() == Complex:
      if verbose: print "variable "+vname+varsuffix+" skipped since it is a complex array"
      continue
    # --- Check if variable with same name has already been written out.
    # --- This only matters when the variable is being written out as
    # --- a plane python variable.
    if '@' not in varsuffix:
      if vname in writtenvars:
        if verbose: print "variable "+objname+"."+vname+" skipped since other variable would have same name in the file"
        continue
      writtenvars.append(vname)
    # --- Check if variable is a Forthon object, if so, recursively call this
    # --- function.
    if IsForthonType(v):
      # --- Note that the attribute passed in is blank, since all components
      # --- are to be written out to the file.
      pydumpforthonobject(ff,[''],vname,v,'@'+vname+varsuffix,writtenvars,
                          fobjlist,serial,verbose,lonlymakespace)
      continue
    # --- If this point is reached, then variable is written out to file
    if verbose: print "writing "+objname+"."+vname+" as "+vname+varsuffix
    # --- If lonlymakespace is true, then use defent to create space in the
    # --- file for arrays but don't write out any data. Scalars are still
    # --- written out.
    if lonlymakespace and type(v) not in [IntType,FloatType]:
      ff.defent(vname+varsuffix,v,shape(v))
    else:
      ff.write(vname+varsuffix,v)

##############################################################################
# Python version of the dump routine. This uses the varlist command to
# list of all of the variables in each package which have the
# attribute attr (and actually attr could be a group name too). It then
# checks on the state of the python object, making sure that unallocated
# arrays are not written out.  Finally, the variable is written out to the
# file with the name in the format vame@pkg.  Additionally, python
# variables can be written to the file by passing in a list of the names
# through vars. The '@' sign is used between the package name and the
# variable name so that no python variable names can be clobbered ('@'
# is not a valid character in python names). The 'ff.write' command is
# used, allowing names with an '@' in them. The writing of python variables
# is put into a 'try' command since some variables cannot be written to
# a pdb file.
def pydump(fname=None,attr=["dump"],vars=[],serial=0,ff=None,varsuffix=None,
           verbose=false,hdf=0,returnfobjlist=0,lonlymakespace=0):
  """
Dump data into a pdb file
  - fname: dump file name
  - attr=["dump"]: attribute or list of attributes of variables to dump
       Any items that are not strings are skipped. To write no variables,
       use attr=None.
  - vars=[]: list of python variables to dump
  - serial=0: switch between parallel and serial versions
  - ff=None: Allows passing in of a file object so that pydump can be called
       multiple times to pass data into the same file. Note that
       the file must be explicitly closed by the user.
  - varsuffix=None: Suffix to add to the variable names. If none is specified,
       the suffix '@pkg' is used, where pkg is the package name that the
       variable is in. Note that if varsuffix is specified, the simulation
       cannot be restarted from the dump file.
  - verbose=false: When true, prints out the names of the variables as they are
       written to the dump file
  - hdf=0: when true, dump into an HDF file rather than a PDB.
  - returnfobjlist=0: when true, returns the list of fobjects that were
                      written to the file
  """
  assert fname is not None or ff is not None,\
         "Either a filename must be specified or a pdb file pointer"
  # --- Open the file if the file object was not passed in.
  # --- If the file object was passed in, then don't close it.
  if ff is None:
    if not hdf:
      # --- Try to open file with PDB format as requested.
      try:
        ff = PW.PW(fname)
        # --- With PDB, pickle dumps can only be done in ascii.
        dumpsmode = 0
      except:
        pass
    if hdf or ff is None:
      # --- If HDF requested or PDB not available, try HDF.
      try:
        ff = PWpyt.PW(fname)
        # --- An advantage of HDF is that pickle dumps can be done in binary
        dumpsmode = 1
      except:
        pass
    if hdf and ff is None:
      # --- If HDF was requested and didn't work, try PDB anyway.
      try:
        ff = PW.PW(fname)
        # --- With PDB, pickle dumps can only be done in ascii.
        dumpsmode = 0
      except:
        pass
    assert ff is not None,"Dump file cannot be opened, no data formats available"
    closefile = 1
  else:
    try:
      if ff.file_type == "HDF":
        dumpsmode = 1
      else:
        dumpsmode = 0
    except:
      dumpsmode = 0
    closefile = 0
  # --- Make sure the file has a file_type. Older versions of the pdb
  # --- wrapper did not define a file type.
  try:
    ff.file_type
  except:
    ff.file_type = 'oldPDB'
  # --- Convert attr into a list if needed
  if not (type(attr) == ListType): attr = [attr]
  # --- Loop through all of the packages (getting pkg object).
  # --- When varsuffix is specified, the list of variables already written
  # --- is created. This solves two problems. It gives proper precedence to
  # --- variables of the same name in different packages. It also fixes
  # --- an obscure bug in the pdb package - writing two different arrays with
  # --- the same name causes a problem and the pdb file header is not
  # --- properly written. The pdb code should really be fixed.
  pkgsuffix = varsuffix
  packagelist = package()
  writtenvars = []
  fobjlist = []
  for pname in packagelist:
    pkg = __main__.__dict__[pname]
    if varsuffix is None: pkgsuffix = '@' + pname
    pydumpforthonobject(ff,attr,pname,pkg,pkgsuffix,writtenvars,fobjlist,
                        serial,verbose,lonlymakespace)

  # --- Now, write out the python variables (that can be written out).
  # --- If supplied, the varsuffix is append to the names here too.
  if varsuffix is None: varsuffix = ''
  for vname in vars:
    # --- Skip python variables that would overwrite fortran variables.
    if len(writtenvars) > 0:
      if vname in writtenvars:
        if verbose: print "variable "+vname+" skipped since other variable would have same name in the file"
        continue
    # --- Get the value of the variable.
    vval = __main__.__dict__[vname]
    # --- Write out the source of functions. Note that the source of functions
    # --- typed in interactively is not retrieveable - inspect.getsource
    # --- returns an IOError.
    if type(vval) in [FunctionType]:
      try:
        source = inspect.getsource(vval)
        #if verbose:
        if verbose: print "writing python function "+vname+" as "+vname+varsuffix+'@function'
        ff.write(vname+varsuffix+'@function',source)
      except (IOError,NameError):
        if verbose: print "could not write python function "+vname
      continue
    # --- Zero length arrays cannot by written out.
    if type(vval) == ArrayType and product(array(shape(vval))) == 0:
      continue
    # --- Check if variable is a Forthon object.
    if IsForthonType(vval):
      pydumpforthonobject(ff,attr,vname,vval,'@'+vname+varsuffix,writtenvars,
                          fobjlist,serial,verbose,lonlymakespace)
      continue
    # --- Try writing as normal variable.
    # --- The docontinue temporary is needed since python1.5.2 doesn't
    # --- seem to like continue statements inside of try statements.
    docontinue = 0
    try:
      if verbose: print "writing python variable "+vname+" as "+vname+varsuffix
      ff.write(vname+varsuffix,vval)
      docontinue = 1
    except:
      pass
    if docontinue: continue
    # --- If that didn't work, try writing as a pickled object
    # --- This is only needed for the old pdb wrapper. The new one
    # --- automatically pickles things as needed.
    if ff.file_type == 'oldPDB':
      try:
        if verbose:
          print "writing python variable "+vname+" as "+vname+varsuffix+'@pickle'
        ff.write(vname+varsuffix+'@pickle',cPickle.dumps(vval,dumpsmode))
        docontinue = 1
      except (cPickle.PicklingError,TypeError):
        pass
      if docontinue: continue
    # --- All attempts failed so write warning message
    if verbose: print "cannot write python variable "+vname
  if closefile: ff.close()

  # --- Return the fobjlist for cases when pydump is called multiple times
  # --- for a single file.
  if returnfobjlist: return fobjlist


#############################################################################
# Python version of the restore routine. It restores all of the variables
# in the pdb file.
# An '@' in the name distinguishes between the two. The 'ff.__getattr__' is
# used so that variables with an '@' in the name can be read. The reading
# in of python variables is put in a 'try' command to make it idiot proof.
# More fancy foot work is done to get new variables read in into the
# global dictionary.
def pyrestore(filename=None,fname=None,verbose=0,skip=[],ff=None,
              varsuffix=None,ls=0,lreturnfobjdict=0):
  """
Restores all of the variables in the specified file.
  - filename: file to read in from (assumes PDB format)
  - verbose=0: When true, prints out the names of variables which are read in
  - skip=[]: list of variables to skip
  - ff=None: Allows passing in of a file object so that pydump can be called
       multiple times to pass data into the same file. Note that
       the file must be explicitly closed by the user.
  - varsuffix: when set, all variables read in will be given the suffix
               Note that fortran variables are then read into python vars
  - ls=0: when true, prints a list of the variables in the file
          when 1 prints as tuple
          when 2 prints in a column
Note that it will automatically detect whether the file is PDB or HDF.
  """
  assert filename is not None or fname is not None or ff is not None,\
         "Either a filename must be specified or a pdb file pointer"
  if ff is None:
    # --- The original had fname, but changed to filename to be consistent
    # --- with restart and dump.
    if filename is None: filename = fname
    # --- Make sure a filename was input.
    assert filename is not None,"A filename must be specified"
    # --- Check if file exists
    assert os.access(filename,os.F_OK),"File %s does not exist"%filename
    # --- open pdb file
    try:
      ff = PR.PR(filename)
    except:
      ff = PRpyt.PR(filename)
    closefile = 1
  else:
    closefile = 0
  # --- Make sure the file has a file_type. Older versions of the pdb
  # --- wrapper did not define a file type.
  try:
    ff.file_type
  except:
    ff.file_type = 'oldPDB'
  # --- Get a list of all of the variables in the file, loop over that list
  vlist = ff.inquire_names()
  # --- Print list of variables
  if ls:
    if ls == 1:
      print vlist
    else:
      for l in vlist: print l

  # --- First, sort out the list of variables
  groups = sortrestorevarsbysuffix(vlist,skip)
  fobjdict = {}

  # --- Read in the variables with the standard suffices.

  # --- These would be interpreter variables written to the file
  # --- from python (or other sources). A simple assignment is done and
  # --- the variable in put in the main dictionary.
  if groups.has_key(''):
    plist = groups['']
    del groups['']
    for vname in plist:
      pyname = vname
      if varsuffix is not None: pyname = pyname + str(varsuffix)
      try:
        if verbose: print "reading in python variable "+vname
        __main__.__dict__[pyname] = ff.__getattr__(vname)
      except:
        if verbose: print "error with variable "+vname

  # --- These would be interpreter variables written to the file
  # --- as pickled objects. The data is unpickled and the variable
  # --- in put in the main dictionary.
  # --- This is only needed with the old pdb wrapper.
  if ff.file_type == 'oldPDB':
    if groups.has_key('pickle'):
      picklelist = groups['pickle']
      del groups['pickle']
      for vname in picklelist:
        pyname = vname
        if varsuffix is not None: pyname = pyname + str(varsuffix)
        try:
          if verbose: print "reading in pickled variable "+vname
          __main__.__dict__[pyname]=cPickle.loads(ff.__getattr__(vname+'@pickle'))
        except:
          if verbose: print "error with variable "+vname

  # --- These would be interpreter variables written to the file
  # --- from Basis. A simple assignment is done and the variable
  # --- in put in the main dictionary.
  if groups.has_key('global'):
    globallist = groups['global']
    del groups['global']
    for vname in globallist:
      pyname = vname
      if varsuffix is not None: pyname = pyname + str(varsuffix)
      try:
        if verbose: print "reading in Basis variable "+vname
        __main__.__dict__[pyname] = ff.__getattr__(vname+'@global')
      except:
        if verbose: print "error with variable "+vname

  # --- User defined Python functions
  if groups.has_key('function'):
    functionlist = groups['function']
    del groups['function']
    for vname in functionlist:
      # --- Skip functions which have already been defined in case the user
      # --- has made source updates since the dump was made.
      if __main__.__dict__.has_key(vname): 
        if verbose:
          print "skipping python function %s since it already is defined"%vname
      else:
        try:
          if verbose: print "reading in python function"+vname
          source = ff.__getattr__(vname+'@function')
          exec(source,__main__.__dict__)
        except:
          if verbose: print "error with function "+vname

  # --- Ignore variables with suffix @parallel
  if groups.has_key('parallel'):
    del groups['parallel']

  for gname in groups.keys():
    pyrestoreforthonobject(ff,gname,groups[gname],fobjdict,varsuffix,
                           verbose,doarrays=0)
  for gname in groups.keys():
    pyrestoreforthonobject(ff,gname,groups[gname],fobjdict,varsuffix,
                           verbose,doarrays=1)

  if closefile: ff.close()
  if lreturnfobjdict: return fobjdict

def sortrestorevarsbysuffix(vlist,skip):
  # --- Sort the variables, collecting them in groups based on there suffix.
  groups = {}
  for v in vlist:
    if '@' in v:
      i = string.rfind(v,'@')
      vname = v[:i]
      gname = v[i+1:]
    else:
      # --- Otherwise, variable is plain python variable.
      vname = v
      gname = ''

    # --- If variable is in the skip list, then skip
    if (vname in skip or
        (len(v) > 4 and v[-4]=='@' and v[-3:]+'.'+v[:-4] in skip)):
#     if verbose: print "skipping "+v
      continue

    # --- Now add the variable to the appropriate group list.
    groups.setdefault(gname,[]).append(vname)

  return groups

#-----------------------------------------------------------------------------
def pyrestoreforthonobject(ff,gname,vlist,fobjdict,varsuffix,verbose,doarrays,
                           gpdbname=None):
  """
  - ff: reference to file being written to
  - gname: name (in python format) of object to read in
  - vlist: list of variables from the file that are part of this object
 - fobjdist: dictionary of objects already read in
  - varsuffix: suffix to apply to all variables (in python)
  - verbose: when true, lists whether each variable is successfully read in
  - doarrays: when true, reads in arrays, otherwise only scalars
  - gpdbname: actual name of object in the data file. If None, extracted
              from gname.
  """

  # --- Convert gname in pdb-style name
  if gpdbname is None:
    gsplit = string.split(gname,'.')
    gsplit.reverse()
    gpdbname = string.join(gsplit,'@')

  # --- Check if the variable gname exists or is allocated.
  # --- If not, create a new variable.
  neednew = 0
  try:
    v = eval(gname,__main__.__dict__)
    if v is None: neednew = 1
  except:
    neednew = 1

  if neednew:
    # --- A new variable needs to be created.
    try:
      fobj = ff.read("FOBJ@"+gpdbname)
    except:
      return
    # --- First, check if the object has already be restored.
    if fobj in fobjdict:
      # --- If so, then point new variable to existing object
      exec("%s = %s"%(gname,fobjdict[fobj]),__main__.__dict__)
      # return ???
    else:
      # --- Otherwise, create a new instance of the appropriate type,
      # --- and add it to the list of objects.
      typename = ff.read("TYPENAME@"+gpdbname)
      exec("%s = %s()"%(gname,typename),__main__.__dict__)
      fobjdict[fobj] = gname

  # --- Sort out the list of variables
  groups = sortrestorevarsbysuffix(vlist,[])

  # --- Get "leaf" variables
  if groups.has_key(''):
    leafvars = groups['']
    del groups['']
  else:
    leafvars = []

  # --- Read in leafs.
  for vname in leafvars:
    if vname == 'FOBJ' or vname == 'TYPENAME': continue
    fullname = gname + '.' + vname
    vpdbname = vname + '@' + gpdbname

    # --- Add suffix to name if given.
    # --- varsuffix is wrapped in str in case a nonstring was passed in.
    if varsuffix is not None: fullname = vname + str(varsuffix)

    try:
      if type(ff.__getattr__(vpdbname)) != ArrayType and not doarrays:
        # --- Simple assignment is done for scalars, using the exec command
        if verbose: print "reading in "+fullname
        exec(fullname+'=ff.__getattr__(vpdbname)',__main__.__dict__,locals())
      elif type(ff.__getattr__(vpdbname)) == ArrayType and doarrays:
        pkg = eval(gname,__main__.__dict__)
        # --- forceassign is used, allowing the array read in to have a
        # --- different size than the current size of the warp array.
        if verbose: print "reading in "+gname+"."+fullname
        pkg.forceassign(vname,ff.__getattr__(vpdbname))
    except:
      # --- The catches errors in cases where the variable is not an
      # --- actual warp variable, for example if it had been deleted
      # --- after the dump was originally made.
      print "Warning: There was problem restoring %s"% (fullname)

  # --- Read in rest of groups.
  for g,v in groups.items():
    pyrestoreforthonobject(ff,gname+'.'+g,v,fobjdict,varsuffix,verbose,doarrays,
                           g+'@'+gpdbname)


# --- create an alias for pyrestore
restore = pyrestore

##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################
##############################################################################

def Forthondoc():
  print """
package(): sets active package or returns list of all packages
gallot(): allocates all dynamic arrays in a group
gchange(): changes all dynamic arrays in a group if needed
gfree(): free all dynamica arrays in a group
gsetdims(): setups up dynamic arrays sizes in datebase
forceassign(): forces assignment to a dynamic array, resizing if necessary
listvar(): prints information about a variable
deprefix(): creates a python variable for each package variable
reprefix(): copies python variables into packages variables of the same name
totmembytes(): returns total memory allocated for dynamic arrays
arraytostr(): converts an array of chars to a string
int(): converts data to integer
nint(): converts data to nearest integer
fones(): returns multi-dimensional array with fortran ordering
fzeros(): returns multi-dimensional array with fortran ordering
doc(): prints info about variables and functions
printgroup(): prints all variables in the group or with an attribute
pydump(): dumps data into pdb format file
pyrestore(): reads data from pdb format file
restore(): equivalent to pyrestore
"""
