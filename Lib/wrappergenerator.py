#!/usr/bin/env python
# Python wrapper generation
# Created by David P. Grote, March 6, 1998
# Modified by T. B. Yang, May 21, 1998
# $Id: wrappergenerator.py,v 1.2 2004/01/08 22:55:22 dave Exp $

import sys
import interfaceparser
import string
import re
import fvars
import getopt
import pickle
from cfinterface import *
import wrappergen_derivedtypes

class PyMAC:
  """
Usage:
  -a       All groups will be allocated on initialization
  -t ARCH  Build for specified architecture (default is HP700)
  -d <.scalars file>  a .scalars file in another module that this module depends on
  --f90    F90 syntax will be assumed
  --f90f   F90 syntax will be assumed, dynamic arrays allocated in fortran
  --nowritemodules The modules will not be written out, assuming
                   that they are already written.
  --macros pkg.v Other interface files that are needed for the definition
                 of macros.
  file1    Main variable description file for the package
  [file2, ...] Subsidiary variable description files
  """

  def __init__(self,pname,f90=1,f90f=0,initialgallot=1,writemodules=1,
               otherfiles=[],other_scalar_dicts=[]):
    self.pname = pname
    self.f90 = f90
    self.f90f = f90f
    self.initialgallot = initialgallot
    self.writemodules = writemodules
    self.otherfiles = otherfiles
    self.other_scalar_dicts = other_scalar_dicts
    self.isz = isz # isz defined in cfinterface

    self.createmodulefile()

  def cname(self,n):
    # --- Standard name of the C interface to a Fortran routine
    # --- pkg_varname
    return self.pname+'_'+n

  def prefixdimsc(self,dim,sdict):
    # --- Convert fortran variable name into reference from list of variables.
    sl=re.split('[ ()/\*\+\-]',dim)
    for ss in sl:
      if re.search('[a-zA-Z]',ss) != None:
        if sdict.has_key (ss):
          dim = re.sub(ss,
                   '*(int *)'+self.pname+'_fscalars['+repr(sdict[ss])+'].data',
                   dim,count=1)
        else:
          for other_dict in self.other_scalar_dicts:
            if other_dict.has_key (ss):
              dim = re.sub(ss,'*(int *)'+other_dict['_module_name_']+
                        '_fscalars['+repr(other_dict[ss])+'].data',dim,count=1)
              break
          else:
            raise ss + ' is not declared in a .v file'
    return string.lower(dim)

  # --- Convert dimensions for unspecified arrays
  def prefixdimsf(self,dim):
    # --- Check for any unspecified dimensions and replace it with an element
    # --- from the dims array.
    sl = re.split(',',dim[1:-1])
    for i in range(len(sl)):
      if sl[i] == ':': sl[i] = 'dims__(%d)'%(i+1)
    dim = '(' + string.join(sl,',') + ')'
    return string.lower(dim)

  def dimsgroups(self,dim,sdict,slist):
    # --- Returns a list of group names that contain the variables listed in
    # --- a dimension statement
    groups = []
    sl=re.split('[ (),:/\*\+\-]',dim)
    for ss in sl:
      if re.search('[a-zA-Z]',ss) != None:
        if sdict.has_key (ss):
          groups.append(slist[sdict[ss]].group)
        else:
          raise ss + ' is not declared in a .v file'
    return groups

  def cw(self,text,noreturn=0):
    if noreturn:
      self.cfile.write(text)
    else:
      self.cfile.write(text+'\n')
  def fw(self,text,noreturn=0):
    if noreturn:
      self.ffile.write(text)
    else:
      self.ffile.write(text+'\n')

  def createmodulefile(self):
    # --- This is the routine that does all of the work

    # --- Get the list of variables and subroutine from the var file
    vlist,hidden_vlist,typelist = interfaceparser.processfile(self.pname,
                                                self.pname+'.v',self.otherfiles)
    if not vlist and not hidden_vlist and not typelist:
      return

    # --- Get a list of all of the group names which have variables in it
    # --- (only used when writing fortran files but done here while complete
    # --- list of variables is still in one place, vlist).
    currentgroup = ''
    groups = []
    for v in vlist:
      if not v.function and v.group != currentgroup:
        groups.append(v.group)
        currentgroup = v.group

    # --- Get a list of all of the hidden group names.
    current_hidden_group = ''
    hidden_groups = []
    for hv in hidden_vlist:
      if not hv.function and hv.group != current_hidden_group:
        hidden_groups.append(hv.group)
        current_hidden_group = hv.group
  
    # --- Select out all of the scalars and build a dictionary
    # --- The dictionary is used to get number of the variables use as
    # --- dimensions for arrays.
    slist = []
    sdict = {}
    i = 0
    temp = vlist[:]
    for v in temp:
      if not v.dims and not v.function:
        slist.append(v)
        sdict[v.name] = i
        i = i + 1
        vlist.remove(v)

    # --- Select out all of the arrays
    alist = []
    i = 0
    temp = vlist[:]
    for v in temp:
      if not v.function:
        alist.append(v)
        i = i + 1
        vlist.remove(v)
    temp = []

    # --- The remaining elements should all be functions
    flist = vlist

    ############################################################################
    # --- Create the module file
    self.cfile = open(self.pname+'pymodule.c','w')
    self.cw('#include "Forthon.h"')
    self.cw('ForthonObject *'+self.pname+'Object;')

    # --- Print out the external commands
    self.cw('extern void '+fname(self.pname+'passpointers')+'();')
    if not self.f90 and not self.f90f:
      self.cw('extern void '+self.pname+'data();')

    # --- fortran routine prototypes
    for f in flist:
      # --- Functions
      self.cw('extern '+fvars.ftoc(f.type)+' '+fnameofobj(f)+'(',noreturn=1)
      i = 0
      istr = 0
      for a in f.args:
        if i > 0:
          self.cw(',',noreturn=1)
        i = i + 1
        self.cw(fvars.ftoc(a.type)+' ',noreturn=1)
        if a.type == 'string' or a.type == 'character':
          istr = istr + 1
        else:
          self.cw('*',noreturn=1)
        self.cw(a.name,noreturn=1)
      if charlen_at_end:
        for i in range(istr):
          self.cw(',int sl'+repr(i),noreturn=1)
      self.cw(');')
    for t in typelist:
      self.cw('extern PyObject *'+self.cname(t.name)+'New();')
    self.cw('')

    # --- setpointer and getpointer routine for f90
    if self.f90:
      for s in slist:
        if s.dynamic:
          self.cw('extern void '+fname(self.pname+'setpointer'+s.name)+
                  '(char *p,long *cobj__);')
          self.cw('extern void '+fname(self.pname+'getpointer'+s.name)+
                  '(ForthonObject **cobj__,long *obj);')
      for a in alist:
        self.cw('extern void '+fname(self.pname+'setpointer'+a.name)+
                '(char *p,long *cobj__,long *dims__);')
        if re.search('fassign',a.attr):
          self.cw('extern void '+fname(self.pname+'getpointer'+a.name)+
                  '(long *i,long *cobj__);')
    if self.f90f:
      for s in slist:
        if s.dynamic:
          self.cw('extern void '+fname(self.pname+'setpointer'+s.name)+
                  '(PyObject *p,long *cobj__);')
          self.cw('extern void '+fname(self.pname+'getpointer'+s.name)+
                  '(ForthonObject **cobj__,long *obj);')
      for a in alist:
        self.cw('extern void '+fname(self.pname+'setpointer'+a.name)+
                '(PyObject *p,long *cobj__,long *dims__);')
        if re.search('fassign',a.attr):
          self.cw('extern void '+fname(self.pname+'getpointer'+a.name)+
                  '(long *i,long *cobj__);')

    ###########################################################################
    # --- Write declarations of c pointers to fortran variables

    # --- Declare scalars from other modules
    for other_dict in self.other_scalar_dicts:
        self.cw('extern Fortranscalars '+other_dict['_module_name_']+
                '_fscalars[];')

    # --- Scalars
    self.cw('int '+self.pname+'nscalars = '+repr(len(slist))+';')
    if len(slist) > 0:
      self.cw('Fortranscalars '+self.pname+'_fscalars['+repr(len(slist))+']={')
      for i in range(len(slist)):
        s = slist[i]
        if (self.f90 or self.f90f) and s.dynamic:
          setpointer = '*'+fname(self.pname+'setpointer'+s.name)
          getpointer = '*'+fname(self.pname+'getpointer'+s.name)
        else:
          setpointer = 'NULL'
          getpointer = 'NULL'
        self.cw('{PyArray_%s,'%fvars.ftop(s.type) + 
                 '"%s",'%s.name + 
                 'NULL,' + 
                 '"%s",'%s.group + 
                 '"%s",'%s.attr + 
                 '"%s",'%string.replace(s.comment,'"','\\"') + 
                 '%i,'%s.dynamic + 
                 '%s,'%setpointer + 
                 '%s}'%getpointer,noreturn=1)
        if i < len(slist)-1: self.cw(',')
      self.cw('};')
    else:
      self.cw('Fortranscalars *'+self.pname+'_fscalars=NULL;')

    # --- Arrays
    self.cw('int '+self.pname+'narrays = '+repr(len(alist))+';')
    if len(alist) > 0:
      self.cw('static Fortranarrays '+
              self.pname+'_farrays['+repr(len(alist))+']={')
      for i in range(len(alist)):
        a = alist[i]
        if (self.f90 or self.f90f) and a.dynamic:
          setpointer = '*'+fname(self.pname+'setpointer'+a.name)
        else:
          setpointer = 'NULL'
        if (self.f90 or self.f90f) and re.search('fassign',a.attr):
          getpointer = '*'+fname(self.pname+'getpointer'+a.name)
        else:
          getpointer = 'NULL'
        if a.data and a.dynamic:
          initvalue = a.data[1:-1]
        else:
          initvalue = '0'
        self.cw('{PyArray_%s,'%fvars.ftop(a.type) +
                  '%d,'%a.dynamic +
                  '%d,'%len(a.dims) +
                  'NULL,' +
                  '"%s",'%a.name +
                  'NULL,' +
                  '%s,'%setpointer +
                  '%s,'%getpointer +
                  '%s,'%initvalue +
                  'NULL,' +
                  '"%s",'%a.group +
                  '"%s",'%a.attr +
                  '"%s",'%string.replace(a.comment,'"','\\"') +
                  '"%s"}'%a.dimstring,noreturn=1)
        if i < len(alist)-1: self.cw(',')
      self.cw('};')
    else:
      self.cw('static Fortranarrays *'+self.pname+'_farrays=NULL;')

    ###########################################################################
    ###########################################################################
    # --- Now, the fun part, writing out the wrapper for the subroutine and
    # --- function calls.
    for f in flist:
      # --- Write out the documentation first.
      docstring = ('static char doc_'+self.cname(f.name)+'[] = "'+f.name+
                   f.dimstring+f.comment+'";')
      # --- Replaces newlines with '\\n' so that the string is all on one line
      # --- in the C coding.
      docstring = re.sub(r'\
  ','\\\\n',docstring)
      self.cw(docstring)
      # --- Now write out the wrapper
      self.cw('static PyObject *')
      self.cw(self.cname(f.name)+'(PyObject *self, PyObject *args)')
      self.cw('{')

      if f.args == []:
        # --- Function with no arguments is easy.
        if f.type == 'void':
          self.cw('  '+fnameofobj(f)+'();')
          self.cw('  returnnone;')
        else:
          self.cw('  return Py_BuildValue("'+fvars.fto1[f.type]+'",'+
                  fnameofobj(f)+'());')
        self.cw('}')
        continue

      # --- With arguments, it gets very messy
      lv = repr(len(f.args))
      self.cw('  PyObject * pyobj['+lv+'];')
      self.cw('  PyArrayObject * ax['+lv+'];')
      self.cw('  PyObject *t;')
      self.cw('  int i,argno=0;')
      self.cw('  char e[80];')

      # --- For character arguments, need to create an FSTRING array.
      istr = 0
      for a in f.args:
        if a.type == 'string' or a.type == 'character':
          istr = istr + 1
      if istr > 0:
        self.cw('  FSTRING fstr['+repr(istr)+'];')

      # --- If this is a function, set up variables to hold return value
      if f.type != 'void':
        self.cw('  PyObject * ret_val;')
        self.cw('  '+fvars.ftoc(f.type)+' r;')

      # --- Parse incoming arguments into a list of PyObjects
      self.cw('  if (!PyArg_ParseTuple(args, "'+'O'*len(f.args)+'"',noreturn=1)
      for i in range(len(f.args)):
        self.cw(',&pyobj['+repr(i)+']',noreturn=1)
      self.cw(')) return NULL;')

      # --- Loop over arguments, extracting the data addresses.
      # --- Convert all arguments into arrays. This allows complete flexibility
      # --- in what can be passed to fortran functions. The caveat is that it
      # --- does no type checking and no array size checking.
      istr = 0
      for i in range(len(f.args)):
        self.cw('  argno++;')
        if not fvars.isderivedtype(f.args[i]):
          self.cw('  FARRAY_FROMOBJECT(ax['+repr(i)+'],'+
                'pyobj['+repr(i)+'], PyArray_'+fvars.ftop(f.args[i].type)+');')
          self.cw('  if (ax['+repr(i)+'] == NULL) goto err;')
          if f.args[i].type == 'string' or f.args[i].type == 'character':
            self.cw('  FSETSTRING(fstr[%d],ax[%d]->data,PyArray_SIZE(ax[%d]));'
                    %(istr,i,i))
            istr = istr + 1
        else:
          self.cw('  ax['+repr(i)+'] = NULL;')
          self.cw('  t = PyObject_Type(pyobj['+repr(i)+']);')
          self.cw('  if (strcmp(((PyTypeObject *)t)->tp_name,"Forthon") != 0)'+
                     'goto err;')
          self.cw('  Py_DECREF(t);')
          self.cw('  if (((ForthonObject *)pyobj['+repr(i)+'])->typename!="'+
                     f.args[i].type+'") goto err;')

      # --- Write the actual call to the fortran routine.
      if f.type == 'void':
        self.cw('')
      else:
        self.cw('  r = ')
      self.cw(fnameofobj(f)+'(',noreturn=1)
      i = 0
      istr = 0
      for a in f.args:
        if i > 0:
          self.cw(',',noreturn=1)
        if fvars.isderivedtype(a):
          self.cw('((ForthonObject *)(pyobj['+repr(i)+']))->fobj',noreturn=1)
        elif a.type == 'string' or a.type == 'character':
          self.cw('fstr[%d]'%(istr),noreturn=1)
          istr = istr + 1
        else:
          self.cw('('+fvars.ftoc(a.type)+' *)(ax['+repr(i)+']->data)',noreturn=1)
        i = i + 1
      if charlen_at_end:
        i = 0
        istr = 0
        for a in f.args:
          if a.type == 'string' or a.type == 'character':
            self.cw(',PyArray_SIZE(ax['+repr(i)+'])',noreturn=1)
            istr = istr + 1
          i = i + 1

      self.cw(');') # --- Closing parenthesis on the call list

      # --- Copy the data that was sent to the routine back into the passed
      # --- in object if it is an PyArray. This needs to be thoroughly checked.
      # --- Decrement reference counts of array objects created.
      self.cw('  for (i=0;i<'+repr(len(f.args))+';i++) {')
      self.cw('    if (PyArray_Check(pyobj[i])) {')
      self.cw('      if (pyobj[i] != (PyObject *)ax[i])')
      self.cw('        PyArray_CopyArray((PyArrayObject *)pyobj[i],ax[i]);}')
      self.cw('    if (ax[i] != NULL) Py_XDECREF(ax[i]);}')

      # --- Write return sequence
      if f.type == 'void':
        self.cw('  returnnone;')
      else:
        self.cw('  ret_val = Py_BuildValue ("'+fvars.fto1[f.type]+'", r);')
        self.cw('  return ret_val;')

      # --- Error section
      self.cw('err:') 

      # --- Decrement reference counts of array objects created.
      self.cw('  sprintf(e,"There is an error in argument %d",argno);')
      self.cw('  PyErr_SetString(ErrorObject,e);')
      self.cw('  for (i=0;i<'+repr(len(f.args))+';i++)')
      self.cw('    if (ax[i] != NULL) Py_XDECREF(ax[i]);')
      self.cw('  return NULL;')

      self.cw('}')

    # --- Add blank line
    self.cw('')

    ###########################################################################
    # --- Write out method list
    self.cw('static struct PyMethodDef '+self.pname+'_methods[] = {')
    for f in flist:
      if f.function:
        self.cw('{"'+f.name+'",(PyCFunction)'+self.cname(f.name)+',1,'+
                'doc_'+self.cname(f.name)+'},')
    for t in typelist:
      self.cw('{"'+t.name+'",(PyCFunction)'+self.cname(t.name)+'New,1,'+
              '"Creates a new instance of fortran derived type '+t.name+'"},')
    self.cw('{NULL,NULL}};')
    self.cw('')

    ###########################################################################
    # --- Write static array initialization routines
    self.cw('void '+self.pname+'setstaticdims(ForthonObject *self)')
    self.cw('{')
  
    i = -1
    for a in alist:
      i = i + 1
      vname = self.pname+'_farrays['+repr(i)+']'
      if a.dims and not a.dynamic:
        j = 0
        for d in a.dims:
          self.cw('  '+vname+'.dimensions['+repr(len(a.dims)-1-j)+'] = ('+
                  d.high+') - ('+d.low+') + 1;')
          j = j + 1

    self.cw('}')
    self.cw('')

    ###########################################################################
    # --- Write routine which sets the dimensions of the dynamic arrays.
    # --- This is done in a seperate routine so it only appears once.
    # --- A routine is written out for each group which has dynamic arrays. Then
    # --- a routine is written which calls all of the individual group routines.
    # --- That is done to reduce the strain on the compiler by reducing the size
    # --- of the routines. (In fact, in one case, with everything in one
    # --- routine the cc compiler was giving a core dump!)
    # --- Loop over the variables. This assumes that the variables are sorted
    # --- by group.
    i = -1
    currentgroup = ''
    dyngroups = []
    for a in alist:
      if a.group != currentgroup and a.dynamic:
        if currentgroup != '':
          self.cw('  }}')
        currentgroup = a.group
        dyngroups.append(currentgroup)
        self.cw('static void '+self.pname+'setdims'+currentgroup+'(char *name)')
        self.cw('{')
        self.cw('  if (strcmp(name,"'+a.group+'") || strcmp(name,"*")) {')

      i = i + 1
      vname = self.pname+'_farrays['+repr(i)+']'
      if a.dynamic:
        j = 0
        # --- create lines of the form dims[1] = high-low+1, in reverse order
        for d in a.dims:
          if d.high == '': continue
          self.cw('   '+vname+'.dimensions['+repr(len(a.dims)-1-j)+']=',
                  noreturn=1)
          j = j + 1
          if re.search('[a-zA-Z]',d.high) == None:
            self.cw('('+d.high+')-',noreturn=1)
          else:
            self.cw('('+self.prefixdimsc(d.high,sdict)+')-',noreturn=1)
          if re.search('[a-zA-Z]',d.low) == None:
            self.cw('('+d.low+')+1;')
          else:
            self.cw('('+self.prefixdimsc(d.low,sdict)+')+1;',noreturn=1)

    if currentgroup != '':
      self.cw('  }}')

    # --- Now write out the setdims routine which calls of the routines
    # --- for the individual groups.
    self.cw('void '+self.pname+'setdims(char *name,ForthonObject *obj)')
    self.cw('{')
    for group in dyngroups:
        self.cw('  '+self.pname+'setdims'+group+'(name);')
    self.cw('}')
  
    self.cw('')

    ###########################################################################
    # --- And finally, the initialization function
    self.cw('void init'+self.pname+'py()')
    self.cw('{')
    self.cw('  PyObject *m, *d, *s;')
    self.cw('  m = Py_InitModule("'+self.pname+'py",'+self.pname+'_methods);')
    self.cw('  d = PyModule_GetDict(m);')
    self.cw('  PyDict_SetItemString(d,"Py'+self.pname+'Type",'+
               '(PyObject *)&ForthonType);')
    self.cw('  '+self.pname+'Object=(ForthonObject *)'+
               'ForthonObject_New(NULL,NULL);')
    self.cw('  '+self.pname+'Object->name = "'+self.pname+'";')
    self.cw('  '+self.pname+'Object->typename = "'+self.pname+'";')
    self.cw('  '+self.pname+'Object->nscalars = '+self.pname+'nscalars;')
    self.cw('  '+self.pname+'Object->fscalars = '+self.pname+'_fscalars;')
    self.cw('  '+self.pname+'Object->narrays = '+self.pname+'narrays;')
    self.cw('  '+self.pname+'Object->farrays = '+self.pname+'_farrays;')
    self.cw('  '+self.pname+'Object->setdims = *'+self.pname+'setdims;')
    self.cw('  '+self.pname+'Object->setstaticdims = *'+
                self.pname+'setstaticdims;')
    self.cw('  '+self.pname+'Object->fmethods = '+self.pname+'_methods;')
    self.cw('  '+self.pname+'Object->fobj = NULL;')
    self.cw('  PyDict_SetItemString(d,"'+self.pname+'",(PyObject *)'+
                self.pname+'Object);')
    self.cw('  ErrorObject = PyString_FromString("'+self.pname+'py.error");')
    self.cw('  PyDict_SetItemString(d, "error", ErrorObject);')
    self.cw('  if (PyErr_Occurred())')
    self.cw('    Py_FatalError("can not initialize module '+self.pname+'");')
    self.cw('  import_array();')
    self.cw('  Forthon_BuildDicts('+self.pname+'Object);')
    self.cw('  ForthonPackage_allotdims('+self.pname+'Object);')
    self.cw('  '+fname(self.pname+'passpointers')+'();')
    self.cw('  ForthonPackage_staticarrays('+self.pname+'Object);')
    if not self.f90 and not self.f90f:
      self.cw('  '+fname(self.pname+'data')+'();')
    if self.initialgallot:
      self.cw('  s = Py_BuildValue("(s)","*");')
      self.cw('  ForthonPackage_gallot((PyObject *)'+self.pname+'Object,s);')
      self.cw('  Py_XDECREF(s);')

    self.cw('  {')
    self.cw('  PyObject *m, *d, *f, *r;')
    self.cw('  m = PyImport_ImportModule("Forthon");')
    self.cw('  if (m != NULL) {')
    self.cw('    d = PyModule_GetDict(m);')
    self.cw('    if (d != NULL) {')
    self.cw('      f = PyDict_GetItemString(d,"registerpackage");')
    self.cw('      if (f != NULL) {')
    self.cw('        r = PyObject_CallFunction(f,"Os",(PyObject *)'+
                self.pname+'Object,"'+self.pname+'");')
    self.cw('  }}}')
    self.cw('  Py_XDECREF(m);')
    self.cw('  Py_XDECREF(r);')
    self.cw('  }')

    if machine=='win32':
      self.cw('  /* Initialize FORTRAN on CYGWIN */')
      self.cw(' initPGfortran();')

    self.cw('}')
    self.cw('')

    ###########################################################################
    # --- Write set pointers routine which gets all of the fortran pointers
    self.cw('void '+fname(self.pname+'setscalarpointers')+'(int *i,char *p',noreturn=1)
    if machine=='J90':
      self.cw(',int *iflag)')
    else:
      self.cw(')')
    self.cw('{')
    self.cw('  /* Get pointers for the scalars */')
    self.cw('  '+self.pname+'_fscalars[*i].data = (char *)p;')
    if machine=='J90':
      self.cw('    if (iflag) {')
      self.cw('      '+self.pname+'_fscalars[*i].data=_fcdtocp((_fcd)p);}')
    self.cw('}')

    self.cw('void '+fname(self.pname+'setarraypointers')+'(int *i,char *p',noreturn=1)
    if machine=='J90':
      self.cw(',int *iflag)')
    else:
      self.cw(')')
    self.cw('{')
    self.cw('  /* Get pointers for the arrays */')
    self.cw('  '+self.pname+'_farrays[*i].data.s = (char *)p;')
    if machine=='J90':
      self.cw('    if (iflag) {')
      self.cw('      '+self.pname+'_farrays[*i].data.s=_fcdtocp((_fcd)p);}')
    self.cw('}')

    # --- This routine gets the dimensions from an array. It is called from
    # --- fortran and the last argument should be shape(array).
    # --- This is only used for routines with the fassign attribute.
    self.cw('void '+fname(self.pname+'setarraydims')+'(int *i,int *nd,int *dims)')
    self.cw('{')
    if self.f90:
      self.cw('  int id;')
      self.cw('  for (id=0;id<*nd;id++)')
      self.cw('    '+self.pname+'_farrays[*i].dimensions[id] = dims[id];')
    self.cw('}')

    ###########################################################################
    # --- --- Close the c package module file
    self.cfile.close()

    ###########################################################################
    ###########################################################################
    ###########################################################################
    # --- Write out fortran initialization routines
    if self.f90 or self.f90f:
      self.ffile = open(self.pname+'_p.F90','w')
    else:
      self.ffile = open(self.pname+'_p.m','w')
    self.ffile.close()

    ###########################################################################
    ###########################################################################
    # --- Process any derived types
    wrappergen_derivedtypes.ForthonDerivedType(typelist,self.pname,
                               self.pname+'pymodule.c',
                               self.pname+'_p.F90',self.f90,self.isz,
                               self.writemodules)
    ###########################################################################
    ###########################################################################

    if self.f90 or self.f90f:
      self.ffile = open(self.pname+'_p.F90','a')
    else:
      self.ffile = open(self.pname+'_p.m','a')

    ###########################################################################
    # --- Write out f90 modules, including any data statements
    if (self.f90 or self.f90f) and self.writemodules:
      if   self.f90 : dyntype = 'pointer'
      elif self.f90f: dyntype = 'allocatable,target'
      for g in groups+hidden_groups:
        self.fw('MODULE '+g)
        # --- Check if any variables are derived types. If so, the module
        # --- containing the type must be used.
        printedtypes = []
        for v in slist + alist:
          if v.group == g and v.derivedtype:
            if v.type not in printedtypes:
              self.fw('  USE '+v.type+'module')
              printedtypes.append(v.type)
        self.fw('  SAVE')
        # --- Declerations for scalars and arrays
        for s in slist:
          if s.group == g:
            self.fw('  '+fvars.ftof(s.type),noreturn=1)
            if s.dynamic: self.fw(',POINTER',noreturn=1)
            self.fw('::'+s.name,noreturn=1)
            if s.data: self.fw('='+s.data[1:-1],noreturn=1)
            self.fw('')
        for a in alist:
          if a.group == g:
            if a.dynamic:
              if a.type == 'character':
                self.fw('  character(len='+a.dims[0].high+'),'+dyntype+'::'+
                        a.name,noreturn=1)
                ndims = len(a.dims) - 1
              else:
                self.fw('  '+fvars.ftof(a.type)+','+dyntype+'::'+a.name,
                        noreturn=1)
                ndims = len(a.dims)
              if ndims > 0:
                self.fw('('+(ndims*':,')[:-1]+')',noreturn=1)
              self.fw('')
            else:
              if a.type == 'character':
                self.fw('  character(len='+a.dims[0].high+')::'+a.name+
                        a.dimstring)
              else:
                self.fw('  '+fvars.ftof(a.type)+'::'+a.name+a.dimstring)
              if a.data:
                # --- Add line continuation marks if the data line extends over
                # --- multiple lines.
                dd = re.sub(r'\n','&\n',a.data)
                self.fw('  data '+a.name+dd)
        self.fw('END MODULE '+g)

    ###########################################################################
    self.fw('SUBROUTINE '+self.pname+'passpointers()')

    # --- Write out the Use statements
    for g in groups+hidden_groups:
      if self.f90 or self.f90f:
       self.fw('  USE '+g)
      else:
       self.fw('  Use('+g+')')
 
    # --- Write out calls to c routine passing down pointers to scalars
    for i in range(len(slist)):
      s = slist[i]
      if not s.derivedtype:
        self.fw('  call '+self.pname+'setscalarpointers('+repr(i)+','+s.name,
                noreturn=1)
        if machine == 'J90':
          if s.type == 'string' or s.type == 'character':
            self.fw(',1)')
          else:
            self.fw(',0)')
        else:
          self.fw(')')
      elif not s.dynamic:
        self.fw('  call init'+s.type+'py('+repr(i)+','+s.name+','+
                s.name+'%cobj__)')

    # --- Write out calls to c routine passing down pointers to arrays
    # --- For f90, setpointers is not needed for dynamic arrays but is called
    # --- anyway to get the numbering of arrays correct.
    if machine == 'J90':
      if a.type == 'string' or a.type == 'character':
        str = ',1)'
      else:
        str = ',0)'
    else:
      str = ')'
    for i in range(len(alist)):
      a = alist[i]
      if a.dynamic:
        if not self.f90 and not self.f90f:
          self.fw('  call '+self.pname+'setarraypointers('+repr(i)+','+
                  'p'+a.name+str)
      else:
        self.fw('  call '+self.pname+'setarraypointers('+repr(i)+','+a.name+str)

    # --- Finish the routine
    self.fw('  return')
    self.fw('end')

    ###########################################################################
    # --- Write routine for each dynamic variable which gets the pointer from the
    # --- wrapper
    if self.f90:
      for s in slist:
        if s.dynamic:
          self.fw('SUBROUTINE '+self.pname+'setpointer'+s.name+'(p__,cobj__)')
          self.fw('  USE '+s.group)
          self.fw('  integer('+self.isz+'):: cobj__')
          self.fw('  '+fvars.ftof(s.type)+',target::p__')
          self.fw('  '+s.name+' => p__')
          self.fw('  RETURN')
          self.fw('END')
          self.fw('SUBROUTINE '+self.pname+'getpointer'+s.name+'(cobj__,obj__)')
          self.fw('  USE '+s.group)
          self.fw('  integer('+self.isz+'):: cobj__,obj__')
          self.fw('  if (ASSOCIATED('+s.name+')) then')
          self.fw('    cobj__ = '+s.name+'%cobj__')
          self.fw('  else')
          self.fw('    cobj__ = 0')
          self.fw('  endif')
          self.fw('  RETURN')
          self.fw('END')

      for a in alist:
        if a.dynamic:
          self.fw('SUBROUTINE '+self.pname+'setpointer'+a.name+'(p__,cobj__,dims__)')
          groups = self.dimsgroups(a.dimstring,sdict,slist)
          groupsprinted = [a.group]
          for g in groups:
            if g not in groupsprinted:
              self.fw('  USE '+g)
              groupsprinted.append(g)
          self.fw('  USE '+a.group)
          self.fw('  integer('+self.isz+'):: cobj__')
          self.fw('  integer('+self.isz+'):: dims__('+repr(len(a.dims))+')')
          self.fw('  '+fvars.ftof(a.type)+',target::p__'+
                    self.prefixdimsf(a.dimstring))
          self.fw('  '+a.name+' => p__')
          self.fw('  return')
          self.fw('end')
          if re.search('fassign',a.attr):
            self.fw('SUBROUTINE '+self.pname+'getpointer'+a.name+'(i__,obj__)')
            self.fw('  USE '+a.group)
            self.fw('  integer('+self.isz+'):: i__,obj__')
            self.fw('  call '+self.pname+'setarraypointers(i__,'+a.name+')')
            self.fw('  call '+self.pname+'setarraydims(i__,'+
                       repr(len(a.dims))+',shape('+a.name+'))')
            self.fw('  return')
            self.fw('end')

    if self.f90f:
      for a in alist:
        if a.dynamic:
          self.fw('SUBROUTINE '+self.pname+'setpointer'+a.name+'(p__,cobj__,dims__)')
          groups = self.dimsgroups(a.dimstring,sdict,slist)
          groupsprinted = [a.group]
          for g in groups:
            if g not in groupsprinted:
              self.fw('  USE '+g)
              groupsprinted.append(g)
          self.fw('  USE '+a.group)
          self.fw('  integer('+self.isz+'):: cobj__')
          self.fw('  integer('+self.isz+'):: dims__('+repr(len(a.dims))+')')
          self.fw('  integer(kind=8)::p__)')
          self.fw('  allocate('+a.name+'('+self.prefixdimsf(a.dimstring)+'))')
          self.fw('  fortranarrayspointerassignment(p__,' + a.name+')')
          self.fw('  return')
          self.fw('end')

    ###########################################################################
    if not self.f90 and not self.f90f:
      # --- Write out fortran data routine, only for f77 version
      self.fw('      SUBROUTINE '+self.pname+'data()')

      # --- Write out the Use statements
      for g in groups:
        self.fw('Use('+g+')')
     
      for hg in hidden_groups:
        self.fw('Use('+hg+')')

      self.fw('      integer iyiyiy')

      # --- Write out data statements
      for s in slist:
        if s.data:
          self.fw('      data '+s.name+s.data)
      for a in alist:
        if a.data and not a.dynamic:
          self.fw('      data '+a.name+a.data)

      self.fw('      iyiyiy=0')
      self.fw('      return')
      self.fw('      end')

    # --- --- Close fortran file
    self.ffile.close()
    scalar_pickle_file = open(self.pname + '.scalars','w')
    sdict ['_module_name_'] = self.pname
    pickle.dump (sdict, scalar_pickle_file)
    scalar_pickle_file.close()

###############################################################################
###############################################################################
###############################################################################
###############################################################################

module_prefix_pat = re.compile ('([a-zA-Z_]+)\.scalars')
def get_another_scalar_dict(file_name):
  m = module_prefix_pat.search (file_name)
  if m.start() == -1: raise 'expection a .scalars file'
  f = open (file_name, 'r')
  other_scalar_dicts.append (pickle.load(f))
  f.close()

def wrappergenerator_main(argv=None):
  if argv is None: argv = sys.argv[1:]
  optlist,args=getopt.getopt(argv,'at:d:',
                     ['f90','f90f','2underscores','nowritemodules','macros='])

  # --- Get package name from argument list
  try:
    pname = args[0][:re.search('\.',args[0]).start()]
  except IndexError:
    print PyMAC.__doc__
    sys.exit(1)

  # --- get other command line options and default actions
  initialgallot = 0
  f90 = 0
  f90f = 0
  writemodules = 1
  othermacros = []

  # --- a list of scalar dictionaries from other modules.
  other_scalar_dicts = []

  for o in optlist:
    if o[0]=='-a':
      initialgallot = 1
    elif o[0]=='-t':
      machine = o[1]
    elif o[0]=='--f90':
      f90 = 1
    elif o[0]=='--f90f':
      f90f = 1
    elif o[0]=='-d':
      get_another_scalar_dict (o[1])
    elif o[0]=='--nowritemodules':
      writemodules = 0
    elif o[0]=='--macros':
      othermacros.append(o[1])

  cc = PyMAC(pname,f90,f90f,initialgallot,writemodules,
             othermacros,other_scalar_dicts)

if __name__ == '__main__':
  wrappergenerator_main(sys.argv[1:])

