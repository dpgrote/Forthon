#!/usr/bin/env python
# Python wrapper generation
# Created by David P. Grote, March 6, 1998
# Modified by T. B. Yang, May 21, 1998
# $Id: wrappergenerator.py,v 1.70 2011/08/22 17:45:58 grote Exp $

import sys
import os.path
from interfaceparser import processfile
import string
import re
import fvars
import pickle
from Forthon_options import options,args
from cfinterface import *
import wrappergen_derivedtypes
if sys.hexversion >= 0x20501f0:
    import hashlib
else:
    # --- hashlib was not available in python earlier than 2.5.
    import md5 as hashlib

class PyWrap:
    """
    Usage:
      -a       All groups will be allocated on initialization
      -t ARCH  Build for specified architecture (default is HP700)
      -d <.scalars file>  a .scalars file in another module that this module depends on
      -F <compiler> The fortran compiler being used. This is needed since some
                    operations depend on the compiler specific oddities.
      --nowritemodules The modules will not be written out, assuming
                       that they are already written.
      --macros pkg.v Other interface files that are needed for the definition
                     of macros.
      --timeroutines Calls to the routines from python will be timed
      file1    Main variable description file for the package
      [file2, ...] Subsidiary variable description files
    """

    def __init__(self,ifile,pname,initialgallot=1,writemodules=1,
                 otherinterfacefiles=[],other_scalar_vars=[],timeroutines=0,
                 otherfortranfiles=[],fcompname=None):
        self.ifile = ifile
        self.pname = pname
        self.initialgallot = initialgallot
        self.writemodules = writemodules
        self.timeroutines = timeroutines
        self.otherinterfacefiles = otherinterfacefiles
        self.other_scalar_vars = other_scalar_vars
        self.otherfortranfiles = otherfortranfiles
        self.fcompname = fcompname
        self.isz = isz # isz defined in cfinterface

        self.processvariabledescriptionfile()

    def cname(self,n):
        # --- Standard name of the C interface to a Fortran routine
        # --- pkg_varname
        return self.pname+'_'+n

    transtable = (10*string.ascii_lowercase)[:256]
    def fsub(self,prefix,suffix=''):
        """
        The fortran standard limits routine names to 31 characters. If the
        routine name is longer than that, this routine takes the first 15
        characters and and creates a hashed string based on the full name to get
        the next 16. This does not guarantee uniqueness, but the nonuniqueness
        should be minute.
        """
        name = self.pname+prefix+suffix
        if len(name) < 32: return name
        transtable = PyWrap.transtable
        if sys.hexversion >= 0x03000000:
            hashbytes = hashlib.md5(name.encode()).digest()
            hash = ''.join([transtable[d] for d in hashbytes])
        else:
            hash = hashlib.md5(name).digest().translate(transtable)
        return name[:15] + hash

    def dimisparameter(self,dim):
        # --- Convert fortran variable name into reference from list of variables
        # --- and check if it is a parameter.
        sl=re.split('[ ()/\*\+\-]',dim)
        for ss in sl:
            if re.search('[a-zA-Z]',ss) != None:
                try:
                    v = self.slist[self.sdict[ss]]
                    if v.parameter:
                        return True
                except KeyError:
                    pass
        return False

    def prefixdimsc(self,dim):
        # --- Convert fortran variable name into reference from list of variables.
        sl=re.split('[ ()/\*\+\-]',dim)
        for ss in sl:
            if re.search('[a-zA-Z]',ss) != None:
                if ss in self.sdict:
                    dim = re.sub(ss,
                             '*(long *)'+self.pname+'_fscalars['+repr(self.sdict[ss])+'].data',
                             dim,count=1)
                else:
                    for other_vars in self.other_scalar_vars:
                        other_dict = other_vars[0]
                        if ss in other_dict:
                            dim = re.sub(ss,'*(long *)'+other_dict['_module_name_']+
                                      '_fscalars['+repr(other_dict[ss])+'].data',dim,count=1)
                            break
                    else:
                        raise SyntaxError(ss + ' is not declared in the interface file')
        return dim.lower()

    # --- Convert dimensions for unspecified arrays
    def prefixdimsf(self,dim):
        # --- Check for any unspecified dimensions and replace it with an element
        # --- from the dims array.
        sl = re.split(',',dim[1:-1])
        for i in range(len(sl)):
            if sl[i] == ':': sl[i] = 'dims__(%d)'%(i+1)
        dim = '(' + ','.join(sl) + ')'
        return dim.lower()

    def dimsgroups(self,dim):
        # --- Returns a list of group names that contain the variables listed in
        # --- a dimension statement
        groups = []
        sl=re.split('[ (),:/\*\+\-]',dim)
        for ss in sl:
            if re.search('[a-zA-Z]',ss) != None:
                if ss in self.sdict:
                    groups.append(self.slist[self.sdict[ss]].group)
                else:
                    for other_vars in self.other_scalar_vars:
                        other_dict = other_vars[0]
                        other_list = other_vars[1]
                        if ss in other_dict:
                            groups.append(other_list[other_dict[ss]].group)
                            break
                    else:
                        raise SyntaxError(ss + ' is not declared in the interface file')
        return groups

    def cw(self,text,noreturn=0):
        if noreturn:
            self.cfile.write(text)
        else:
            self.cfile.write(text+'\n')
    def fw(self,text,noreturn=0):
        i = 0
        while len(text[i:]) > 132 and text[i:].find('&') == -1:
            # --- If the line is too long, then break it up, adding line
            # --- continuation marks in between any variable names.
            # --- This is the same as \W, but also skips %, since PG compilers
            # --- don't seem to like a line continuation mark just before a %.
            ss = re.search('[^a-zA-Z0-9_%]',text[i+130::-1])
            assert ss is not None,\
                   "Forthon can't find a place to break up this line:\n"+text
            text = text[:i+130-ss.start()] + '&\n' + text[i+130-ss.start():]
            i += 130 - ss.start() + 1
        if noreturn:
            self.ffile.write(text)
        else:
            self.ffile.write(text+'\n')

    def setffile(self):
        """
        Set the ffile attribute, which is the fortran file object.
        It the attribute hasn't been created, then open the file with write status.
        If it has, and the file is closed, then open it with append status.
        """
        if 'ffile' in self.__dict__: status = 'a'
        else:                        status = 'w'
        if status == 'w' or (status == 'a' and self.ffile.closed):
            self.ffile = open(self.pname+'_p.F90',status)

    def processvariabledescriptionfile(self):
        """
        Read in and parse the variable description file and create the lists
        of scalars and arrays.
        """

        # --- Get the list of variables and subroutine from the var file
        vlist,hidden_vlist,typelist = processfile(self.pname,self.ifile,
                                                  self.otherinterfacefiles,
                                                  self.timeroutines)

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

        self.typelist = typelist
        self.slist = slist
        self.sdict = sdict
        self.alist = alist
        self.flist = flist
        self.groups = groups
        self.hidden_groups = hidden_groups

    ###########################################################################
    def createmodulefile(self):
        # --- This is the routine that does all of the work

        # --- Create the module file
        self.cfile = open(self.pname+'pymodule.c','w')
        self.cw('#include "Forthon.h"')
        self.cw('#include <setjmp.h>')
        self.cw('ForthonObject *'+self.pname+'Object;')

        # --- See the kaboom command in Forthon.c for information on these two
        # --- variables.
        self.cw('extern jmp_buf stackenvironment;')
        self.cw('extern int lstackenvironmentset;')

        # --- Print out the external commands
        self.cw('extern void '+fname(self.fsub('passpointers'))+'(void);')
        self.cw('extern void '+fname(self.fsub('nullifypointers'))+'(void);')

        # --- fortran routine prototypes
        for f in self.flist:
            # --- Functions
            self.cw('extern '+fvars.ftoc(f.type)+' '+fname(f.name)+'(',noreturn=1)
            i = 0
            istr = 0
            if len(f.args) == 0: self.cw('void',noreturn=1)
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
        for t in self.typelist:
            self.cw('extern PyObject *'+self.cname(t.name)+'New(PyObject *self, PyObject *args);')
        self.cw('')

        # --- setpointer and getpointer routines
        # --- Note that setpointer gets written out for all derived types -
        # --- for non-dynamic derived types, the setpointer routine does a copy.
        # --- The double underscores in the argument names are to avoid name
        # --- collisions with package variables.
        for s in self.slist:
            if (s.dynamic or s.derivedtype) and not s.parameter:
                self.cw('extern void '+fname(self.fsub('setscalarpointer',s.name))+
                        '(char *p__,char *fobj__,npy_intp *nullit__);')
            if s.dynamic:
                self.cw('extern void '+fname(self.fsub('getscalarpointer',s.name))+
                        '(ForthonObject **cobj__,char *fobj__,int *createnew__);')
        for a in self.alist:
            self.cw('extern void '+fname(self.fsub('setarraypointer',a.name))+
                    '(char *p__,char *fobj__,npy_intp *dims__);')
            if re.search('fassign',a.attr):
                self.cw('extern void '+fname(self.fsub('getarraypointer',a.name))+
                        '(Fortranarray *farray__,char *fobj__);')
        self.cw('')

        # --- setaction and getaction routines
        for s in self.slist:
            if s.setaction is not None:
                self.cw('extern void '+fname(self.fsub('setaction',s.name))+
                        '('+fvars.ftoc_dict[s.type]+' *v);')
            if s.getaction is not None:
                self.cw('extern void '+fname(self.fsub('getaction',s.name))+'(void);')
        for a in self.alist:
            if a.setaction is not None:
                self.cw('extern void '+fname(self.fsub('setaction',a.name))+
                        '('+fvars.ftoc_dict[a.type]+' *v);')
            if a.getaction is not None:
                self.cw('extern void '+fname(self.fsub('getaction',a.name))+'(void);')

        ###########################################################################
        # --- Write declarations of c pointers to fortran variables

        # --- Declare scalars from other modules
        for other_vars in self.other_scalar_vars:
            other_dict = other_vars[0]
            self.cw('extern Fortranscalar '+other_dict['_module_name_']+
                    '_fscalars[];')

        # --- Note that the pointers to the subroutines set and getpointer and
        # --- set and get action are at first set to NULL. The data is then setup
        # --- in a seperate call to the declarevars. This is done since it is not
        # --- standard C to have function pointers in structure initializers.

        # --- Scalars
        self.cw('int '+self.pname+'nscalars = '+repr(len(self.slist))+';')
        if len(self.slist) > 0:
            self.cw('Fortranscalar '+self.pname+'_fscalars['+repr(len(self.slist))+']={')
            for i in range(len(self.slist)):
                s = self.slist[i]
                self.cw('{NPY_%s,'%fvars.ftop(s.type) +
                         '"%s",'%s.type +
                         '"%s",'%s.name +
                         'NULL,' +
                         '"%s",'%s.group +
                         '"%s",'%s.attr +
                         '"%s",'%repr(s.comment)[1:-1].replace('"','\\"') +
                         '%i,'%s.dynamic +
                         '%i,'%s.parameter +
                         'NULL,' + # setscalarpointer
                         'NULL,' + # getscalarpointer
                         'NULL,' + # setaction
                         'NULL' + # getaction
                         '}',noreturn=1)
                if i < len(self.slist)-1: self.cw(',')
            self.cw('};')
        else:
            self.cw('Fortranscalar *'+self.pname+'_fscalars=NULL;')

        # --- Arrays
        self.cw('int '+self.pname+'narrays = '+repr(len(self.alist))+';')
        if len(self.alist) > 0:
            self.cw('static Fortranarray '+
                    self.pname+'_farrays['+repr(len(self.alist))+']={')
            for i in range(len(self.alist)):
                a = self.alist[i]
                if a.data and a.dynamic:
                    initvalue = a.data[1:-1]
                else:
                    initvalue = '0'
                self.cw('{NPY_%s,'%fvars.ftop(a.type) +
                          '%d,'%a.dynamic +
                          '%d,'%len(a.dims) +
                          'NULL,' +
                          '"%s",'%a.name +
                          '{NULL},' +
                          'NULL,' + # setarraypointer
                          'NULL,' + # getarraypointer
                          'NULL,' + # setaction
                          'NULL,' + # getaction
                          '%s,'%initvalue +
                          'NULL,' +
                          '"%s",'%a.group +
                          '"%s",'%a.attr +
                          '"%s",'%repr(a.comment)[1:-1].replace('"','\\"') +
                          '"%s"}'%repr(a.dimstring)[1:-1],noreturn=1)
                if i < len(self.alist)-1: self.cw(',')
            self.cw('};')
        else:
            self.cw('static Fortranarray *'+self.pname+'_farrays=NULL;')

        #########################################################################
        # --- Write declarations of c pointers to fortran variables
        self.cw('void '+self.pname+'declarevars(ForthonObject *obj) {')

        # --- Scalars
        for i in range(len(self.slist)):
            s = self.slist[i]
            if (s.dynamic or s.derivedtype) and not s.parameter:
                setscalarpointer = '*'+fname(self.fsub('setscalarpointer',s.name))
                self.cw('obj->fscalars[%d].setscalarpointer = %s;'%(i,setscalarpointer))
                if s.dynamic:
                    getscalarpointer = '*'+fname(self.fsub('getscalarpointer',s.name))
                    self.cw('obj->fscalars[%d].getscalarpointer = %s;'%(i,getscalarpointer))
            if s.setaction is not None:
                setaction = '*'+fname(self.fsub('setaction',s.name))
                self.cw('obj->fscalars[%d].setaction = %s;'%(i,setaction))
            if s.getaction is not None:
                getaction = '*'+fname(self.fsub('getaction',s.name))
                self.cw('obj->fscalars[%d].getaction = %s;'%(i,getaction))

        # --- Arrays
        for i in range(len(self.alist)):
            a = self.alist[i]
            if a.dynamic:
                setarraypointer = '*'+fname(self.fsub('setarraypointer',a.name))
                self.cw('obj->farrays[%d].setarraypointer = %s;'%(i,setarraypointer))
            if re.search('fassign',a.attr):
                getarraypointer = '*'+fname(self.fsub('getarraypointer',a.name))
                self.cw('obj->farrays[%d].getarraypointer = %s;'%(i,getarraypointer))
            if a.setaction is not None:
                setaction = '*'+fname(self.fsub('setaction',a.name))
                self.cw('obj->farrays[%d].setaction = %s;'%(i,setaction))
            if a.getaction is not None:
                getaction = '*'+fname(self.fsub('getaction',a.name))
                self.cw('obj->farrays[%d].getaction = %s;'%(i,getaction))

        self.cw('}')

# Some extra work is needed to get the getset attribute access scheme working.
#   # --- Write out the table of getset routines
#   self.cw('')
#   self.cw('static PyGetSetDef '+self.pname+'_getseters[] = {')
#   for i in range(len(self.slist)):
#     s = self.slist[i]
#     if s.type == 'real': gstype = 'double'
#     elif s.type == 'double': gstype = 'double'
#     elif s.type == 'float': gstype = 'float'
#     elif s.type == 'integer': gstype = 'integer'
#     elif s.type == 'complex': gstype = 'cdouble'
#     else:                    gstype = 'derivedtype'
#     self.cw('{"'+s.name+'",(getter)Forthon_getscalar'+gstype+
#                          ',(setter)Forthon_setscalar'+gstype+
#                    ',"%s"'%repr(s.comment)[1:-1].replace('"','\\"') +
#                         ',(void *)'+repr(i)+'},')
#   for i in range(len(self.alist)):
#     a = self.alist[i]
#     self.cw('{"'+a.name+'",(getter)Forthon_getarray'+
#                          ',(setter)Forthon_setarray'+
#                    ',"%s"'%repr(a.comment)[1:-1].replace('"','\\"') +
#                         ',(void *)'+repr(i)+'},')
#   self.cw('{"scalardict",(getter)Forthon_getscalardict,'+
#                         '(setter)Forthon_setscalardict,'+
#           '"internal scalar dictionary",NULL},')
#   self.cw('{"arraydict",(getter)Forthon_getarraydict,'+
#                        '(setter)Forthon_setarraydict,'+
#           '"internal array dictionary",NULL},')
#   self.cw('{NULL}};')

        ###########################################################################
        if self.timeroutines:
            # --- This is written out here instead of just being in Forthon.h
            # --- so that when it is not used, the compiler doesn't complain
            # --- about cputime being unused.
            self.cw('#include <sys/times.h>')
            self.cw('#include <unistd.h>')
            self.cw('static double cputime(void)')
            self.cw('{')
            self.cw('  struct tms usage;')
            self.cw('  long hardware_ticks_per_second;')
            self.cw('  (void) times(&usage);')
            self.cw('  hardware_ticks_per_second = sysconf(_SC_CLK_TCK);')
            self.cw('  return (double) usage.tms_utime/hardware_ticks_per_second;')
            self.cw('}')

        ###########################################################################
        ###########################################################################
        # --- Now, the fun part, writing out the wrapper for the subroutine and
        # --- function calls.
        for f in self.flist:
            # --- Write out the documentation first.
            docstring = ('static char doc_'+self.cname(f.name)+'[] = "'+f.name+
                         f.dimstring+'\n'+f.comment+'";')
            # --- Replaces newlines with '\\n' so that the string is all on one line
            # --- in the C coding. Using repr does the same thing, but more easily.
            # --- The [1:-1] strips off the single quotes that repr puts there.
#      docstring = re.sub(r'\
#','\\\\n',docstring)
            self.cw(repr(docstring)[1:-1])
            # --- Now write out the wrapper
            self.cw('static PyObject *')
            self.cw(self.cname(f.name)+'(PyObject *self, PyObject *args)')
            self.cw('{')

            # --- With arguments, it gets very messy
            lv = repr(len(f.args))
            if len(f.args) > 0:
                self.cw('  PyObject * pyobj['+lv+'];')
                self.cw('  PyArrayObject * ax['+lv+'];')
                self.cw('  int i;')
                self.cw('  char e[256];')

            if self.timeroutines:
                # --- Setup for the timer, getting time routine started.
                self.cw('  double time1,time2;')
                self.cw('  time1 = cputime();')

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

            # --- Set all of the ax's to NULL
            if len(f.args) > 0:
                self.cw('  for (i=0;i<'+repr(len(f.args))+';i++) ax[i] = NULL;')

            # --- Parse incoming arguments into a list of PyObjects
            self.cw('  if (!PyArg_ParseTuple(args, "'+'O'*len(f.args)+'"',noreturn=1)
            for i in range(len(f.args)):
                self.cw(',&pyobj['+repr(i)+']',noreturn=1)
            self.cw(')) return NULL;')

            # --- Loop over arguments, extracting the data addresses.
            # --- Convert all arguments into arrays. This allows complete flexibility
            # --- in what can be passed to fortran functions.
            istr = 0
            for i in range(len(f.args)):
                if not fvars.isderivedtype(f.args[i]):
                    self.cw('  if (!Forthon_checksubroutineargtype(pyobj['+repr(i)+'],'+
                        'NPY_'+fvars.ftop(f.args[i].type)+')) {')
                    self.cw('    sprintf(e,"Argument '+f.args[i].name+ ' in '+f.name+
                                         ' has the wrong type");')
                    self.cw('    PyErr_SetString(ErrorObject,e);')
                    self.cw('    goto err;}')
                    if f.function == 'fsub':
                        self.cw('  ax['+repr(i)+'] = FARRAY_FROMOBJECT('+
                              'pyobj['+repr(i)+'], NPY_'+fvars.ftop(f.args[i].type)+');')
                    elif f.function == 'csub':
                        self.cw('  ax['+repr(i)+']=(PyArrayObject *)PyArray_ContiguousFromObject('+
                              'pyobj['+repr(i)+'], NPY_'+fvars.ftop(f.args[i].type)+',0,0);')
                    self.cw('  if (ax['+repr(i)+'] == NULL) {')
                    self.cw('    sprintf(e,"There is an error in argument '+f.args[i].name+
                                         ' in '+f.name+'");')
                    self.cw('    PyErr_SetString(ErrorObject,e);')
                    self.cw('    goto err;}')
                    if f.args[i].type == 'string' or f.args[i].type == 'character':
                        self.cw(' FSETSTRING(fstr[%d],PyArray_BYTES(ax[%d]),PyArray_ITEMSIZE(ax[%d]));'
                                %(istr,i,i))
                        istr = istr + 1
                else:
                    self.cw('  {')
                    self.cw('  PyObject *t;')
                    self.cw('  t = PyObject_Type(pyobj['+repr(i)+']);')
                    self.cw('  if (strcmp(((PyTypeObject *)t)->tp_name,"Forthon") != 0) {')
                    self.cw('    sprintf(e,"Argument '+f.args[i].name+ ' in '+f.name+
                                        ' has the wrong type");')
                    self.cw('    PyErr_SetString(ErrorObject,e);')
                    self.cw('    goto err;}')
                    self.cw('  Py_DECREF(t);')
                    typename = '((ForthonObject *)pyobj['+repr(i)+'])->typename'
                    self.cw('  if (strcmp('+typename+',"'+f.args[i].type+'") != 0) {')
                    self.cw('    sprintf(e,"Argument '+f.args[i].name+ ' in '+f.name+
                                        ' has the wrong type");')
                    self.cw('    PyErr_SetString(ErrorObject,e);')
                    self.cw('    goto err;}')
                    self.cw('  }')

            # --- Write the code checking dimensions of arrays
            # --- This must be done after all of the ax's are setup in case the
            # --- dimensioning arguments come after the array argument.
            # --- This creates a local variable with the same name as the argument
            # --- and gives it the value passed in. Then any expressions in the
            # --- dimensions statements will be explicitly evaluated in C.
            if len(f.dimvars) > 0:
                self.cw('  {')
                self.cw('  long _n;')
                # --- Declare the dimension variables.
                for var,i in f.dimvars:
                    self.cw('  '+fvars.ftoc(var.type)+' '+var.name+'=*'+
                            '('+fvars.ftoc(var.type)+' *)(PyArray_BYTES(ax['+repr(i)+']));')
                # --- Loop over the arguments, looking for dimensioned arrays
                i = -1
                for arg in f.args:
                    i += 1
                    if len(arg.dims) > 0:
                        # --- Check the rank of the input argument
                        # --- For a 1-D argument, allow a scaler to be passed, which has
                        # --- a number of dimensions (nd) == 0.
                        self.cw('  if (!(PyArray_NDIM(ax[%d]) == %d'%(i,len(arg.dims)),noreturn=1)
                        if len(arg.dims) == 1:
                            self.cw('      ||PyArray_NDIM(ax[%d]) == 0)) {'%i)
                        else:
                            self.cw('      )) {')
                        self.cw('    sprintf(e,"Argument %s in %s '%(arg.name,f.name) +
                                         'has the wrong number of dimensions");')
                        self.cw('    PyErr_SetString(ErrorObject,e);')
                        self.cw('    goto err;}')

                        # --- Skip the check of dimension sizes if the total size of
                        # --- the array should be zero. This gets around an issue with
                        # --- numpy that zero length arrays seem to be always in
                        # --- C ordering.
                        self.cw('  if (1',noreturn=1)
                        for dim in arg.dims:
                            self.cw('*(('+dim.high+')-('+dim.low+')+1)',noreturn=1)
                        self.cw(' != 0) {')

                        j = -1
                        for dim in arg.dims:
                            j += 1
                            # --- Compare each dimension with its specified value
                            # --- For a 1-D argument, allow a scalar to be passed, which has
                            # --- a number of dimensions (nd) == 0, but only if the
                            # --- argument needs to have a length of 0 or 1.
                            self.cw('    _n = ('+dim.high+')-('+dim.low+')+1;')
                            if len(arg.dims) == 1:
                                self.cw('    if (!((_n==0||_n==1)||(PyArray_NDIM(ax[%d]) > 0 &&'%i,
                                        noreturn=1)
                            else:
                                self.cw('    if (!((',noreturn=1)
                            self.cw('_n == (long)(PyArray_DIMS(ax[%d])[%d])))) {'%(i,j))
                            self.cw('      sprintf(e,"Dimension '+repr(j+1)+' of argument '+
                                             arg.name + ' in '+f.name+
                                             ' has the wrong size");')
                            self.cw('      PyErr_SetString(ErrorObject,e);')
                            self.cw('      goto err;}')
                        self.cw('  }')
                self.cw('  }')

            # --- If the stackenvironment has not already been set, then make a call
            # --- to setjmp to save the state in case an error happens.
            # --- If there was an error, setjmp returns 1, so exit out.
            # --- If this routine is called from a python routine that was called
            # --- from another fortran routine, then the stackenvironment will
            # --- already have been setup (from the calling fortran routine) so don't
            # --- reset it.
            self.cw('  if (!(lstackenvironmentset++) && setjmp(stackenvironment)) goto err;')

            # --- Write the actual call to the fortran routine.
            if f.type == 'void':
                self.cw('  ')
            else:
                self.cw('  r = ')
            self.cw(fname(f.name)+'(',noreturn=1)
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
                    self.cw('('+fvars.ftoc(a.type)+' *)(PyArray_BYTES(ax['+repr(i)+']))',noreturn=1)
                i = i + 1
            if charlen_at_end:
                i = 0
                istr = 0
                for a in f.args:
                    if a.type == 'string' or a.type == 'character':
                        self.cw(',(int)PyArray_ITEMSIZE(ax['+repr(i)+'])',noreturn=1)
                        istr = istr + 1
                    i = i + 1

            self.cw(');') # --- Closing parenthesis on the call list

            # --- Decrement the counter. This will reach zero when the top of the
            # --- fortran call chain is reached and is about to return to the top
            # --- level python.
            self.cw('  lstackenvironmentset--;')

            # --- Copy the data that was sent to the routine back into the passed
            # --- in object if it is an PyArray.
            # --- Decrement reference counts of array objects created.
            # --- This is now handled by a separate subroutine included in Forthon.h
            if len(f.args) > 0:
                self.cw('  Forthon_restoresubroutineargs('+repr(len(f.args))+
                           ',pyobj,ax);')

            if self.timeroutines:
                # --- Now get ending time and add to timer variable
                self.cw('  time2 = cputime();')
                self.cw('  *(double *)'+self.pname+'_fscalars['+
                             repr(self.sdict[f.name+'runtime'])+'].data += (time2-time1);')

            # --- Write return sequence
            if f.type == 'void':
                self.cw('  returnnone;')
            else:
                self.cw('  ret_val = Py_BuildValue ("'+fvars.fto1[f.type]+'", r);')
                self.cw('  return ret_val;')

            # --- Error section, in case there was an error above or in the
            # --- fortran call
            self.cw('err:')

            if len(f.args) > 0:
                # --- Decrement reference counts of array objects created.
                self.cw('  for (i=0;i<'+repr(len(f.args))+';i++)')
                self.cw('    if (ax[i] != NULL) {Py_XDECREF(ax[i]);}')

            self.cw('  return NULL;')

            self.cw('}')

        # --- Add blank line
        self.cw('')

        ###########################################################################
        # --- Write out method list
        self.cw('static struct PyMethodDef '+self.pname+'_methods[] = {')
        for f in self.flist:
            if f.function:
                self.cw('{"'+f.name+'",(PyCFunction)'+self.cname(f.name)+',1,'+
                        'doc_'+self.cname(f.name)+'},')
        for t in self.typelist:
            self.cw('{"'+t.name+'",(PyCFunction)'+self.cname(t.name)+'New,1,'+
                    '"Creates a new instance of fortran derived type '+t.name+'"},')
        self.cw('{NULL,NULL}};')
        self.cw('')

        ###########################################################################
        # --- Write static array initialization routines
        self.cw('void '+self.pname+'setstaticdims(ForthonObject *self)')
        self.cw('{')

        i = -1
        for a in self.alist:
            i = i + 1
            vname = self.pname+'_farrays['+repr(i)+']'
            if a.dims and not a.dynamic:
                j = 0
                for d in a.dims:
                    if d.high == '': continue
                    self.cw('   '+vname+'.dimensions['+repr(j)+']=(npy_intp)((int)',
                            noreturn=1)
                    j = j + 1
                    if re.search('[a-zA-Z]',d.high) == None:
                        self.cw('('+d.high+')-',noreturn=1)
                    else:
                        if not self.dimisparameter(d.high):
                            raise SyntaxError('%s: static dims must be constants or parameters'%a.name)
                        self.cw('('+self.prefixdimsc(d.high)+')-',noreturn=1)
                    if re.search('[a-zA-Z]',d.low) == None:
                        self.cw('('+d.low+')+1);')
                    else:
                        if not self.dimisparameter(d.low):
                            raise SyntaxError('%s: static dims must be constants or parameters'%a.name)
                        self.cw('('+self.prefixdimsc(d.low)+')+1);')

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
        for a in self.alist:
            if a.group != currentgroup and a.dynamic:
                if currentgroup != '':
                    self.cw('  }}')
                currentgroup = a.group
                if len(dyngroups) > 0: dyngroups[-1][2] = i
                dyngroups.append([currentgroup,i+1,len(self.alist)])
                self.cw('static void '+self.pname+'setdims'+currentgroup+'(char *name,long i)')
                self.cw('{')
                self.cw('  if (strcmp(name,"'+a.group+'") || strcmp(name,"*")) {')

            i = i + 1
            vname = self.pname+'_farrays['+repr(i)+']'
            if a.dynamic == 1 or a.dynamic == 2:
                j = 0
                self.cw('  if (i == -1 || i == %d) {'%i)
                # --- create lines of the form dims[1] = high-low+1
                for d in a.dims:
                    if d.high == '': continue
                    self.cw('   '+vname+'.dimensions['+repr(j)+']=(npy_intp)((int)',
                            noreturn=1)
                    j = j + 1
                    if re.search('[a-zA-Z]',d.high) == None:
                        self.cw('('+d.high+')-',noreturn=1)
                    else:
                        self.cw('('+self.prefixdimsc(d.high)+')-',noreturn=1)
                    if re.search('[a-zA-Z]',d.low) == None:
                        self.cw('('+d.low+')+1);')
                    else:
                        self.cw('('+self.prefixdimsc(d.low)+')+1);')
                self.cw('  }')

        if currentgroup != '':
            self.cw('  }}')

        # --- Now write out the setdims routine which calls of the routines
        # --- for the individual groups.
        self.cw('void '+self.pname+'setdims(char *name,ForthonObject *obj,long i)')
        self.cw('{')
        for groupinfo in dyngroups:
            self.cw('  if (i == -1 || (%d <= i && i <= %d))'%tuple(groupinfo[1:]),
                    noreturn=1)
            self.cw('  '+self.pname+'setdims'+groupinfo[0]+'(name,i);')
        self.cw('}')

        self.cw('')

        ###########################################################################
        # --- Write set pointers routine which gets all of the fortran pointers
        self.cw('void '+fname(self.fsub('grabscalarpointers'))+'(long *i,char *p)')
        self.cw('{')
        self.cw('  /* Gabs pointer for the scalar */')
        self.cw('  '+self.pname+'_fscalars[*i].data = (char *)p;')
        self.cw('}')

        # --- A serarate routine is needed for derived types since the cobj__
        # --- that is passed in is already a pointer, so **p is needed.
        self.cw('void '+fname(self.fsub('setderivedtypepointers'))+'(long *i,char **p)')
        self.cw('{')
        self.cw('  /* Gabs pointer for the scalar */')
        self.cw('  '+self.pname+'_fscalars[*i].data = (char *)(*p);')
        self.cw('}')

        # --- Get pointer to an array. This takes an integer to specify which array
        self.cw('void '+fname(self.fsub('grabarraypointers'))+'(long *i,char *p)')
        self.cw('{')
        self.cw('  /* Grabs pointer for the array */')
        self.cw('  '+self.pname+'_farrays[*i].data.s = (char *)p;')
        self.cw('}')

        # --- This takes a Fortranarray object directly.
        self.cw('void '+fname(self.fsub('grabarraypointersobj'))+'(Fortranarray *farray,char *p)')
        self.cw('{')
        self.cw('  /* Grabs pointer for the array */')
        self.cw('  farray->data.s = (char *)p;')
        self.cw('}')

        # --- This routine gets the dimensions from an array. It is called from
        # --- fortran and the last argument should be shape(array).
        # --- This is only used for routines with the fassign attribute.
        # --- Note that the dimensions are stored in C order.
        self.cw('void '+fname(self.fsub('setarraydims'))+
                '(Fortranarray *farray,long *dims)')
        self.cw('{')
        self.cw('  int id;')
        self.cw('  for (id=0;id<farray->nd;id++)')
        self.cw('    farray->dimensions[id] = (npy_intp)(dims[id]);')
        self.cw('}')

        ###########################################################################
        # --- And finally, the initialization function
        if sys.hexversion >= 0x03000000:
            self.cw('static struct PyModuleDef moduledef = {')
            self.cw('  PyModuleDef_HEAD_INIT,')
            self.cw('  "{0}py", /* m_name */'.format(self.pname))
            self.cw('  "{0}", /* m_doc */'.format(self.pname))
            self.cw('  -1,                  /* m_size */')
            self.cw('  {0}_methods,    /* m_methods */'.format(self.pname))
            self.cw('  NULL,                /* m_reload */')
            self.cw('  NULL,                /* m_traverse */')
            self.cw('  NULL,                /* m_clear */')
            self.cw('  NULL,                /* m_free */')
            self.cw('  };')

        self.cw('PyMODINIT_FUNC')
        if sys.hexversion >= 0x03000000:
            self.cw('PyInit_'+self.pname+'py(void)')
        else:
            self.cw('init'+self.pname+'py(void)')
        self.cw('{')

        self.cw('  PyObject *m;')
        if self.fcompname == 'nag':
            self.cw('  int argc; char **argv;')
            self.cw('  Py_GetArgcArgv(&argc,&argv);')
            self.cw('  f90_init(argc,argv);')
#   self.cw('  ForthonType.tp_getset = '+self.pname+'_getseters;')
#   self.cw('  ForthonType.tp_methods = '+self.pname+'_methods;')
        self.cw('  if (PyType_Ready(&ForthonType) < 0)')
        if sys.hexversion >= 0x03000000:
            self.cw('    return NULL;')
        else:
            self.cw('    return;')

        if sys.hexversion >= 0x03000000:
            self.cw('  m = PyModule_Create(&moduledef);')
        else:
            self.cw('  m = Py_InitModule("'+self.pname+'py",'+self.pname+'_methods);')

         #self.cw('  PyModule_AddObject(m,"'+self.pname+'Type",'+
         #               '(PyObject *)&ForthonType);')
        self.cw('  '+self.pname+'Object=(ForthonObject *)'+
                   'PyObject_GC_New(ForthonObject, &ForthonType);')
                            #'ForthonObject_New(NULL,NULL);')
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
        self.cw('  '+self.pname+'Object->__module__ = Py_BuildValue("s","'+
                     self.pname+'py");')
        self.cw('  '+self.pname+'Object->fobj = NULL;')
        self.cw('  '+self.pname+'Object->fobjdeallocate = NULL;')
        self.cw('  '+self.pname+'Object->nullifycobj = NULL;')
        self.cw('  '+self.pname+'Object->allocated = 0;')
        self.cw('  '+self.pname+'Object->garbagecollected = 0;')
        self.cw('  PyModule_AddObject(m,"'+self.pname+'",(PyObject *)'+
                    self.pname+'Object);')
        self.cw('  ErrorObject = PyErr_NewException("'+self.pname+'py.error",NULL,NULL);')
        self.cw('  PyModule_AddObject(m,"'+self.pname+'error", ErrorObject);')
        self.cw('  PyModule_AddObject(m,"fcompname",'+
                   'PyUnicode_FromString("'+self.fcompname+'"));')
        self.cw('  PyModule_AddObject(m,"realsize",'+
                   'PyLong_FromLong((long)%s'%realsize+'));')
        self.cw('  if (PyErr_Occurred()) {')
        self.cw('    PyErr_Print();')
        self.cw('    Py_FatalError("can not initialize module '+self.pname+'");')
        self.cw('    }')
        self.cw('  import_array();')
        self.cw('  '+self.pname+'declarevars('+self.pname+'Object);')
        self.cw('  Forthon_BuildDicts('+self.pname+'Object);')
        self.cw('  ForthonPackage_allotdims('+self.pname+'Object);')
        self.cw('  '+fname(self.fsub('passpointers'))+'();')
        self.cw('  '+fname(self.fsub('nullifypointers'))+'();')
        self.cw('  ForthonPackage_staticarrays('+self.pname+'Object);')
        if self.initialgallot:
            self.cw('  {')
            self.cw('  PyObject *s;')
            self.cw('  s = Py_BuildValue("(s)","*");')
            self.cw('  ForthonPackage_gallot((PyObject *)'+self.pname+'Object,s);')
            self.cw('  Py_XDECREF(s);')
            self.cw('  }')

        self.cw('  {')
        self.cw('  PyObject *m, *d, *f, *r;')
        self.cw('  r = NULL;')
        self.cw('  m = PyImport_ImportModule("Forthon");')
        self.cw('  if (m != NULL) {')
        self.cw('    d = PyModule_GetDict(m);')
        self.cw('    if (d != NULL) {')
        self.cw('      f = PyDict_GetItemString(d,"registerpackage");')
        self.cw('      if (f != NULL) {')
        self.cw('        r = PyObject_CallFunction(f,"Os",(PyObject *)'+
                    self.pname+'Object,"'+self.pname+'");')
        self.cw('  }}}')
        self.cw('  if (NULL == r) {')
        self.cw('    if (PyErr_Occurred()) PyErr_Print();')
        self.cw('    Py_FatalError("unable to find a compatible Forthon module in which to register module ' + self.pname + '");')
        self.cw('  }')
        self.cw('  Py_XDECREF(m);')
        self.cw('  Py_XDECREF(r);')
        self.cw('  }')

        if machine=='win32':
            self.cw('  /* Initialize FORTRAN on CYGWIN */')
            self.cw(' initPGfortran();')

        if sys.hexversion >= 0x03000000:
            self.cw('  return m;')

        self.cw('}')
        self.cw('')

        ###########################################################################
        # --- Close the c package module file
        self.cfile.close()

        ###########################################################################
        ###########################################################################
        ###########################################################################
        # --- Write out fortran initialization routines
        self.setffile()
        self.ffile.close()

        ###########################################################################
        ###########################################################################
        # --- Process any derived types
        wrappergen_derivedtypes.ForthonDerivedType(self.typelist,self.pname,
                                   self.pname+'pymodule.c',
                                   self.pname+'_p.F90',self.isz,
                                   self.writemodules,self.fcompname)
        ###########################################################################
        ###########################################################################

        self.setffile()

        ###########################################################################
        # --- Write out f90 modules, including any data statements
        if self.writemodules:
            self.writef90modules()

        ###########################################################################
        self.fw('SUBROUTINE '+self.fsub('passpointers')+'()')

        # --- Write out the Use statements
        for g in self.groups+self.hidden_groups:
            self.fw('  USE '+g)

        # --- Write out calls to c routine passing down pointers to scalars
        for i in range(len(self.slist)):
            s = self.slist[i]
            if s.dynamic: continue
            if s.derivedtype:
            # --- This is only called for static instances, so deallocatable is
            # --- set to false (the last argument).
                self.fw('  call init'+s.type+'py(int('+repr(i)+','+self.isz+'),'+s.name+','+
                        s.name+'%cobj__,int(1,'+self.isz+'),int(0,'+self.isz+'))')
                self.fw('  call '+self.fsub('setderivedtypepointers')+'(int('+repr(i)+','+self.isz+'),'+s.name+'%cobj__)')
            else:
                self.fw('  call '+self.fsub('grabscalarpointers')+'(int('+repr(i)+','+self.isz+'),'+s.name+')')

        # --- Write out calls to c routine passing down pointers to arrays.
        for i in range(len(self.alist)):
            a = self.alist[i]
            if not a.dynamic:
                self.fw('  call '+self.fsub('grabarraypointers')+'(int('+repr(i)+','+self.isz+'),'+a.name+')')

        # --- Finish the routine
        self.fw('  return')
        self.fw('end')

        ###########################################################################
        # --- Nullifies the pointers of all dynamic variables. This is needed
        # --- since in some compilers, the associated routine returns
        # --- erroneous information if the status of a pointer is undefined.
        # --- Pointers must be explicitly nullified in order to get
        # --- associated to return a false value.
        self.fw('SUBROUTINE '+self.fsub('nullifypointers')+'()')

        # --- Write out the Use statements
        for g in self.groups+self.hidden_groups:
            self.fw('  USE '+g)

        for i in range(len(self.slist)):
            s = self.slist[i]
            if s.dynamic: self.fw('  NULLIFY('+s.name+')')
        for i in range(len(self.alist)):
            a = self.alist[i]
            if a.dynamic: self.fw('  NULLIFY('+a.name+')')

        self.fw('  return')
        self.fw('end')
        ###########################################################################
        # --- Write routine for each dynamic variable which gets the pointer from the
        # --- wrapper
        for s in self.slist:
            if (s.dynamic or s.derivedtype) and not s.parameter:
                self.fw('SUBROUTINE '+self.fsub('setscalarpointer',s.name)+'(p__,fobj__,nullit__)')
                self.fw('  USE '+s.group)
                self.fw('  INTEGER('+self.isz+'):: fobj__')
                self.fw('  INTEGER('+self.isz+'):: nullit__')
                if s.type == 'character':
                    self.fw('  character(len='+s.dims[0].high+'),target:: p__')
                else:
                    self.fw('  '+fvars.ftof(s.type)+',target:: p__')
                if s.dynamic:
                    self.fw('  if (nullit__ == 0) then')
                    self.fw('    '+s.name+' => p__')
                    self.fw('  else')
                    self.fw('    NULLIFY('+s.name+')')
                    self.fw('  endif')
                else:
                    self.fw('  '+s.name+' = p__')
                self.fw('  RETURN')
                self.fw('END')
            if s.dynamic:
                # --- In all cases, it is not desirable to create a new instance,
                # --- for example when the object is being deleted.
                self.fw('SUBROUTINE '+self.fsub('getscalarpointer',s.name)+
                                       '(cobj__,fobj__,createnew__)')
                self.fw('  USE '+s.group)
                self.fw('  INTEGER('+self.isz+'):: cobj__,fobj__')
                self.fw('  INTEGER(4):: createnew__')
                self.fw('  if (ASSOCIATED('+s.name+')) then')
                self.fw('    if ('+s.name+'%cobj__ == 0 .and. createnew__ == 1) then')
                self.fw('      call init'+s.type+'py(int(-1,'+self.isz+'),'+s.name+','+
                                                     s.name+'%cobj__,int(0,'+self.isz+'),int(0,'+self.isz+'))')
                self.fw('    endif')
                self.fw('    cobj__ = '+s.name+'%cobj__')
                self.fw('  else')
                self.fw('    cobj__ = 0')
                self.fw('  endif')
                self.fw('  RETURN')
                self.fw('END')

        for a in self.alist:
            if a.dynamic:
                self.fw('SUBROUTINE '+self.fsub('setarraypointer',a.name)+'(p__,fobj__,dims__)')
                groups = self.dimsgroups(a.dimstring)
                groupsprinted = [a.group]
                for g in groups:
                    if g not in groupsprinted:
                        self.fw('  USE '+g)
                        groupsprinted.append(g)
                self.fw('  USE '+a.group)
                self.fw('  integer('+self.isz+'):: fobj__')
                self.fw('  integer('+self.isz+'):: dims__('+repr(len(a.dims))+')')

                if a.type == 'character':
                    self.fw('  character(len='+a.dims[0].high+'),target:: p__'+
                            self.prefixdimsf(re.sub('[ \t\n]','',a.dimstring)))
                else:
                    self.fw('  '+fvars.ftof(a.type)+',target:: p__'+
                            self.prefixdimsf(re.sub('[ \t\n]','',a.dimstring)))
                self.fw('  '+a.name+' => p__')
                self.fw('  return')
                self.fw('end')
                if re.search('fassign',a.attr):
                    self.fw('SUBROUTINE '+self.fsub('getarraypointer',a.name)+'(farray__,fobj__)')
                    self.fw('  USE '+a.group)
                    self.fw('  integer('+self.isz+'):: farray__,fobj__')
                    self.fw('  integer('+self.isz+'):: ss(%d)'%(len(a.dims)))
                    self.fw('  if (.not. associated('+a.name+')) return')
                    self.fw('  call '+self.fsub('grabarraypointersobj')+'(farray__,'+a.name+')')
                    self.fw('  ss = shape('+a.name+')')
                    self.fw('  call '+self.fsub('setarraydims')+'(farray__,ss)')
                    self.fw('  return')
                    self.fw('end')

        ###########################################################################

        # --- Close fortran file
        self.ffile.close()

        scalar_pickle_file = open(self.pname + '.scalars','wb')
        self.sdict['_module_name_'] = self.pname
        pickle.dump(self.sdict, scalar_pickle_file)
        pickle.dump(self.slist, scalar_pickle_file)
        scalar_pickle_file.close()

    def writef90modules(self):
        """
        Write the fortran90 modules
        """
        self.setffile()
        if   self.fcompname == 'xlf': save = ',SAVE'
        else:                         save = ''
        for g in self.groups+self.hidden_groups:
            self.fw('MODULE '+g)
            # --- Check if any variables are derived types. If so, the module
            # --- containing the type must be used.
            printedtypes = []
            for v in self.slist + self.alist:
                if v.group == g and v.derivedtype:
                    if v.type not in printedtypes:
                        self.fw('  USE '+v.type+'module')
                        printedtypes.append(v.type)
            self.fw('  SAVE')
            # --- Declerations for scalars and arrays
            for s in self.slist:
                if s.group == g:
                    self.fw('  '+fvars.ftof(s.type),noreturn=1)
                    if s.dynamic: self.fw(',POINTER',noreturn=1)
                    self.fw(save+':: '+s.name,noreturn=1)
                    if s.data: self.fw('='+s.data[1:-1],noreturn=1)
                    self.fw('')
            for a in self.alist:
                if a.group == g:
                    if a.dynamic:
                        if a.type == 'character':
                            self.fw('  character(len='+a.dims[0].high+'),pointer'+save+':: '+a.name,noreturn=1)
                            ndims = len(a.dims) - 1
                        else:
                            self.fw('  '+fvars.ftof(a.type)+',pointer'+save+':: '+a.name,noreturn=1)
                            ndims = len(a.dims)
                        if ndims > 0:
                            self.fw('('+(ndims*':,')[:-1]+')',noreturn=1)
                        self.fw('')
                    else:
                        if a.type == 'character':
                            self.fw('  character(len='+a.dims[0].high+')'+save+':: '+a.name+
                                    re.sub('[ \t\n]','',a.dimstring))
                        else:
                            self.fw('  '+fvars.ftof(a.type)+save+':: '+
                                    a.name+re.sub('[ \t\n]','',a.dimstring))
                        if a.data:
                            # --- Add line continuation marks if the data line extends over
                            # --- multiple lines.
                            dd = re.sub(r'\n','&\n',a.data)
                            self.fw('  data '+a.name+dd)
            self.fw('END MODULE '+g)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

module_prefix_pat = re.compile ('([a-zA-Z_][a-zA-Z0-9_]*)\.scalars')
def get_another_scalar_dict(file_name,other_scalar_vars):
    m = module_prefix_pat.search(file_name)
    if m.start() == -1: raise SyntaxError('expect a .scalars file')
    f = open(file_name,'rb')
    vars = []
    vars.append(pickle.load(f))
    vars.append(pickle.load(f))
    other_scalar_vars.append(vars)
    f.close()

def wrappergenerator_main(argv=None,writef90modulesonly=0):
    # --- Get package name from argument list
    try:
        pname = args[0]
        ifile = args[1]
        otherfortranfiles = args[2:]
        #pname = os.path.splitext(os.path.split(ifile)[1])[0]
        #pname = args[0][:re.search('\.',args[0]).start()]
    except IndexError:
        print PyWrap.__doc__
        sys.exit(1)

    # --- get other command line options and default actions
    initialgallot = options.initialgallot
    fcompname = options.fcomp
    writemodules = options.writemodules
    timeroutines = options.timeroutines
    otherinterfacefiles = options.othermacros

    # --- a list of scalar dictionaries from other modules.
    other_scalar_vars = []
    for d in options.dependencies:
        get_another_scalar_dict(d,other_scalar_vars)

    cc = PyWrap(ifile,pname,initialgallot,writemodules,
                otherinterfacefiles,other_scalar_vars,timeroutines,
                otherfortranfiles,fcompname)
    if writef90modulesonly:
        cc.writef90modules()
    else:
        cc.createmodulefile()

    # --- forthonf2c.h is imported by Forthon.h, and defines macros needed for strings.
    writeforthonf2c()

# --- This might make some of the write statements cleaner.
# --- From http://aspn.activestate.com/ASPN/Python/Cookbook/
class PrintEval:
    def __init__(self, globals=None, locals=None):
        self.globals = globals or {}
        self.locals = locals or None

    def __getitem__(self, key):
        if self.locals is None:
            self.locals = sys._getframe(1).f_locals
        key = key % self
        return eval(key, self.globals, self.locals)

if __name__ == '__main__':
    wrappergenerator_main(sys.argv[1:])

