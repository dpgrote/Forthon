"""Generates the wrapper for derived types.
"""
import fvars
import string,md5
from cfinterface import *

class ForthonDerivedType:
  def __init__(self,typelist,pname,c,f,f90,isz,writemodules):
    if not typelist: return

    self.cfile = open(c,'a')
    self.ffile = open(f,'a')
    self.wrapderivedtypes(typelist,pname,f90,isz,writemodules)
    self.cfile.close()
    self.ffile.close()

  transtable = (10*string.ascii_lowercase)[:256]
  def fsub(self,type,prefix,suffix=''):
    """
    The fortran standard limits routine names to 31 characters. If the
    routine name is longer than that, this routine takes the first 15
    characters and and creates a hashed string based on the full name to get
    the next 16. This does not guarantee uniqueness, but the nonuniqueness
    should be minute.
    """
    name = type.name+prefix+suffix
    if len(name) < 32: return name
    hash = string.translate(md5.new(name).digest(),
                            ForthonDerivedType.transtable)
    return name[:15] + hash

  # --- Convert fortran variable name into reference from list of variables.
  def prefixdimsc(self,dim,sdict):
    sl=re.split('[ ()/\*\+\-]',dim)
    for ss in sl:
      if re.search('[a-zA-Z]',ss) != None:
        if sdict.has_key (ss):
          dim = re.sub(ss,'*(int *)obj->fscalars['+repr(sdict[ss])+'].data',
                       dim,count=1)
        else:
          raise ss + ' is not declared in a .v file'
    return string.lower(dim)

  # --- Convert variable names in to type elements
  def prefixdimsf(self,dim,sdict):
    sl=re.split('[ ()/\*\+\-:,]',dim)
    # --- Loop over the list of items twice. The first time, add in the 'obj%'
    # --- prefix but overwrite the item with '=='. Then go back again and
    # --- overwrite the '==' with the item. This is done in case one name
    # --- has another in it or a item is repeated. Note that this could also
    # --- be done instead by using re syntax in the substring being searched
    # --- for, by forcing complete words to be found, without a leading % sign.
    # --- But this works as is and is fast enough.
    for ss in sl:
      if re.search('[a-zA-Z]',ss) != None:
        if sdict.has_key(ss):
          dim = re.sub(ss,'obj__%==',dim,count=1)
    for ss in sl:
      if re.search('[a-zA-Z]',ss) != None:
        if sdict.has_key(ss):
          dim = re.sub('==',ss,dim,count=1)
    # --- Check for any unspecified dimensions and replace it with an element
    # --- from the dims array.
    sl = re.split(',',dim[1:-1])
    for i in range(len(sl)):
      if sl[i] == ':': sl[i] = 'dims__(%d)'%(i+1)
    dim = '(' + string.join(sl,',') + ')'
    return string.lower(dim)

  # --------------------------------------------
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

  # --- This is the routine that does all of the work for derived types
  def wrapderivedtypes(self,typelist,pname,f90,isz,writemodules):
    for t in typelist:
      self.cw('')
      vlist = t.vlist[:]

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

      #########################################################################
      # --- Print out the external commands
      self.cw('extern void '+fname(self.fsub(t,'passpointers'))+'(char *fobj,'+
                                                       'long *setinitvalues);')
      self.cw('extern PyObject *'+fname(self.fsub(t,'newf'))+'(void);')

      # --- setpointer and getpointer routine for f90
      # --- Note that setpointers get written out for all derived types -
      # --- for non-dynamic derived types, the setpointer routine does a copy.
      for s in slist:
        self.cw('extern void '+fname(self.fsub(t,'setpointer',s.name))+
                '(char *p,char *fobj);')
        if s.dynamic:
          self.cw('extern void '+fname(self.fsub(t,'getpointer',s.name))+
                  '(ForthonObject **cobj__,char *obj);')
      for a in alist:
        self.cw('extern void '+fname(self.fsub(t,'setpointer',a.name))+
                  '(char *p,char *fobj,long *dims__);')
        if re.search('fassign',a.attr):
          self.cw('extern void '+fname(self.fsub(t,'getpointer',a.name))+
                  '(long *i,char* fobj);')
      self.cw('')
  
      #########################################################################
      # --- Write declarations of c pointers to fortran variables
      self.cw('void '+t.name+'declarevars(ForthonObject *obj) {')

      # --- Scalars
      self.cw('obj->nscalars = '+repr(len(slist))+';')
      if len(slist) > 0:
        self.cw('obj->fscalars = malloc(obj->nscalars*sizeof(Fortranscalar));')
      else:
        self.cw('obj->fscalars = NULL;')
      for i in range(len(slist)):
        s = slist[i]
        setpointer = '*'+fname(self.fsub(t,'setpointer',s.name))
        if s.dynamic: getpointer = '*'+fname(self.fsub(t,'getpointer',s.name))
        else:         getpointer = 'NULL'
        self.cw('obj->fscalars[%d].type = PyArray_%s;'%(i,fvars.ftop(s.type)))
        self.cw('obj->fscalars[%d].name = "%s";'%(i,s.name))
        self.cw('obj->fscalars[%d].data = NULL;'%i)
        self.cw('obj->fscalars[%d].group = "%s";'%(i,t.name))
        self.cw('obj->fscalars[%d].attributes = "%s";'%(i,s.attr))
        self.cw('obj->fscalars[%d].comment = "%s";'%(i,
                             string.replace(repr(s.comment)[1:-1],'"','\\"')))
                             # The repr is used so newlines get written out
                             # as \n.
        self.cw('obj->fscalars[%d].dynamic = %d;'%(i,s.dynamic))
        self.cw('obj->fscalars[%d].setpointer = %s;'%(i,setpointer))
        self.cw('obj->fscalars[%d].getpointer = %s;'%(i,getpointer))

      # --- Arrays
      self.cw('obj->narrays = '+repr(len(alist))+';')
      if len(alist) > 0:
        self.cw('obj->farrays = malloc(obj->narrays*sizeof(Fortranarray));')
      else:
        self.cw('obj->farrays = NULL;')
      for i in range(len(alist)):
        a = alist[i]
        if a.dynamic: setpointer = '*'+fname(self.fsub(t,'setpointer',a.name))
        else:         setpointer = 'NULL'
        if re.search('fassign',a.attr):
          getpointer = '*'+fname(self.fsub(t,'getpointer',a.name))
        else:
          getpointer = 'NULL'
        if a.data and a.dynamic: initvalue = a.data[1:-1]
        else:                    initvalue = '0'
        self.cw('obj->farrays[%d].type = PyArray_%s;'%(i,fvars.ftop(a.type)))
        self.cw('obj->farrays[%d].dynamic = %d;'%(i,a.dynamic))
        self.cw('obj->farrays[%d].nd = %d;'%(i,len(a.dims)))
        self.cw('obj->farrays[%d].dimensions = NULL;'%i)
        self.cw('obj->farrays[%d].name = "%s";'%(i,a.name))
        self.cw('obj->farrays[%d].data.s = (char *)NULL;'%i)
        self.cw('obj->farrays[%d].setpointer = %s;'%(i,setpointer))
        self.cw('obj->farrays[%d].getpointer = %s;'%(i,getpointer))
        self.cw('obj->farrays[%d].initvalue = %s;'%(i,initvalue))
        self.cw('obj->farrays[%d].pya = NULL;'%i)
        self.cw('obj->farrays[%d].group = "%s";'%(i,a.group))
        self.cw('obj->farrays[%d].attributes = "%s";'%(i,a.attr))
        self.cw('obj->farrays[%d].comment = "%s";'%(i,
                            string.replace(repr(a.comment)[1:-1],'"','\\"')))
        self.cw('obj->farrays[%d].dimstring = "%s";'%(i,a.dimstring))
      self.cw('}')

#     # --- Write out the table of getset routines
#     self.cw('')
#     self.cw('static PyGetSetDef '+t.name+'_getseters[] = {')
#     for i in range(len(slist)):
#       s = slist[i]
#       if s.type == 'real': gstype = 'double'
#       elif s.type == 'integer': gstype = 'integer'
#       elif s.type == 'complex': gstype = 'cdouble'
#       else:                    gstype = 'derivedtype'
#       self.cw('{"'+s.name+'",(getter)Forthon_getscalar'+gstype+
#                            ',(setter)Forthon_setscalar'+gstype+
#                      ',"%s"'%string.replace(repr(s.comment)[1:-1],'"','\\"') +
#                           ',(void *)'+repr(i)+'},')
#     for i in range(len(alist)):
#       a = alist[i]
#       self.cw('{"'+a.name+'",(getter)Forthon_getarray'+
#                            ',(setter)Forthon_setarray'+
#                      ',"%s"'%string.replace(repr(a.comment)[1:-1],'"','\\"') +
#                           ',(void *)'+repr(i)+'},')
#     self.cw('{"scalardict",(getter)Forthon_getscalardict,'+
#                           '(setter)Forthon_setscalardict,'+
#             '"internal scalar dictionary",NULL},')
#     self.cw('{"arraydict",(getter)Forthon_getarraydict,'+
#                          '(setter)Forthon_setarraydict,'+
#             '"internal array dictionary",NULL},')
#     self.cw('{NULL}};')

      #########################################################################
      # --- Write static array initialization routines
      self.cw('void '+t.name+'setstaticdims(ForthonObject *obj)')
      self.cw('{')

      i = -1
      for a in alist:
        i = i + 1
        vname = 'obj->farrays[%d]'%i
        if a.dims and not a.dynamic:
          j = 0
          for d in a.dims:
            self.cw('  '+vname+'.dimensions['+repr(len(a.dims)-1-j)+'] = ('+
                    d.high+') - ('+d.low+') + 1;')
            j = j + 1
      self.cw('}')

      #########################################################################
      # --- Write routine which sets the dimensions of the dynamic arrays.
      # --- This is done in a seperate routine so it only appears once.
      # --- A routine is written out for each group which has dynamic arrays.
      # --- Then a routine is written which calls all of the individual group
      # --- routines. That is done to reduce the strain on the compiler by
      # --- reducing the size of the routines. (In fact, in one case, with
      # --- everything in one routine, the cc compiler was giving a core dump!)
      # --- Loop over the variables. This assumes that the variables are sorted
      # --- by group.
      self.cw('static void '+t.name+'setdims(char *name,ForthonObject *obj)')
      self.cw('{')

      i = -1
      for a in alist:
        i = i + 1
        vname = 'obj->farrays[%d]'%i
        if a.dynamic:
          j = 0
          # --- create lines of the form dims[1] = high - low + 1, in
          # --- reverse order
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
              self.cw('('+self.prefixdimsc(d.low,sdict)+')+1;')
      self.cw('}')

      #########################################################################
      self.cw('PyObject *'+pname+'_'+t.name+
              'New(PyObject *self, PyObject *args)')
      self.cw('{')
      self.cw('return '+fname(self.fsub(t,'newf'))+'();')
      self.cw('}')
      #########################################################################
      # --- Write out an empty list of methods
      self.cw('static struct PyMethodDef '+t.name+'_methods[]={{NULL,NULL}};')
      #########################################################################
      # --- And finally, the initialization function
      self.cw('void '+fname('init'+t.name+'py')+
                   '(long *i,char *fobj,ForthonObject **cobj__,'+
                    'long *setinitvalues)')
      self.cw('{')
      self.cw('  ForthonObject *obj;')
      self.cw('  obj=(ForthonObject *) PyObject_NEW(ForthonObject,'+
                 '&ForthonType);')
      self.cw('  if (*i > 0) {obj->name = '+pname+'_fscalars[*i].name;}')
      self.cw('  else        {obj->name = "pointee";}')
      self.cw('  obj->typename = "'+t.name+'";')
      self.cw('  '+t.name+'declarevars(obj);')
      self.cw('  obj->setdims = *'+t.name+'setdims;')
      self.cw('  obj->setstaticdims = *'+t.name+'setstaticdims;')
      self.cw('  obj->fmethods = '+t.name+'_methods;')
      self.cw('  obj->fobj = fobj;')
      self.cw('  obj->allocated = 0;')
      self.cw('  *cobj__ = obj;')
      self.cw('  if (PyErr_Occurred())')
      self.cw('    Py_FatalError("can not initialize type '+t.name+'");')
      self.cw('  import_array();')
      self.cw('  Forthon_BuildDicts(obj);')
      self.cw('  ForthonPackage_allotdims(obj);')
      self.cw('  '+fname(self.fsub(t,'passpointers'))+'(fobj,setinitvalues);')
      self.cw('  ForthonPackage_staticarrays(obj);')
      self.cw('}')

      #########################################################################
      # --- The destructor
      self.cw('void '+fname('del'+t.name+'py')+'(ForthonObject **cobj__)')
      self.cw('{')
      self.cw('  int i;')
      self.cw('  for (i=0;i<(*cobj__)->narrays;i++) {')
      self.cw('    free((*cobj__)->farrays[i].dimensions);')
      self.cw('    Py_XDECREF((*cobj__)->farrays[i].pya);')
      self.cw('    }')
      self.cw('  free((*cobj__)->fscalars);')
      self.cw('  free((*cobj__)->farrays);')
      self.cw('  Forthon_DeleteDicts(*cobj__);')
      self.cw('  Py_XDECREF(*cobj__);')
      self.cw('}')

      #########################################################################
      # --- Write set pointers routine which gets all of the fortran pointers
      self.cw('void '+fname(self.fsub(t,'setscalarpointers'))+
              '(int *i,char *p,ForthonObject **obj',noreturn=1)
      if machine=='J90':
        self.cw(',int *iflag)')
      else:
        self.cw(')')
      self.cw('{')
      self.cw('  /* Get pointers for the scalars */')
      self.cw('  (*obj)->fscalars[*i].data = (char *)p;')
      if machine=='J90':
        self.cw('    if (iflag) {')
        self.cw('      (*obj)->fscalars[*i].data=_fcdtocp((_fcd)p);}')
      self.cw('}')

      self.cw('void '+fname(self.fsub(t,'setderivedtypepointers'))+
              '(int *i,char **p,ForthonObject **obj)')
      self.cw('{')
      self.cw('  /* Get pointers for the derived types */')
      self.cw('  (*obj)->fscalars[*i].data = (char *)(*p);')
      self.cw('}')

      self.cw('void '+fname(self.fsub(t,'setarraypointers'))+
              '(int *i,char *p,ForthonObject **obj',noreturn=1)
      if machine=='J90':
        self.cw(',int *iflag)')
      else:
        self.cw(')')
      self.cw('{')
      self.cw('  /* Get pointers for the arrays */')
      self.cw('  (*obj)->farrays[*i].data.s = (char *)p;')
      if machine=='J90':
        self.cw('    if (iflag) {')
        self.cw('      (*obj)->farrays[*i].data.s=_fcdtocp((_fcd)p);}')
      self.cw('}')

      self.cw('void '+fname(self.fsub(t,'setarraypointersobj'))+
              '(Fortranarray *farray,char *p',noreturn=1)
      if machine=='J90':
        self.cw(',int *iflag)')
      else:
        self.cw(')')
      self.cw('{')
      self.cw('  /* Get pointers for the arrays */')
      self.cw('  farray->data.s = (char *)p;')
      if machine=='J90':
        self.cw('    if (iflag) {')
        self.cw('      farray->data.s=_fcdtocp((_fcd)p);}')
      self.cw('}')

      # --- This routine gets the dimensions from an array. It is called from
      # --- fortran and the last argument should be shape(array).
      # --- This is only used for routines with the fassign attribute.
      self.cw('void '+fname(self.fsub(t,'setarraydims'))+
              '(Fortranarray *farray,int *dims)')
      self.cw('{')
      if f90:
        self.cw('  int id;')
        self.cw('  for (id=0;id<farray->nd;id++)')
        self.cw('    farray->dimensions[id] = dims[id];')
      self.cw('}')

      #########################################################################
      #########################################################################

      #########################################################################
      # --- Write out f90 modules, including any data statements
      if writemodules:
        self.fw('')
        g = t.name
        self.fw('MODULE '+t.name+'module')

        # --- Check if any variables are derived types. If so, the module
        # --- containing the type must be used. This module though does not
        # --- need to include itself of course.
        printedtypes = [t.name]
        for v in slist + alist:
          if v.derivedtype:
            if v.type not in printedtypes:
              self.fw('  USE '+v.type+'module')
              printedtypes.append(v.type)

        self.fw('  SAVE')
        self.fw('  TYPE '+t.name+'')
        self.fw('    INTEGER('+isz+'):: cobj__')
        for s in slist:
          self.fw('    '+fvars.ftof(s.type),noreturn=1)
          if s.dynamic: self.fw(',POINTER',noreturn=1)
          self.fw(':: '+s.name)
          # --- data statement is handle by the passpointer routine
        for a in alist:
          if a.dynamic:
            if a.type == 'character':
              self.fw('    CHARACTER(LEN='+a.dims[0].high+'),POINTER:: '+
                      a.name,noreturn=1)
              ndims = len(a.dims) - 1
            else:
              self.fw('    '+fvars.ftof(a.type)+',POINTER:: '+a.name,noreturn=1)
              ndims = len(a.dims)
            if ndims > 0:
              self.fw('('+(ndims*':,')[:-1]+')',noreturn=1)
            self.fw('')
          else:
            if a.type == 'character':
              self.fw('    CHARACTER(LEN='+a.dims[0].high+'):: '+
                      a.name+a.dimstring,noreturn=1)
            else:
              self.fw('    '+fvars.ftof(a.type)+':: '+
                      a.name+a.dimstring)
            # --- data statement is handle by the passpointer routine
            #if a.data:
            #  # --- Add line continuation marks if the data line extends over
            #  # --- multiple lines.
            #  dd = re.sub(r'\n','&',a.data)
            #  self.fw(' = ('+dd+')',noreturn=1)
            #self.fw('')

        self.fw('  END TYPE '+t.name+'')

        # --- These functions must be in the module so that its return type
        # --- is defined.
        # --- Note that the body of the New function is replicated below in
        # --- the NewF function. Any changes here should be made there.
        self.fw('CONTAINS')
        self.fw('  FUNCTION New'+t.name+'() RESULT(newobj__)')
        self.fw('    TYPE('+t.name+'),pointer:: newobj__')
        self.fw('    integer:: error')
        self.fw('    ALLOCATE(newobj__,STAT=error)')
        self.fw('    if (error /= 0) then')
        self.fw('      print*,"ERROR during allocation of '+t.name+'"')
        self.fw('      stop')
        self.fw('    endif')
        for s in slist:
          if s.dynamic:
            self.fw('    NULLIFY(newobj__%'+s.name+')')
        self.fw('    call InitPyRef'+t.name+'(newobj__,1)')
        self.fw('    RETURN')
        self.fw('  END FUNCTION New'+t.name+'')
        self.fw('  SUBROUTINE Del'+t.name+'(oldobj__)')
        self.fw('    TYPE('+t.name+'),pointer:: oldobj__')
        self.fw('    integer:: error')
        self.fw('    call DelPyRef'+t.name+'(oldobj__)')
        self.fw('    DEALLOCATE(oldobj__,STAT=error)')
        self.fw('    if (error /= 0) then')
        self.fw('      print*,"ERROR during deallocation of '+t.name+'"')
        self.fw('      stop')
        self.fw('    endif')
        self.fw('    RETURN')
        self.fw('  END SUBROUTINE Del'+t.name+'')
        self.fw('END MODULE '+t.name+'module')

      # --- These subroutines are written outside of the module in case
      # --- write module is false. This way, they are always written
      # --- out.
      # --- The InitPyRef and DelPyRef are called by the New and Del routines
      # --- if the modules are written. They are also meant to be explicitly
      # --- called from the users Fortran code if the creation and deletion
      # --- of derived type instances is down there.
      # --- The memory handling routines check if the cobj__ has been
      # --- set - if not, then set it. Also delete it afterwards so there
      # --- is not a memory leak. Note that the check may not always work
      # --- since some compilers may leave garbage in cobj__ upon object
      # --- creation.
      self.fw('SUBROUTINE InitPyRef'+t.name+'(newobj__,setinitvalues)')
      self.fw('  USE '+t.name+'module')
      self.fw('  TYPE('+t.name+'):: newobj__')
      self.fw('  INTEGER('+isz+'):: setinitvalues')
      self.fw('  call init'+t.name+'py(-1,newobj__,newobj__%cobj__,'+
                                       'setinitvalues)')
      self.fw('  RETURN')
      self.fw('END SUBROUTINE InitPyRef'+t.name)
      self.fw('SUBROUTINE DelPyRef'+t.name+'(oldobj__)')
      self.fw('  USE '+t.name+'module')
      self.fw('  TYPE('+t.name+'):: oldobj__')
      self.fw('  call del'+t.name+'py(oldobj__%cobj__)')
      self.fw('  oldobj__%cobj__ = 0')
      self.fw('  RETURN')
      self.fw('END SUBROUTINE DelPyRef'+t.name)
      self.fw('SUBROUTINE '+t.name+'allot(obj__)')
      self.fw('  USE '+t.name+'module')
      self.fw('  TYPE('+t.name+'):: obj__')
      self.fw('  logical:: delcobj__ = .false.')
      self.fw('  if (obj__%cobj__ == 0) then')
      self.fw('    call InitPyRef'+t.name+'(obj__,0)')
      self.fw('    delcobj__ = .true.')
      self.fw('  endif')
      self.fw('  CALL tallot(obj__%cobj__)')
      self.fw('  if (delcobj__) call DelPyRef'+t.name+'(obj__)')
      self.fw('  RETURN')
      self.fw('END SUBROUTINE '+t.name+'allot')
      self.fw('SUBROUTINE '+t.name+'change(obj__)')
      self.fw('  USE '+t.name+'module')
      self.fw('  TYPE('+t.name+'):: obj__')
      self.fw('  logical:: delcobj__ = .false.')
      self.fw('  if (obj__%cobj__ == 0) then')
      self.fw('    call InitPyRef'+t.name+'(obj__,0)')
      self.fw('    delcobj__ = .true.')
      self.fw('  endif')
      self.fw('  CALL tchange(obj__%cobj__)')
      self.fw('  if (delcobj__) call DelPyRef'+t.name+'(obj__)')
      self.fw('  RETURN')
      self.fw('END SUBROUTINE '+t.name+'change')
      self.fw('SUBROUTINE '+t.name+'free(obj__)')
      self.fw('  USE '+t.name+'module')
      self.fw('  TYPE('+t.name+'):: obj__')
      self.fw('  logical:: delcobj__ = .false.')
      self.fw('  if (obj__%cobj__ == 0) then')
      self.fw('    call InitPyRef'+t.name+'(obj__,0)')
      self.fw('    delcobj__ = .true.')
      self.fw('  endif')
      self.fw('  CALL tfree(obj__%cobj__)')
      self.fw('  if (delcobj__) call DelPyRef'+t.name+'(obj__)')
      self.fw('  RETURN')
      self.fw('END SUBROUTINE '+t.name+'free')

      #########################################################################
      self.fw('SUBROUTINE '+self.fsub(t,'passpointers')+'(obj__,setinitvalues)')

      # --- Write out the Use statements
      self.fw('  USE '+t.name+'module')
      self.fw('  TYPE('+t.name+'):: obj__')
      self.fw('  INTEGER('+isz+'):: setinitvalues')
 
      # --- Write out calls to c routine passing down pointers to scalars
      for i in range(len(slist)):
        s = slist[i]
        if s.dynamic: continue
        if s.derivedtype:
          self.fw('  CALL init'+s.type+'py(-1,obj__%'+s.name+
                  ',obj__%'+s.name+'%cobj__,setinitvalues)')
          self.fw('  CALL '+self.fsub(t,'setderivedtypepointers')+'('+
                  repr(i)+',obj__%'+s.name+'%cobj__,obj__%cobj__)')
        else:
          self.fw('  CALL '+self.fsub(t,'setscalarpointers')+'('+
                  repr(i)+',obj__%'+s.name+',obj__%cobj__',noreturn=1)
          if machine == 'J90':
            if s.type == 'string' or s.type == 'character':
              self.fw(',1)')
            else:
              self.fw(',0)')
          else:
            self.fw(')')

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
        if not a.dynamic:
          if not a.derivedtype:
            # --- This assumes that a scalar is given which is broadcasted
            # --- to fill the array.
            if a.data: self.fw('  obj__%'+a.name+' = '+a.data[1:-1])
            self.fw('  CALL '+self.fsub(t,'setarraypointers')+'('+repr(i)+
                    ',obj__%'+a.name+',obj__%cobj__'+str)

      # --- Set the initial values only if the input flag is 1.
      self.fw('  if (setinitvalues == 1) then')
      for s in slist:
        if s.dynamic or s.derivedtype or not s.data: continue
        self.fw('    obj__%'+s.name+' = '+s.data[1:-1])
      for a in alist:
        if a.dynamic or a.derivedtype or not a.data: continue
        self.fw('    obj__%'+a.name+' = '+a.data[1:-1])
      self.fw('  endif')

      # --- Finish the routine
      self.fw('  RETURN')
      self.fw('END')

      #########################################################################
      # --- Write routine for each dynamic variable which gets the pointer
      # --- from the wrapper
      for s in slist:
        self.fw('SUBROUTINE '+self.fsub(t,'setpointer',s.name)+'(p__,obj__)')
        self.fw('  USE '+t.name+'module')
        self.fw('  TYPE('+t.name+'):: obj__')
        self.fw('  '+fvars.ftof(s.type)+',target::p__')
        if s.dynamic:
          self.fw('  obj__%'+s.name+' => p__')
        else:
          self.fw('  obj__%'+s.name+' = p__')
        self.fw('  RETURN')
        self.fw('END')
        if s.dynamic:
          self.fw('SUBROUTINE '+self.fsub(t,'getpointer',s.name)+'(cobj__,obj__)')
          self.fw('  USE '+t.name+'module')
          self.fw('  integer('+isz+'):: cobj__')
          self.fw('  TYPE('+t.name+'):: obj__')
          self.fw('  if (ASSOCIATED(obj__%'+s.name+')) then')
          self.fw('    cobj__ = obj__%'+s.name+'%cobj__')
          self.fw('  else')
          self.fw('    cobj__ = 0')
          self.fw('  endif')
          self.fw('  RETURN')
          self.fw('END')

      for a in alist:
        if a.dynamic:
          self.fw('SUBROUTINE '+self.fsub(t,'setpointer',a.name)+'(p__,obj__,dims__)')
          self.fw('  USE '+t.name+'module')
          self.fw('  TYPE('+t.name+'):: obj__')
          self.fw('  integer('+isz+'):: dims__('+repr(len(a.dims))+')')
          self.fw('  '+fvars.ftof(a.type)+',target:: '+
                  'p__'+self.prefixdimsf(a.dimstring,sdict)+'')
          self.fw('  obj__%'+a.name+' => p__')
          self.fw('  RETURN')
          self.fw('END')
          if re.search('fassign',a.attr):
            self.fw('SUBROUTINE '+self.fsub(t,'getpointer',a.name)+'(i__,obj__)')
            self.fw('  USE '+t.name+'module')
            self.fw('  integer('+isz+'):: i__')
            self.fw('  TYPE('+t.name+'):: obj__')
            self.fw('  call '+t.name+'setarraypointersobj(i__,obj__%'+a.name+')')
            self.fw('  call '+t.name+'setarraydims(i__'+
                                                   ',shape(obj__%'+a.name+'))')
            self.fw('  return')
            self.fw('end')

      #########################################################################
      # --- Write the routine which creates a new instance of the derived type
      # --- Note that part of the body of this routine is taken from the New
      # --- routine above. Any change in one should be copied to the other.
      # --- The body is copied from New since in cases where the modules
      # --- are not written out, the New routine will not exist.
      self.fw('FUNCTION '+self.fsub(t,'NewF')+'() RESULT(cobj__)')
      self.fw('  USE '+t.name+'module')
      self.fw('  integer('+isz+'):: cobj__')
      self.fw('  integer:: error')
      self.fw('  TYPE('+t.name+'),pointer:: newobj__')
      self.fw('  ALLOCATE(newobj__,STAT=error)')
      self.fw('  if (error /= 0) then')
      self.fw('    print*,"ERROR during allocation of '+t.name+'"')
      self.fw('    stop')
      self.fw('  endif')
      for s in slist:
        if s.dynamic:
          self.fw('  NULLIFY(newobj__%'+s.name+')')
      self.fw('  call InitPyRef'+t.name+'(newobj__,1)')
      self.fw('  cobj__ = newobj__%cobj__')
      self.fw('  RETURN')
      self.fw('END')

#     self.fw('SUBROUTINE '+self.fsub(t,'DelF')+'(oldobj__)')
#     self.fw('  USE '+t.name+'module')
#     self.fw('  TYPE('+t.name+'),pointer:: oldobj__')
#     self.fw('  integer:: error')
#     self.fw('  call DelPyRef'+t.name+'(oldobj__)')
#     self.fw('  DEALLOCATE(oldobj__,STAT=error)')
#     self.fw('  if (error /= 0) then')
#     self.fw('    print*,"ERROR during deallocation of '+t.name+'"')
#     self.fw('    stop')
#     self.fw('  endif')
#     self.fw('  RETURN')
#     self.fw('END SUBROUTINE '+self.fsub(t,'DelF'))

