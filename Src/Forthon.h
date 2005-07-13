/* Created by David P. Grote, March 6, 1998 */
/* $Id: Forthon.h,v 1.37 2005/07/13 01:05:06 dave Exp $ */

#include <Python.h>
#include <Numeric/arrayobject.h>
#include <pythonrun.h>
#include "forthonf2c.h"

/* These are included for the cputime routine below. */
#include <sys/times.h>
#include <unistd.h>

static PyObject *ErrorObject;

#define ARRAY_REVERSE_STRIDE(A) {                             \
  int _i,_m,_n=A->nd;                                         \
  for(_i=0; _i < _n/2 ; _i++) {                               \
    _m = A->strides[_i];                                      \
    A->strides[_i] = A->strides[_n-_i-1];                     \
    A->strides[_n-_i-1] = _m;}                                \
  A->flags &= ~CONTIGUOUS;}

#define ARRAY_REVERSE_DIM(A) {                                \
  int _i,_m,_n=A->nd;                                         \
  for(_i=0; _i < _n/2 ; _i++) {                               \
    _m = A->dimensions[_i];                                   \
    A->dimensions[_i] = A->dimensions[_n-_i-1];               \
    A->dimensions[_n-_i-1] = _m;}                             \
  A->flags &= ~CONTIGUOUS;}

/* This macro converts a python object into a python array    */
/* and then checks if the array is in fortran order.          */
/* If it is not, the array is reordered.                      */
/* If PyArray_FromObject returns NULL meaning the the input   */
/* can not be converted into an array, A1 is also set to NULL.*/
/* This probably should be modified to include casting if it  */
/* is needed.                                                 */
#define FARRAY_FROMOBJECT(A1, A2, ARRAY_TYPE) {               \
  PyArrayObject *_tmp;int _c,_j;                              \
  _tmp=(PyArrayObject *)PyArray_FromObject(A2,ARRAY_TYPE,0,0);\
  _c = 1;                                                     \
  if (_tmp != NULL) {                                         \
    _c = _tmp->descr->elsize;                                 \
    for (_j=0 ; _j < _tmp->nd-1 ; _j++) {                     \
      if (_tmp->strides[_j] != _c) {_c=0;break;}              \
      _c *= _tmp->dimensions[_j];}}                           \
  if (!_c) {                                                  \
    ARRAY_REVERSE_DIM(_tmp);                                  \
    ARRAY_REVERSE_STRIDE(_tmp);                               \
    A1 = (PyArrayObject *)PyArray_ContiguousFromObject((PyObject *)_tmp,ARRAY_TYPE,0,0);                                                      \
    ARRAY_REVERSE_DIM(_tmp);                                  \
    ARRAY_REVERSE_STRIDE(_tmp);                               \
    Py_XDECREF(_tmp);                                         \
    ARRAY_REVERSE_DIM(A1);                                    \
    ARRAY_REVERSE_STRIDE(A1);}                                \
  else                                                        \
    {A1 = _tmp;}                                              \
  }

#define onError(message)                                      \
   { PyErr_SetString(ErrorObject,message); return NULL;}

#define returnnone {Py_INCREF(Py_None);return Py_None;}
#define MYSTRCMP(S1,L1,S2,L2) (L1==L2?strncmp(S1,S2,L1)==0:0)

/* ####################################################################### */
/* # Write definition of scalar and array structures. Note that if these   */
/* # are changed, then the declarations written out in wrappergenerator    */
/* # must also be changed. */
typedef struct {
  int type;
  char *typename;
  char* name;
  char* data;
  char* group;
  char* attributes;
  char* comment;
  int dynamic;
  void (*setpointer)();
  void (*getpointer)();
  } Fortranscalar;

typedef struct {
  int type;
  int dynamic;
  int nd;
  int* dimensions;
  char* name;
  union {char* s;char** d;} data;
  void (*setpointer)();
  void (*getpointer)();
  double initvalue;
  PyArrayObject* pya;
  char* group;
  char* attributes;
  char* comment;
  char* dimstring;
  } Fortranarray;
  
/* ######################################################################### */
/* # Write definition of fortran package type */
typedef struct {
  PyObject_HEAD
  char *name;
  char *typename;
  int nscalars;
  Fortranscalar *fscalars;
  int narrays;
  Fortranarray  *farrays;
  void (*setdims)();
  void (*setstaticdims)();
  PyMethodDef *fmethods;
  PyObject *scalardict,*arraydict;
  char *fobj;
  void (*fobjdeallocate)();
  void (*nullifycobj)();
  int allocated;
  int garbagecollected;
} ForthonObject;
staticforward PyTypeObject ForthonType;

/* This is needed to settle circular dependencies */
static PyObject *Forthon_getattro(ForthonObject *self,PyObject *name);
static int Forthon_setattro(ForthonObject *self,PyObject *name,PyObject *v);
static PyMethodDef *getForthonPackage_methods(void);

/* ######################################################################### */
/* ######################################################################### */
/* ######################################################################### */
/* This variable is used to keep track of the total amount of memory         */
/* dynamically allocated in the package. */
static long totmembytes=0;

/* ######################################################################### */
/* Utility function used by attribute handling routines.                     */
/* It returns the index of the string v in the string s. If the string is    */
/* not found, it returns -1.                                                 */
static int strfind(char *v,char *s)
{
  int ls,lv,ind=0;
  ls = strlen(s);
  lv = strlen(v);
  while (ls >= lv) {
    if (strncmp(s,v,strlen(v))==0) return ind;
    s++;
    ind++;
    ls--;
    }
  return -1;
}
  
static double cputime(void)
{
  struct tms usage;
  long hardware_ticks_per_second;
  (void) times(&usage);
  hardware_ticks_per_second = sysconf(_SC_CLK_TCK);
  return (double) usage.tms_utime/hardware_ticks_per_second;
}

/* ###################################################################### */
/* Builds a scalar and an array dictionary for the package. The           */
/* dictionaries are then used in the getattr and setattr to look up the   */
/* indices given a variable name. That lookup is faster than a linear     */
/* scan through the list of variables.                                    */
static void Forthon_BuildDicts(ForthonObject *self)
{
  int i;
  PyObject *sdict,*adict,*iobj;
  sdict = PyDict_New();
  adict = PyDict_New();
  for (i=0;i<self->nscalars;i++) {
    iobj = Py_BuildValue("i",i);
    PyDict_SetItemString(sdict,self->fscalars[i].name,iobj);
    Py_DECREF(iobj);
    }
  for (i=0;i<self->narrays;i++) {
    iobj = Py_BuildValue("i",i);
    PyDict_SetItemString(adict,self->farrays[i].name,iobj);
    Py_DECREF(iobj);
    }
  self->scalardict = sdict;
  self->arraydict = adict;
}
static void Forthon_DeleteDicts(ForthonObject *self)
{
  Py_XDECREF(self->scalardict);
  Py_XDECREF(self->arraydict);
}

/* ######################################################################### */
/* # Update the data element of a derived type.                              */
static void ForthonPackage_updatederivedtype(ForthonObject *self,long i,
                                             long createnew)
{
  ForthonObject *objid;
  PyObject *oldobj;
  if (self->fscalars[i].type == PyArray_OBJECT && self->fscalars[i].dynamic) {
    /* If dynamic, use getpointer to get the current address of the */
    /* python object from the fortran variable. */
    /* This is needed since the association may have changed in fortran. */
    (self->fscalars[i].getpointer)(&objid,self->fobj,&createnew);
    /* If the address has changed, that means that a reassignment was done */
    /* in fortran. The data needs to be updated and the reference */
    /* count possibly incremented. */
    if (self->fscalars[i].data != (char *)objid) {
      oldobj = (PyObject *)self->fscalars[i].data;
      /* Make sure that the correct python object is pointed to. */
      /* The pointer is redirected before the DECREF is done to avoid */
      /* infinite loops. */
      self->fscalars[i].data = (char *)objid;
      Py_XDECREF(oldobj);
      /* Increment the reference count since a new thing points to it. */
      Py_XINCREF((PyObject *)self->fscalars[i].data);
      }
    }
}

/* ######################################################################### */
/* # Update the data element of a dynamic, fortran assignable array.         */
/* ------------------------------------------------------------------------- */
static int dimensionsmatch(Fortranarray *farray)
{
  int i;
  int result = 1;
  for (i=0;i<farray->nd;i++) {
    if (farray->dimensions[i] != farray->pya->dimensions[farray->nd-1-i])
      result = 0;}
  return result;
}
/* ------------------------------------------------------------------------- */
static void ForthonPackage_updatearray(ForthonObject *self,long i)
{
  Fortranarray *farray = &(self->farrays[i]);
  /* If the getpointer routine exists, call it to assign a value to data.s */
  if (farray->getpointer != NULL) {
    (farray->getpointer)(farray,self->fobj);
    /* If the data.s is NULL, then the fortran array is not associated. */
    /* Decrement the python object counter if there is one. */
    /* Set the pointer to the python object to NULL. */
    if (farray->data.s == NULL) {
      if (farray->pya != NULL) {Py_XDECREF(farray->pya);}
      farray->pya = NULL;}
    else if (farray->pya == NULL ||
             farray->data.s != farray->pya->data ||
             !dimensionsmatch(farray)) {
      /* If data.s is not NULL and there is no python object or its */
      /* data is different, then create a new one. */
      if (farray->pya != NULL) {Py_XDECREF(farray->pya);}
      farray->pya = (PyArrayObject *)PyArray_FromDimsAndData(
                       farray->nd,farray->dimensions,
                       farray->type,farray->data.s);
      /* Reverse the order of the dims and strides so       */
      /* indices can be refered to in the correct (fortran) */
      /* order in python                                    */
      ARRAY_REVERSE_STRIDE(farray->pya);
      ARRAY_REVERSE_DIM(farray->pya);
    }
  }
}

/* ######################################################################### */
/* # Allocate the dimensions element for all of the farrays in the object.   */
static void ForthonPackage_allotdims(ForthonObject *self)
{
  int i;
  for (i=0;i<self->narrays;i++) {
    self->farrays[i].dimensions=(int *)malloc(self->farrays[i].nd*sizeof(int));
    if (self->farrays[i].dimensions == NULL) {
      printf("Failure allocating space for array dimensions.\n");
      exit(EXIT_FAILURE);
      }
    /* Fill the dimensions with zeros. This is only needed for arrays with */
    /* unspecified shape, since setdims won't fill the dimensions. */
    memset(self->farrays[i].dimensions,0,self->farrays[i].nd*sizeof(int));
    }
}

/* ######################################################################### */
/* # Static array initialization routines */
static void ForthonPackage_staticarrays(ForthonObject *self)
{
  int i;
  char *c;

  /* Call the routine which sets the dimensions */
  (*(self->setstaticdims))(self);

  for (i=0;i<self->narrays;i++) {
    if (!self->farrays[i].dynamic) {
      /* Allocate the space */
      Py_XDECREF(self->farrays[i].pya);
      self->farrays[i].pya = (PyArrayObject *)PyArray_FromDimsAndData(
                       self->farrays[i].nd,self->farrays[i].dimensions,
                       self->farrays[i].type,self->farrays[i].data.s);
      /* Check if the allocation was unsuccessful. */
      if (self->farrays[i].pya==NULL) {
        printf("Failure creating python object for static array.\n");
        exit(EXIT_FAILURE);}
      /* Reverse the order of the dims and strides so indices */
      /* can be refered to in the correct (fortran) order in */
      /* python */
      ARRAY_REVERSE_STRIDE(self->farrays[i].pya);
      ARRAY_REVERSE_DIM(self->farrays[i].pya);
      /* For strings, replace nulls with blank spaces */
      if (self->farrays[i].type == PyArray_CHAR)
        if ((c=memchr(self->farrays[i].data.s,0,
                     PyArray_SIZE(self->farrays[i].pya))))
          memset(c,(int)' ',
                 (int)(PyArray_SIZE(self->farrays[i].pya)-(long)c+
                 (long)self->farrays[i].data.s));
      /* Add the array size to totmembytes. */
      totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya);
      }
    }
}

/* ######################################################################### */
/* # Get attribute handlers                                                  */
static PyObject *Forthon_getscalardouble(ForthonObject *self,void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  return Py_BuildValue("d",*((double *)(fscalar->data)));
}
/* ------------------------------------------------------------------------- */
static PyObject *Forthon_getscalarcdouble(ForthonObject *self,void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  return PyComplex_FromDoubles(((double *)fscalar->data)[0],
                               ((double *)fscalar->data)[1]);
}
/* ------------------------------------------------------------------------- */
static PyObject *Forthon_getscalarinteger(ForthonObject *self,void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  return Py_BuildValue("l",*((long *)(fscalar->data)));
}
/* ------------------------------------------------------------------------- */
static PyObject *Forthon_getscalarderivedtype(ForthonObject *self,void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  ForthonObject *objid;
  long createnew=1;
  /* These are attached to variables of fortran derived type */
  ForthonPackage_updatederivedtype(self,(long)closure,createnew);
  objid = (ForthonObject *)fscalar->data;
  if (objid != NULL) {
    Py_INCREF(objid);
    return (PyObject *)objid;}
  else {
    PyErr_SetString(ErrorObject,"variable unassociated");
    return NULL;}
}
/* ------------------------------------------------------------------------- */
static PyObject *Forthon_getarray(ForthonObject *self,void *closure)
{
  Fortranarray *farray = &(self->farrays[(long)closure]);
  /* Update the array if it is dynamic and fortran assignable. */
  ForthonPackage_updatearray(self,(long)closure);
  /* Increment the python object counter to prepare handing it to the */
  /* interpreter. */
  if (farray->pya == NULL) {
    PyErr_SetString(ErrorObject,"Array is unallocated");
    return NULL;}
  Py_XINCREF(farray->pya);
  if (farray->pya->nd==1 &&
      farray->pya->strides[0]==farray->pya->descr->elsize)
    farray->pya->flags |= CONTIGUOUS;
  return (PyObject *)farray->pya;
}
/* ------------------------------------------------------------------------- */
static PyObject *Forthon_getscalardict(ForthonObject *self,void *closure)
{
  Py_INCREF(self->scalardict);
  return self->scalardict;
}
/* ------------------------------------------------------------------------- */
static PyObject *Forthon_getarraydict(ForthonObject *self,void *closure)
{
  Py_INCREF(self->arraydict);
  return self->arraydict;
}

/* ######################################################################### */
/* Memory allocation routines */
static int Forthon_freearray(ForthonObject *self,void *closure)
{
  Fortranarray *farray = &(self->farrays[(long)closure]);

  if (farray->dynamic) {
    if (farray->pya != NULL) {
      /* Subtract the array size from totmembytes. */
      totmembytes -= (long)PyArray_NBYTES(farray->pya);
      Py_XDECREF(farray->pya);
      farray->pya = NULL;
%py_ifelse(f90 and not f90f,0,'*(farray->data.d)=0;','')
      /* Note that the dimensions passed in are in the wrong (C) order, but */
      /* here it doesn't matter since the array pointer is just be zero. */
      /* This probably should be done correctly just to avoid any error. */
%py_ifelse(f90 and not f90f,1,'(farray->setpointer)(0,(self->fobj),farray->dimensions);','')
      }
    }
  return 0;
}

/* ######################################################################### */
/* # Set attribute handlers                                                  */
static int Forthon_setscalardouble(ForthonObject *self,PyObject *value,
                                   void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  double lv;
  int e;
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
    return -1;}
  e = PyArg_Parse(value,"d",&lv);
  if (e) {
    memcpy((fscalar->data),&lv,sizeof(double));}
  else {
    PyErr_SetString(ErrorObject,"Right hand side has incorrect type");
    return -1;}
  return 0;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setscalarcdouble(ForthonObject *self,PyObject *value,
                                    void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  Py_complex lv;
  int e;
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
    return -1;}
  e = PyArg_Parse(value,"D",&lv);
  if (e) {
    memcpy((fscalar->data),&lv,2*sizeof(double));}
  else {
    PyErr_SetString(ErrorObject,"Right hand side has incorrect type");
    return -1;}
  return 0;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setscalarinteger(ForthonObject *self,PyObject *value,
                                    void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  long lv;
  int e;
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
    return -1;}
  e = PyArg_Parse(value,"l",&lv);
  if (e) {
    memcpy((fscalar->data),&lv,sizeof(long));}
  else {
    PyErr_SetString(ErrorObject,"Right hand side has incorrect type");
    return -1;}
  return 0;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setscalarderivedtype(ForthonObject *self,PyObject *value,
                                        void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  void *d;
  long createnew;
  PyObject *oldobj;

  /* Only create a new instance if a non-NULL value is passed in. */
  /* With a NULL value, the object will be decref'ed so there's no */
  /* point creating a new one. */
  createnew = (value != NULL);
  ForthonPackage_updatederivedtype(self,(long)closure,createnew);

  if (value == NULL) {
    if (fscalar->dynamic) {
      if (fscalar->data != NULL) {
        /* Decrement the reference counter and nullify the fortran pointer. */
        oldobj = (PyObject *)fscalar->data;
        d = (void *)((ForthonObject *)fscalar->data)->fobjdeallocate;
        if (d != NULL)
          (fscalar->setpointer)(0,(self->fobj));
        fscalar->data = NULL;
        Py_DECREF(oldobj);
        }
      return 0;
      }
    else {
      PyErr_SetString(PyExc_TypeError,
                      "Cannot delete a static derived type object");
      return -1;
      }
    }

  if (strcmp("Forthon",value->ob_type->tp_name) != 0 ||
      strcmp(((ForthonObject *)value)->typename,fscalar->typename) != 0) {
    PyErr_SetString(ErrorObject,"Right hand side has incorrect type");
    return -1;}
  Py_INCREF(value);
  Py_XDECREF((PyObject *)fscalar->data);
  fscalar->data = (char *)value;
  (fscalar->setpointer)(((ForthonObject *)value)->fobj,(self->fobj));
  return 0;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setarray(ForthonObject *self,PyObject *value,
                            void *closure)
{
  Fortranarray *farray = &(self->farrays[(long)closure]);
  int j,k,r,d,setit;
  PyObject *pyobj;
  PyArrayObject *ax;

  if (value == NULL) {
    if (farray->dynamic) {
      /* Deallocate the dynamic array. */
      r = Forthon_freearray(self,closure);
      return r;
      }
    else {
      PyErr_SetString(PyExc_TypeError, "Cannot delete a static array");
      return -1;
      }
    }

  PyArg_Parse(value, "O", &pyobj);
  FARRAY_FROMOBJECT(ax,pyobj,farray->type);
  if ((farray->dynamic && ax->nd == farray->nd) ||
      (farray->dynamic == 3 && farray->nd == 1 && ax->nd == 0 &&
       farray->pya == NULL)) {
    /* The long list of checks above looks for the special case of assigning */
    /* a scalar to a 1-D deferred-shape array that is unallocated. In that   */
    /* case, a 1-D array is created with the value of the scalar. If the     */
    /* array is already allocated, the code below broadcasts the scalar over */
    /* the array. */
    if (farray->dynamic == 3) {
      /* This type of dynamic array (deferred-shape) does not have the */
      /* dimensions specified so they can take on whatever is input. */
      for (j=0;j<ax->nd;j++) {
        k = ax->nd - j - 1;
        farray->dimensions[j] = ax->dimensions[k];
        }
      }
      if (ax->nd == 0) {
        /* Special handling is needed when a 0-D array is assigned to a      */
        /* 1-D deferred-shape array. The Numeric routine used by             */
        /* FARRAY_FROMOBJECT returns a 0-D array when the input is a scalar. */
        /* This can cause problems elsewhere, so it is replaced with a 1-D   */
        /* array of length 1.                                                */
        farray->dimensions[0] = 1;
        Py_XDECREF(ax);
        ax = (PyArrayObject *)PyArray_FromDims(1,farray->dimensions,
                                               farray->type);
        ax->descr->setitem(pyobj, ax->data);
        }
    else {
      /* Call the routine which sets the dimensions */
      /* Note that this sets the dimensions for everything in the group. */
      /* This may cause some slow down, but is hard to get around. */
      (*self->setdims)(farray->group,self);
      }
    setit = 1;
    for (j=0;j<ax->nd;j++) {
      k = ax->nd - j - 1;
      if (ax->dimensions[j] != farray->dimensions[k])
        setit=0;
      }
    if (setit) {
      if (farray->pya != NULL) {Py_XDECREF(farray->pya);}
      farray->pya = ax;
      /* Note that pya->dimensions are in the correct fortran order, but */
      /* farray->dimensions are in C order. */
%py_ifelse(f90 and not f90f,1,'(farray->setpointer)((farray->pya)->data,(self->fobj),(farray->pya)->dimensions);','')
%py_ifelse(f90 and not f90f,0,'*(farray->data.d)=(farray->pya)->data;','')
      r = 0;}
    else {
      r = -1;
      Py_XDECREF(ax);
      PyErr_SetString(ErrorObject,
                      "Right hand side has incorrect dimensions");}}
  else {
    /* Update the array if it is dynamic and fortran assignable. */
    ForthonPackage_updatearray(self,(long)closure);
    /* At this point, the array must already have been allocated */
    if (farray->pya == NULL) {
      Py_XDECREF(ax);
      PyErr_SetString(ErrorObject,"Array is unallocated");
      return -1;}
    /* For strings, allow the length of the input to be   */
    /* different than the array. Before the copy, force   */
    /* first dimensions to be the same so the copy works. */
    /* If the input is shorter than the variable, then    */
    /* overwrite the rest of the array with spaces.       */
    d = -1;
    if (farray->type == PyArray_CHAR && ax->nd > 0) {
      if (ax->dimensions[0] < farray->pya->dimensions[0]){
        memset(farray->pya->data+ax->dimensions[0],(int)' ',
               PyArray_SIZE(farray->pya)-ax->dimensions[0]);
        d = farray->pya->dimensions[0];
        farray->pya->dimensions[0] = ax->dimensions[0];}
      else
        {ax->dimensions[0] = farray->pya->dimensions[0];}}
    /* Copy input data into the array. This does the copy */
    /* for static arrays and also does any broadcasting   */
    /* when the dimensionality of the input is different  */
    /* than the array.                                    */
    r = PyArray_CopyArray(farray->pya,ax);
    /* Reset the value of the first dimension if it was   */
    /* changed to accomodate a string.                    */
    if (d > -1) farray->pya->dimensions[0] = d;
    Py_XDECREF(ax);}
  return r;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setscalardict(ForthonObject *self,void *closure)
{
  PyErr_SetString(PyExc_TypeError, "Cannot set the scalardict attribute");
  return -1;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setarraydict(ForthonObject *self,void *closure)
{
  PyErr_SetString(PyExc_TypeError, "Cannot set the arraydict attribute");
  return -1;
}

/* ------------------------------------------------------------------------- */
static int Forthon_traverse(ForthonObject *self,visitproc visit,void *arg)
{
  int i;
  long createnew=0;
  for (i=0;i<self->nscalars;i++) {
    if (self->fscalars[i].type == PyArray_OBJECT &&
        self->fscalars[i].dynamic &&
        strcmp(self->typename,self->fscalars[i].typename) != 0) {
      ForthonPackage_updatederivedtype(self,i,createnew);
      if (self->fscalars[i].data != NULL)
        return visit((PyObject *)(self->fscalars[i].data), arg);
      }
    }
  return 0;
}


/* ------------------------------------------------------------------------- */
static int Forthon_clear(ForthonObject *self)
{
  /* Note that this is called by Forthon_dealloc. */
  int i;
  long createnew=0;
  void *d;
  PyObject *oldobj;

  for (i=0;i<self->nscalars;i++) {
    if (self->fscalars[i].type == PyArray_OBJECT)
      {
      ForthonPackage_updatederivedtype(self,i,createnew);
      if (self->fscalars[i].data != NULL) {
        d = (void *)((ForthonObject *)self->fscalars[i].data)->fobjdeallocate;
        oldobj = (PyObject *)self->fscalars[i].data;
        self->fscalars[i].data = NULL;
        if (d != NULL)
          (self->fscalars[i].setpointer)(0,(self->fobj));
        /* Only delete the object after deleting references to it. */
        Py_DECREF(oldobj);
        }
      }
    }
  for (i=0;i<self->narrays;i++) {
    free(self->farrays[i].dimensions);
    Py_XDECREF(self->farrays[i].pya);
    }
  if (self->fobj != NULL) {
    /* Note that for package instance (as opposed to derived type */
    /* instances), the fscalars and farrays are statically defined and */
    /* can't be freed. */
    if (self->fscalars != NULL) free(self->fscalars);
    if (self->farrays  != NULL) free(self->farrays);
    }
  if (self->fobj != NULL) {
    if (self->fobjdeallocate != NULL) {(self->fobjdeallocate)(self->fobj);}
    else                              {(self->nullifycobj)(self->fobj);}
    }

  Forthon_DeleteDicts(self);
  return 0;
}

/* ######################################################################### */
/* ######################################################################### */
/* ######################################################################### */
/* ######################################################################### */
/* The following are all callable from python                                */
/* ######################################################################### */
/* ######################################################################### */
/* ######################################################################### */
/* ######################################################################### */

/* ######################################################################### */
static char allocated_doc[] = "Checks whether a dynamic variable is allocated. If a static array or a scalar is passed, it just returns true.";
static PyObject *ForthonPackage_allocated(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i;
  long createnew=1;
  char *name;
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* Check for scalar of derived type which could be dynamic */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    if (self->fscalars[i].type == PyArray_OBJECT) {
      ForthonPackage_updatederivedtype(self,i,createnew);
      if (self->fscalars[i].data == NULL) {return Py_BuildValue("i",0);}
      else {
        return Py_BuildValue("i",
                      ((ForthonObject *)(self->fscalars[i].data))->allocated);
        }
    }}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    if (self->farrays[i].pya == NULL) {return Py_BuildValue("i",0);}
    else                              {return Py_BuildValue("i",1);}
    }

  return Py_BuildValue("i",1);
}

/* ######################################################################### */
static char getdict_doc[] = "Builds a dictionary, including every variable in the package. For arrays, the dictionary value points to the same memory location as the fortran. If a dictionary is input, then that one is updated rather then creating a new one.";
static PyObject *ForthonPackage_getdict(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  long j;
  PyObject *dict=NULL;
  PyObject *v,*n;
  Fortranscalar *s;
  Fortranarray *a;
  if (!PyArg_ParseTuple(args,"|O",&dict)) return NULL;
  if (dict == NULL) {
    dict = PyDict_New();}
  else {
    if (!PyDict_Check(dict)) {
      PyErr_SetString(ErrorObject,"Optional argument must be a dictionary.");
      return NULL;}
    }
  for (j=0;j<self->nscalars;j++) {
    s = self->fscalars + j;
    if (s->type == PyArray_DOUBLE) {
      v = Forthon_getscalardouble(self,(void *)j);}
    else if (s->type == PyArray_CDOUBLE) {
      v = Forthon_getscalarcdouble(self,(void *)j);}
    else if (s->type == PyArray_OBJECT) {
      v = Forthon_getscalarderivedtype(self,(void *)j);}
    else {
      v = Forthon_getscalarinteger(self,(void *)j);}
    if (v != NULL) {
      n = Py_BuildValue("s",s->name);
      PyDict_SetItem(dict,n,v);
      Py_DECREF(n);
      Py_DECREF(v);}
    else {
      PyErr_Clear();}
    }
  for (j=0;j<self->narrays;j++) {
    v = Forthon_getarray(self,(void *)j);
    if (v != NULL) {
      a = self->farrays + j;
      n = Py_BuildValue("s",a->name);
      PyDict_SetItem(dict,n,v);
      Py_DECREF(n);
      }
    else {
      PyErr_Clear();}
    }
  return dict;
}

/* ######################################################################### */
static char deprefix_doc[] = "For each variable in the package, a python object is created which has the same name and same value. For arrays, the new objects points to the same memory location.";
static PyObject *ForthonPackage_deprefix(PyObject *_self_,PyObject *args)
{
  /* ForthonObject *self = (ForthonObject *)_self_; */
  PyObject *m,*d, *a;
  if (!PyArg_ParseTuple(args,"")) return NULL;
  /* printf("deprefixing %s, please wait\n",self->name); */
  m = PyImport_AddModule("__main__");
  d = PyModule_GetDict(m);
  a = PyTuple_New(1);
  PyTuple_SET_ITEM(a,0,d);
  ForthonPackage_getdict(_self_,a);
  /* The reference count on d must be incremented since the Tuple steals */
  /* a reference, but on deletion decrements the reference count. */
  Py_INCREF(d);
  Py_DECREF(a);
  /* printf("done deprefixing %s\n",self->name); */
  returnnone;
}

/* ######################################################################### */
static char getfunctions_doc[] = "Builds a dictionary containing all of the functions in the package.";
static PyObject *ForthonPackage_getfunctions(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *dict,*name;
  PyMethodDef *ml;
  if (!PyArg_ParseTuple(args,"")) return NULL;
  dict = PyDict_New();
  ml = getForthonPackage_methods();
  for (; ml->ml_name != NULL; ml++) {
    name = Py_BuildValue("s",ml->ml_name);
    PyDict_SetItem(dict,name,(PyObject *)PyCFunction_New(ml, _self_));
    Py_DECREF(name);
    }
  ml = self->fmethods;
  for (; ml->ml_name != NULL; ml++) {
    name = Py_BuildValue("s",ml->ml_name);
    PyDict_SetItem(dict,name,(PyObject *)PyCFunction_New(ml, _self_));
    Py_DECREF(name);
    }
  return dict;
}

/* ######################################################################### */
/* # Create routine to force assignment of arrays                            */
/* The routines forces assignment of arrays. It takes two                    */
/* arguments, a string (the array name) and a PyArray_object.                */
/* For dynamic arrays, the array is pointed to the input array               */
/* regardless of its dimension sizes. For static arrays, a copy              */
/* is done similar to that done in gchange, where what ever                  */
/* fits into the space is copied. The main purpose of this                   */
/* routine is to allow a restore from a dump file to work.                   */
static char forceassign_doc[] = "Forces assignment to a dynamic array, resizing it if necessary";
static PyObject *ForthonPackage_forceassign(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  int i,j,r=-1;
  int *d,*pyadims,*axdims;
  PyObject *pyobj;
  PyArrayObject *ax;
  PyObject *pyi;
  char *name;
  if (!PyArg_ParseTuple(args,"sO",&name,&pyobj)) return NULL;

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    FARRAY_FROMOBJECT(ax,pyobj,self->farrays[i].type);
    if (self->farrays[i].dynamic && ax->nd == self->farrays[i].nd) {
      if (self->farrays[i].pya != NULL)
        totmembytes -= (long)PyArray_NBYTES(self->farrays[i].pya);
      Py_XDECREF(self->farrays[i].pya);
      self->farrays[i].pya = ax;
      /* Note that pya->dimensions are in the correct fortran order, but */
      /* farray->dimensions are in C order. */
%py_ifelse(f90 and not f90f,1,'(self->farrays[i].setpointer)((self->farrays[i].pya)->data,(self->fobj),(self->farrays[i].pya)->dimensions);','')
%py_ifelse(f90 and not f90f,0,'*(self->farrays[i].data.d)=(self->farrays[i].pya)->data;','')
      totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya);
      returnnone;}
    else if (ax->nd == self->farrays[i].nd) {
      /* Copy input data into the array. This does a copy   */
      /* even if the dimensions do not match. If an input   */
      /* dimension is larger, the extra data is truncated.  */
      /* This code ensures that the dimensions of ax        */
      /* remain intact since there may be other references  */
      /* to it.                                             */
      d = (int *)malloc(self->farrays[i].nd*sizeof(int));
      for (j=0;j<ax->nd;j++) {
        if (self->farrays[i].pya->dimensions[j] < ax->dimensions[j]) {
          d[j] = self->farrays[i].pya->dimensions[j];}
        else {
          d[j] = ax->dimensions[j];}
        }
      pyadims = self->farrays[i].pya->dimensions;
      axdims = ax->dimensions;
      self->farrays[i].pya->dimensions = d;
      ax->dimensions = d;
      r = PyArray_CopyArray(self->farrays[i].pya,ax);
      self->farrays[i].pya->dimensions = pyadims;
      ax->dimensions = axdims;
      free(d);
      Py_XDECREF(ax);
      if (r == 0) {
        returnnone;}
      else {
        return NULL;}
      }
    else {
      PyErr_SetString(ErrorObject,
                  "Both arguments must have the same number of dimensions");
      return NULL;}}
  PyErr_SetString(ErrorObject,"First argument must be an array");
  return NULL;
}

/* ######################################################################### */
/* # Group allocation routine                                                */
/* # The following three routines are callable as attributes of a package    */
/* The dimensions for the dynamic arrays are set in a routine which is       */
/* specific to each package.                                                 */
static char gallot_doc[] = "Allocates all dynamic arrays in a group";
static PyObject *ForthonPackage_gallot(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  char *s=NULL;
  int i,j,r=0,allotit,iverbose=0;
  PyObject *star;
  if (!PyArg_ParseTuple(args,"|si",&s,&iverbose)) return NULL;
  self->allocated = 1;
  if (s == NULL) s = "*";

  /* Check for any scalars of derived type. These must also be allocated */
  for (i=0;i<self->nscalars;i++) {
    if (strcmp(s,self->fscalars[i].group)==0 || strcmp(s,"*")==0) {
      if (!(self->fscalars[i].dynamic)) {
        if (self->fscalars[i].type == PyArray_OBJECT &&
            self->fscalars[i].data != NULL) {
          r = 1;
          star = Py_BuildValue("(s)","*");
          ForthonPackage_gallot((PyObject *)self->fscalars[i].data,star);
          Py_DECREF(star);
      }}}}

  /* Call the routine which sets the dimensions */
  (*self->setdims)(s,self);

  /* Now Process the arrays now that the dimensions are set */
  for (i=0;i<self->narrays;i++) {
   if (strcmp(s,self->farrays[i].group)==0 || strcmp(s,"*")==0) {
    r = 1;
    /* Note that deferred-shape arrays shouldn't be allocated in this way */
    /* since they have no specified dimensions. */
    if (self->farrays[i].dynamic && self->farrays[i].dynamic != 3) {
      /* Subtract the array size from totmembytes. */
      if (self->farrays[i].pya != NULL)
        totmembytes -= (long)PyArray_NBYTES(self->farrays[i].pya);
      /* Always dereference the previous value so array can become */
      /* Unallocated if it has dimensions <= 0. */
      Py_XDECREF(self->farrays[i].pya);
      self->farrays[i].pya = NULL;
      /* Make sure the dimensions are all greater then zero. */
      /* If not, then don't allocate the array. */
      allotit = 1;
      for (j=0;j<self->farrays[i].nd;j++)
        if (self->farrays[i].dimensions[j] <= 0) allotit = 0;
      if (allotit) {
        /* Use array routine to create space */
        /* Note that in the call to setpointer, the dimensions are in the */
        /* wrong order. It is not fixed since the f90f is not really supported*/
        /* nor needed. */
%py_ifelse(f90f,0,'self->farrays[i].pya = (PyArrayObject *)PyArray_FromDims(self->farrays[i].nd,self->farrays[i].dimensions,self->farrays[i].type);')
%py_ifelse(f90f,1,'(self->farrays[i].setpointer)(&(self->farrays[i]),(self->fobj),self->farrays[i].dimensions);')
%py_ifelse(f90f,1,'self->farrays[i].pya = (PyArrayObject *)PyArray_FromDimsAndData(self->farrays[i].nd,self->farrays[i].dimensions,self->farrays[i].type,self->farrays[i].data.s);')
        /* Check if the allocation was unsuccessful. */
        if (self->farrays[i].pya==NULL) {
          long arraysize=1;
          for (j=0;j<self->farrays[i].nd;j++)
            arraysize *= self->farrays[i].dimensions[j];
          printf("GALLOT: allocation failure for %s to size %ld\n",
                 self->farrays[i].name,arraysize);
          exit(EXIT_FAILURE);
          }
        /* Reverse the order of the dims and strides so indices */
        /* can be refered to in the correct (fortran) order in  */
        /* python                                               */
        ARRAY_REVERSE_STRIDE(self->farrays[i].pya);
        ARRAY_REVERSE_DIM(self->farrays[i].pya);
        /* Point fortran pointer to new space */
        /* Note that pya->dimensions are in the correct fortran order, but */
        /* farray->dimensions are in C order. */
%py_ifelse(f90 and not f90f,0,'*(self->farrays[i].data.d)=(self->farrays[i].pya)->data;','')
%py_ifelse(f90 and not f90f,1,'(self->farrays[i].setpointer)((self->farrays[i].pya)->data,(self->fobj),(self->farrays[i].pya)->dimensions);','')
        /* Fill array with initial value. A check could probably be made */
        /* of whether the initial value is zero since the initialization */
        /* doesn't need to be done then. Not having the check gaurantees */
        /* that it is set correctly, but is slower. */
        if (self->farrays[i].type == PyArray_CHAR) {
          memset((self->farrays[i].pya)->data,(int)' ',
                 PyArray_SIZE(self->farrays[i].pya));
          }
        else if (self->farrays[i].type == PyArray_LONG) {
          for (j=0;j<PyArray_SIZE(self->farrays[i].pya);j++)
            *((long *)((self->farrays[i].pya)->data)+j) = self->farrays[i].initvalue;
          }
        else if (self->farrays[i].type == PyArray_DOUBLE) {
          for (j=0;j<PyArray_SIZE(self->farrays[i].pya);j++)
            *((double *)((self->farrays[i].pya)->data)+j) = self->farrays[i].initvalue;
          }
        /* Add the array size to totmembytes. */
        totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya);
        if (iverbose) printf("%s.%s %d\n",self->name,self->farrays[i].name,
                                       PyArray_SIZE(self->farrays[i].pya));
        }
      }
    }
  }

  /* If a variable was found, returns 1, otherwise returns 0. */
  return Py_BuildValue("i",r);
}

/* ######################################################################### */
/* # Group allocation change routine */
static char gchange_doc[] = "Changes allocation of all dynamic arrays in a group if needed";
static PyObject *ForthonPackage_gchange(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  char *s=NULL;
  int i,r=0;
  PyArrayObject *ax;
  int j,rt,changeit,freeit,iverbose=0;
  int *d,*pyadims,*axdims;
  PyObject *star;

  if (!PyArg_ParseTuple(args,"|si",&s,&iverbose)) return NULL;
  self->allocated = 1;
  if (s == NULL) s = "*";

  /* Check for any scalars of derived type. These must also be allocated */
  for (i=0;i<self->nscalars;i++) {
    if (strcmp(s,self->fscalars[i].group)==0 || strcmp(s,"*")==0) {
      if (!(self->fscalars[i].dynamic)) {
        if (self->fscalars[i].type == PyArray_OBJECT &&
            self->fscalars[i].data != NULL) {
          r = 1;
          star = Py_BuildValue("(s)","*");
          ForthonPackage_gchange((PyObject *)self->fscalars[i].data,star);
          Py_DECREF(star);
      }}}}

  /* Call the routine which sets the dimensions */
  (*self->setdims)(s,self);

  /* Now Process the arrays now that the dimensions are set */
  for (i=0;i<self->narrays;i++) {
   if (strcmp(s,self->farrays[i].group)==0 || strcmp(s,"*")==0) {
    r = 1;
    if (self->farrays[i].dynamic) {
      /* Check if any of the dimensions have changed or if array is */
      /* unallocated. In either case, change it. */
      changeit = 0;
      if (self->farrays[i].pya == NULL) {
        changeit = 1;}
      else {
        /* Note that self->farrays[i].dimensions are in C order while */
        /* self->farrays[i].pya->dimensions are in fortran order.     */
        for (j=0;j<self->farrays[i].nd;j++) {
          if (self->farrays[i].dimensions[j] !=
              self->farrays[i].pya->dimensions[self->farrays[i].nd-j-1])
            changeit = 1;}
        /* Subtract the array size from totmembytes. */
        if (changeit) totmembytes -= (long)PyArray_NBYTES(self->farrays[i].pya);
        }
      /* Make sure all of the dimensions are >= 0. If not, then free it. */
      freeit = 0;
      for (j=0;j<self->farrays[i].nd;j++)
        if (self->farrays[i].dimensions[j] <= 0) freeit = 1;
      if (freeit) {
        Py_XDECREF(self->farrays[i].pya);
        self->farrays[i].pya = NULL;}
      /* Only allocate new space and copy old data if */
      /* any dimensions are different. */
      if (changeit && !freeit) {
        /* Use array routine to create space */
        ax = (PyArrayObject *)PyArray_FromDims(self->farrays[i].nd,
                            self->farrays[i].dimensions,self->farrays[i].type);
        /* Check if the allocation was unsuccessful. */
        if (ax==NULL) {
          long arraysize=1;
          for (j=0;j<self->farrays[i].nd;j++)
            arraysize *= self->farrays[i].dimensions[j];
          printf("GCHANGE: allocation failure for %s to size %ld\n",
                 self->farrays[i].name,arraysize);
          exit(EXIT_FAILURE);
          }
        /* Reverse the order of the dims and strides so       */
        /* indices can be refered to in the correct (fortran) */
        /* order in python                                    */
        ARRAY_REVERSE_STRIDE(ax);
        ARRAY_REVERSE_DIM(ax);
        /* Fill array with initial value. A check could probably be made */
        /* of whether the initial value is zero since the initialization */
        /* doesn't need to be done then. Not having the check gaurantees */
        /* that it is set correctly, but is slower. */
        if (self->farrays[i].type == PyArray_CHAR) {
          memset(ax->data,(int)' ',PyArray_SIZE(ax));
          }
        else if (self->farrays[i].type == PyArray_LONG) {
          for (j=0;j<PyArray_SIZE(ax);j++)
            *((long *)(ax->data)+j) = self->farrays[i].initvalue;
          }
        else if (self->farrays[i].type == PyArray_DOUBLE) {
          for (j=0;j<PyArray_SIZE(ax);j++)
            *((double *)(ax->data)+j) = self->farrays[i].initvalue;
          }
        /* Copy the existing data to the new space. The       */
        /* minimum of each dimension is found and put into    */
        /* the old arrays dimensions. The new arrays          */
        /* dimensions are then temporarily set to the         */
        /* minimums so that the copy will work.  Note that    */
        /* the strides are still different though. The        */
        /* minimums are used to avoid writing beyond a        */
        /* dimension that may have been reduced.              */
        /* This code ensures that the dimensions of pya       */
        /* remain intact since there may be other references  */
        /* to it.                                             */
        if (self->farrays[i].pya != NULL) {
          d = (int *)malloc(self->farrays[i].nd*sizeof(int));
          for (j=0;j<self->farrays[i].nd;j++) {
            if (ax->dimensions[j] < self->farrays[i].pya->dimensions[j]) {
              d[j] = ax->dimensions[j];}
            else {
              d[j] = self->farrays[i].pya->dimensions[j];}
            }
          pyadims = self->farrays[i].pya->dimensions;
          axdims = ax->dimensions;
          self->farrays[i].pya->dimensions = d;
          ax->dimensions = d;
          rt=PyArray_CopyArray(ax,self->farrays[i].pya);
          self->farrays[i].pya->dimensions = pyadims;
          ax->dimensions = axdims;
          free(d);
          }
        /* Point pointers to new space. */
        Py_XDECREF(self->farrays[i].pya);
        self->farrays[i].pya = ax;
        /* Note that pya->dimensions are in the correct fortran order, but */
        /* farray->dimensions are in C order. */
%py_ifelse(f90 and not f90f,1,'(self->farrays[i].setpointer)((self->farrays[i].pya)->data,(self->fobj),(self->farrays[i].pya)->dimensions);','')
%py_ifelse(f90 and not f90f,0,'*(self->farrays[i].data.d)=(self->farrays[i].pya)->data;','')
        /* Add the array size to totmembytes. */
        totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya);
        if (iverbose) printf("%s.%s %d\n",self->name,self->farrays[i].name,
                                       PyArray_SIZE(self->farrays[i].pya));
        }
      }
    }
  }

  /* If a variable was found, returns 1, otherwise returns 0. */
  return Py_BuildValue("i",r);
}

/* ######################################################################### */
static char getfobject_doc[] = "Gets id to f object";
static PyObject *ForthonPackage_getfobject(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  if (!PyArg_ParseTuple(args,"")) return NULL;
  return Py_BuildValue("l",(long)self->fobj);
}

/* ######################################################################### */
static char getgroup_doc[] = "Returns group that a variable is in.";
static PyObject *ForthonPackage_getgroup(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i;
  char *name;
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    return Py_BuildValue("s",self->fscalars[i].group);}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    return Py_BuildValue("s",self->farrays[i].group);}

  PyErr_SetString(ErrorObject,"No such variable");
  return NULL;
}

/* ######################################################################### */
/* # Get the python object associated with a variable. This does exactly     */
/* # the same thing as getattr except that for unallocated objects, it       */
/* # returns None                                                            */
static char getpyobject_doc[] = "Returns the python object associated with a name, returns None is object is unallocated";
static PyObject *ForthonPackage_getpyobject(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *obj;
  PyObject *name;
  if (!PyArg_ParseTuple(args,"O",&name)) return NULL;

  obj = Forthon_getattro(self,name);

  if (obj == NULL && PyErr_Occurred() && PyErr_ExceptionMatches(ErrorObject)) {
    PyErr_Clear();
    returnnone;
    }
  else {
    return obj;
    }
}

/* ######################################################################### */
static char gettypename_doc[] = "Returns name of type of object.";
static PyObject *ForthonPackage_gettypename(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  if (!PyArg_ParseTuple(args,"")) return NULL;
  return Py_BuildValue("s",self->typename);
}

/* ######################################################################### */
/* # Set information about the variable name.                                */
static char addvarattr_doc[] = "addvarattr(varname,attr) Adds an attribute to a variable";
static PyObject *ForthonPackage_addvarattr(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i;
  char *name,*attr,*newattr;
  if (!PyArg_ParseTuple(args,"ss",&name,&attr)) return NULL;

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    newattr = (char *)malloc(strlen(self->fscalars[i].attributes) +
                             strlen(attr)+3);
    strcpy(newattr,self->fscalars[i].attributes);
    strcat(newattr," ");
    strcat(newattr,attr);
    strcat(newattr," ");
    /* The call to free is commented out intentionally. When the attributes */
    /* are first initialized, they are put in static memory and so can not  */
    /* be freed. Some check is really needed so that free is skipped the    */
    /* first time that this routine is called for a variable. The other     */
    /* option is to create the attribute memory different, explicitly using */
    /* malloc.  This is such a tiny memory leak without the free that the   */
    /* effort is not worth it.                                              */
    /* free(self->fscalars[i].attributes); */
    self->fscalars[i].attributes = newattr;
    returnnone;}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    newattr = (char *)malloc(strlen(self->farrays[i].attributes) +
                             strlen(attr)+3);
    memset(newattr,0,strlen(self->farrays[i].attributes) + strlen(attr)+2);
    strcpy(newattr,self->farrays[i].attributes);
    strcat(newattr," ");
    strcat(newattr,attr);
    strcat(newattr," ");
    /* The call to free is commented out intentionally. When the attributes */
    /* are first initialized, they are put in static memory and so can not  */
    /* be freed. Some check is really needed so that free is skipped the    */
    /* first time that this routine is called for a variable. The other     */
    /* option is to create the attribute memory different, explicitly using */
    /* malloc.  This is such a tiny memory leak without the free that the   */
    /* effort is not worth it.                                              */
    /* free(self->farrays[i].attributes); */
    self->farrays[i].attributes = newattr;
    returnnone;
    }

  PyErr_SetString(ErrorObject,"No such variable");
  return NULL;
}

/* ######################################################################### */
/* # Get information about the variable name.                                */
static char getvarattr_doc[] = "Returns the attributes of a variable";
static PyObject *ForthonPackage_getvarattr(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i;
  char *name;
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    return Py_BuildValue("s",self->fscalars[i].attributes);}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    return Py_BuildValue("s",self->farrays[i].attributes);}

  PyErr_SetString(ErrorObject,"No such variable");
  return NULL;
}

/* ######################################################################### */
/* # Set information about the variable name.                                */
static char setvarattr_doc[] = "Sets the attributes of a variable";
static PyObject *ForthonPackage_setvarattr(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i;
  char *name,*attr;
  if (!PyArg_ParseTuple(args,"ss",&name,&attr)) return NULL;

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    /* See comments in addvarattr why the free is commented out */
    /* free(self->fscalars[i].attributes); */
    self->fscalars[i].attributes = (char *)malloc(strlen(attr) + 1);
    strcpy(self->fscalars[i].attributes,attr);
    returnnone;}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    /* See comments in addvarattr why the free is commented out */
    /* free(self->farrays[i].attributes); */
    self->farrays[i].attributes = (char *)malloc(strlen(attr) + 1);
    strcpy(self->farrays[i].attributes,attr);
    returnnone;}

  PyErr_SetString(ErrorObject,"No such variable");
  return NULL;
}

/* ######################################################################### */
/* # Deletes information about the variable name.                            */
static char delvarattr_doc[] = "delvarattr(varname,attr) Deletes the specified attributes of a variable";
static PyObject *ForthonPackage_delvarattr(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i,ind;
  char *name,*attr,*newattr;
  if (!PyArg_ParseTuple(args,"ss",&name,&attr)) return NULL;

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    newattr = (char *)malloc(strlen(self->fscalars[i].attributes) -
                             strlen(attr) + 1);
    ind = strfind(attr,self->fscalars[i].attributes);
    /* Check if attr was found, and make sure it is surrounded by spaces. */
    if (ind == -1 ||
        (ind > 0 && self->fscalars[i].attributes[ind-1] != ' ') ||
        (ind < strlen(self->fscalars[i].attributes) &&
         self->fscalars[i].attributes[ind+strlen(attr)] != ' ')) {
      PyErr_SetString(ErrorObject,"Variable has no such attribute");
      return NULL;
      }
    strncpy(newattr,self->fscalars[i].attributes,ind);
    newattr[ind] = (char) NULL;
    if (ind+strlen(attr) < strlen(self->fscalars[i].attributes))
      strcat(newattr,self->fscalars[i].attributes+ind+strlen(attr));
    /* See comments in addvarattr why the free is commented out */
    /* free(self->fscalars[i].attributes); */
    self->fscalars[i].attributes = newattr;
    returnnone;}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    newattr = (char *)malloc(strlen(self->farrays[i].attributes) -
                             strlen(attr) + 1);
    ind = strfind(attr,self->farrays[i].attributes);
    /* Check if attr was found, and make sure it is surrounded by spaces. */
    if (ind == -1 ||
        (ind > 0 && self->farrays[i].attributes[ind-1] != ' ') ||
        (ind < strlen(self->farrays[i].attributes) &&
         self->farrays[i].attributes[ind+strlen(attr)] != ' ')) {
      PyErr_SetString(ErrorObject,"Variable has no such attribute");
      return NULL;
      }
    strncpy(newattr,self->farrays[i].attributes,ind);
    newattr[ind] = (char) NULL;
    if (ind+strlen(attr) < strlen(self->farrays[i].attributes))
    strcat(newattr,self->farrays[i].attributes+ind+strlen(attr));
    /* See comments in addvarattr why the free is commented out */
    /* free(self->farrays[i].attributes); */
    self->farrays[i].attributes = newattr;
    returnnone;}

  PyErr_SetString(ErrorObject,"No such variable");
  return NULL;
}

/* ######################################################################### */
/* # Get comment for the variable name.                                      */
static char getvardoc_doc[] = "Gets the documentation for a variable";
static PyObject *ForthonPackage_getvardoc(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i;
  char *name;
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    return Py_BuildValue("s",self->fscalars[i].comment);
    }

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    return Py_BuildValue("s",self->farrays[i].comment);
    }

  returnnone;
}

/* ######################################################################### */
static char isdynamic_doc[] = "Checks whether a variable is dynamic.";
static PyObject *ForthonPackage_isdynamic(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i;
  char *name;
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* Check for scalar of derived type which could be dynamic */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    return Py_BuildValue("i",self->fscalars[i].dynamic);
    }

  /* Get index for variable from array dictionary */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    return Py_BuildValue("i",self->farrays[i].dynamic);
    }

  PyErr_SetString(PyExc_AttributeError,"package has no such attribute");
  return NULL;
}

/* ######################################################################### */
/* # Group allocation freeing routine */
static char gfree_doc[] = "Frees the memory of all dynamic arrays in a group";
static PyObject *ForthonPackage_gfree(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  long i;
  int r=0;
  char *s=NULL;
  PyObject *star;

  if (!PyArg_ParseTuple(args,"|s",&s)) return NULL;
  if (s == NULL) s = "*";

  self->allocated = 0;

  /* Check for any scalars of derived type. These must also be freed */
  for (i=0;i<self->nscalars;i++) {
    if (strcmp(s,self->fscalars[i].group)==0 || strcmp(s,"*")==0) {
      if (!(self->fscalars[i].dynamic)) {
        if (self->fscalars[i].type == PyArray_OBJECT &&
            self->fscalars[i].data != NULL) {
          r = 1;
          star = Py_BuildValue("(s)","*");
          ForthonPackage_gfree((PyObject *)self->fscalars[i].data,star);
          Py_DECREF(star);
      }}}}

  for (i=0;i<self->narrays;i++) {
    if (strcmp(s,self->farrays[i].group)==0 || strcmp(s,"*")==0) {
      r = 1;
      Forthon_freearray(self,(void *)i);
      }
    }

  return Py_BuildValue("i",r);
}

/* ######################################################################### */
/* # Group set dimensions routine                                            */
/* The dimensions for the dynamic arrays are set in a routine which is       */
/* specific to each package.                                                 */
static char gsetdims_doc[] = "Sets the dimensions of dynamic arrays in the wrapper database";
static PyObject *ForthonPackage_gsetdims(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  char *s=NULL;
  int i,iverbose;
  PyObject *star;
  if (!PyArg_ParseTuple(args,"|si",&s,&iverbose)) return NULL;
  if (s == NULL) s = "*";

  /* Check for any scalars of derived type. These must also be allocated */
  for (i=0;i<self->nscalars;i++) {
    if (strcmp(s,self->fscalars[i].group)==0 || strcmp(s,"*")==0) {
      if (!(self->fscalars[i].dynamic)) {
        if (self->fscalars[i].type == PyArray_OBJECT &&
            self->fscalars[i].data != NULL) {
          star = Py_BuildValue("(s)","*");
          ForthonPackage_gsetdims((PyObject *)self->fscalars[i].data,star);
          Py_DECREF(star);
      }}}}

  /* Call the routine which sets the dimensions */
  (*self->setdims)(s,self);

  returnnone;
}

/* ######################################################################### */
/* # Print information about the variable name.                              */
static char getvartype_doc[] = "Returns the fortran type of a variable";
static PyObject *ForthonPackage_getvartype(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i;
  char *name;
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* The PyString stuff is done to avoid having to deal with strings at the
     C level, which would require explicit memory allocations (yuck!) */

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    if (self->fscalars[i].type == PyArray_CHAR) {
      return PyString_FromString("character");}
    else if (self->fscalars[i].type == PyArray_LONG) {
      return PyString_FromString("integer");}
    else if (self->fscalars[i].type == PyArray_DOUBLE) {
      return PyString_FromString("double");}
    else if (self->fscalars[i].type == PyArray_CDOUBLE) {
      return PyString_FromString("double complex");}
    }

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    if (self->farrays[i].type == PyArray_CHAR) {
      return PyString_FromString("character");}
    else if (self->farrays[i].type == PyArray_LONG) {
      return PyString_FromString("integer");}
    else if (self->farrays[i].type == PyArray_DOUBLE) {
      return PyString_FromString("double");}
    else if (self->farrays[i].type == PyArray_CDOUBLE) {
      return PyString_FromString("double complex");}
    }

  returnnone;

}

/* ######################################################################### */
/* # Print information about the variable name.                              */
static char listvar_doc[] = "Returns information about a variable";
static PyObject *ForthonPackage_listvar(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  PyObject *doc;
  int i;
  char *name;
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* The PyString stuff is done to avoid having to deal with strings at the
     C level, which would require explicit memory allocations (yuck!) */

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    doc = PyString_FromString("");
    PyString_ConcatAndDel(&doc,PyString_FromString("Package:    "));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->name));
    PyString_ConcatAndDel(&doc,PyString_FromString("\nGroup:      "));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->fscalars[i].group));
    PyString_ConcatAndDel(&doc,PyString_FromString("\nAttributes:"));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->fscalars[i].attributes));
    PyString_ConcatAndDel(&doc,PyString_FromString("\nType:       "));
    if (self->fscalars[i].type == PyArray_CHAR) {
      PyString_ConcatAndDel(&doc,PyString_FromString("char"));}
    else if (self->fscalars[i].type == PyArray_LONG) {
      PyString_ConcatAndDel(&doc,PyString_FromString("integer"));}
    else if (self->fscalars[i].type == PyArray_DOUBLE) {
      PyString_ConcatAndDel(&doc,PyString_FromString("double"));}
    else if (self->fscalars[i].type == PyArray_CDOUBLE) {
      PyString_ConcatAndDel(&doc,PyString_FromString("double complex"));}
    PyString_ConcatAndDel(&doc,PyString_FromString("\nAddress:    "));
    if (self->fscalars[i].type == PyArray_OBJECT)
      ForthonPackage_updatederivedtype(self,i,(long) 1);
    PyString_ConcatAndDel(&doc,PyObject_Str(PyInt_FromLong((long)(self->fscalars[i].data))));
    PyString_ConcatAndDel(&doc,PyString_FromString("\nComment:\n"));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->fscalars[i].comment));
    return doc;
    }

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    doc = PyString_FromString("");
    PyString_ConcatAndDel(&doc,PyString_FromString("Package:    "));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->name));
    PyString_ConcatAndDel(&doc,PyString_FromString("\nGroup:      "));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->farrays[i].group));
    PyString_ConcatAndDel(&doc,PyString_FromString("\nAttributes:"));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->farrays[i].attributes));
    PyString_ConcatAndDel(&doc,PyString_FromString("\nDimension:  "));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->farrays[i].dimstring));
    PyString_ConcatAndDel(&doc,PyString_FromString("\nType:       "));
    if (self->farrays[i].type == PyArray_CHAR) {
      PyString_ConcatAndDel(&doc,PyString_FromString("char"));}
    else if (self->farrays[i].type == PyArray_LONG) {
      PyString_ConcatAndDel(&doc,PyString_FromString("integer"));}
    else if (self->farrays[i].type == PyArray_DOUBLE) {
      PyString_ConcatAndDel(&doc,PyString_FromString("double"));}
    else if (self->farrays[i].type == PyArray_CDOUBLE) {
      PyString_ConcatAndDel(&doc,PyString_FromString("double complex"));}

    PyString_ConcatAndDel(&doc,PyString_FromString("\nAddress:    "));
    if (self->farrays[i].pya == NULL) {
      PyString_ConcatAndDel(&doc,PyString_FromString("unallocated"));}
    else {
      PyString_ConcatAndDel(&doc,PyObject_Str(PyInt_FromLong((long)((self->farrays[i].pya)->data))));}

    PyString_ConcatAndDel(&doc,PyString_FromString("\nPyaddress:  "));
    if ((self->farrays[i].pya) == 0)
      PyString_ConcatAndDel(&doc,PyString_FromString("unallocated"));
    else
      PyString_ConcatAndDel(&doc,PyObject_Str(PyInt_FromLong((long)(self->farrays[i].pya))));

    PyString_ConcatAndDel(&doc,PyString_FromString("\nComment:\n"));
    PyString_ConcatAndDel(&doc,PyString_FromString(self->farrays[i].comment));
    return doc;
    }

  returnnone;

}

/* ######################################################################### */
/* # Print information about the variable name.                              */
static char name_doc[] = "Returns the name of the package";
static PyObject *ForthonPackage_name(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  if (!PyArg_ParseTuple(args,"")) return NULL;
  return Py_BuildValue("s",self->name);
}

/* ######################################################################### */
static char reprefix_doc[] = "For each variable in the main dictionary, if there is a package variable with the same name it is assigned to that value. For arrays, the data is copied.";
static PyObject *ForthonPackage_reprefix(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *m,*d;
  PyObject *key, *value;
  int pos=0;
  int e;
  if (!PyArg_ParseTuple(args,"")) return NULL;
  m = PyImport_AddModule("__main__");
  d = PyModule_GetDict(m);
  while (PyDict_Next(d,&pos,&key,&value)) {
    if (value == Py_None) continue;
    e = Forthon_setattro(self,key,value);
    if (e==0) continue;
    PyErr_Clear();
    }
  returnnone;
}

/* ######################################################################### */
static char setdict_doc[] = "For each variable in the main dictionary, if there is a package variable with the same name it is assigned to that value. For arrays, the data is copied.";
static PyObject *ForthonPackage_setdict(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *dict;
  PyObject *key, *value;
  int pos=0;
  int e;
  if (!PyArg_ParseTuple(args,"O",&dict)) return NULL;
  while (PyDict_Next(dict,&pos,&key,&value)) {
    if (value == Py_None) continue;
    e = Forthon_setattro(self,key,value);
    if (e==0) continue;
    PyErr_Clear();
    }
  returnnone;
}

/* ######################################################################### */
/* # Returns the total number of bytes which have been allocated.           */
static char totmembytes_doc[] = "Returns total number of bytes dynamically allocated for the object.";
static PyObject *ForthonPackage_totmembytes(PyObject *self,PyObject *args)
{
  return Py_BuildValue("l",totmembytes);
}

/* ######################################################################### */
/* # Get list of variable names matching either the attribute or group name. */
static char varlist_doc[] = "Returns a list of variables having either an attribute or in a group";
static PyObject *ForthonPackage_varlist(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *result,*pyname;
  int i;
  char *name = "*";
  if (!PyArg_ParseTuple(args,"|s",&name)) return NULL;

  /* # Create the list to be returned, initially an empty list */
  result = PyList_New(0);

  /* # Loop over scalars */
  for (i=0;i<self->nscalars;i++) {
    if (strcmp(name,self->fscalars[i].group) == 0 ||
        strcmp(name,"*")==0 ||
        strfind(name,self->fscalars[i].attributes)>=0) {
      pyname = Py_BuildValue("s",self->fscalars[i].name);
      PyList_Append(result,pyname);
      Py_DECREF(pyname);
      }
    }

  /* # Loop over arrays */
  for (i=0;i<self->narrays;i++) {
    if (strcmp(name,self->farrays[i].group) == 0 ||
        strcmp(name,"*")==0 ||
        strfind(name,self->farrays[i].attributes)>=0) {
      pyname = Py_BuildValue("s",self->farrays[i].name);
      PyList_Append(result,pyname);
      Py_DECREF(pyname);
      }
    }

  return result;
}

/* ######################################################################### */
/* # Method list                                                            */
/* Methods which are callable as attributes of a package                   */
static struct PyMethodDef ForthonPackage_methods[] = {
  {"addvarattr"  ,(PyCFunction)ForthonPackage_addvarattr,1,addvarattr_doc},
  {"allocated"   ,(PyCFunction)ForthonPackage_allocated,1,allocated_doc},
  {"deprefix"    ,(PyCFunction)ForthonPackage_deprefix,1,deprefix_doc},
  {"forceassign" ,(PyCFunction)ForthonPackage_forceassign,1,forceassign_doc},
  {"gallot"      ,(PyCFunction)ForthonPackage_gallot,1,gallot_doc},
  {"gchange"     ,(PyCFunction)ForthonPackage_gchange,1,gchange_doc},
  {"getdict"     ,(PyCFunction)ForthonPackage_getdict,1,getdict_doc},
  {"getfobject"  ,(PyCFunction)ForthonPackage_getfobject,1,getfobject_doc},
  {"getfunctions",(PyCFunction)ForthonPackage_getfunctions,1,getfunctions_doc},
  {"getgroup"    ,(PyCFunction)ForthonPackage_getgroup,1,getgroup_doc},
  {"getpyobject" ,(PyCFunction)ForthonPackage_getpyobject,1,getpyobject_doc},
  {"gettypename" ,(PyCFunction)ForthonPackage_gettypename,1,gettypename_doc},
  {"getvarattr"  ,(PyCFunction)ForthonPackage_getvarattr,1,getvarattr_doc},
  {"setvarattr"  ,(PyCFunction)ForthonPackage_setvarattr,1,setvarattr_doc},
  {"delvarattr"  ,(PyCFunction)ForthonPackage_delvarattr,1,delvarattr_doc},
  {"getvardoc"   ,(PyCFunction)ForthonPackage_getvardoc,1,getvardoc_doc},
  {"gfree"       ,(PyCFunction)ForthonPackage_gfree,1,gfree_doc},
  {"gsetdims"    ,(PyCFunction)ForthonPackage_gsetdims,1,gsetdims_doc},
  {"isdynamic"   ,(PyCFunction)ForthonPackage_isdynamic,1,isdynamic_doc},
  {"getvartype"  ,(PyCFunction)ForthonPackage_getvartype,1,getvartype_doc},
  {"listvar"     ,(PyCFunction)ForthonPackage_listvar,1,listvar_doc},
  {"name"        ,(PyCFunction)ForthonPackage_name,1,name_doc},
  {"reprefix"    ,(PyCFunction)ForthonPackage_reprefix,1,reprefix_doc},
  {"setdict"     ,(PyCFunction)ForthonPackage_setdict,1,setdict_doc},
  {"totmembytes" ,(PyCFunction)ForthonPackage_totmembytes,1,totmembytes_doc},
  {"varlist"     ,(PyCFunction)ForthonPackage_varlist,1,varlist_doc},
  {NULL,NULL}};

static PyMethodDef *getForthonPackage_methods(void)
{
  return ForthonPackage_methods;
}

static void Forthon_dealloc(ForthonObject *self)
{
  if (self->garbagecollected) PyObject_GC_UnTrack((PyObject *) self);
  Forthon_clear(self);
  PyObject_GC_Del((PyObject*)self);
  /* self->ob_type->tp_free((PyObject*)self); */
}

/* ######################################################################### */
/* # Get attribute handler                                                   */
static PyObject *Forthon_getattro(ForthonObject *self,PyObject *oname)
{
  long i;
  PyObject *pyi;
  PyObject *meth;
  char *name;

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItem(self->scalardict,oname);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"l",&i);
    if (self->fscalars[i].type == PyArray_DOUBLE) {
      return Forthon_getscalardouble(self,(void *)i);}
    else if (self->fscalars[i].type == PyArray_CDOUBLE) {
      return Forthon_getscalarcdouble(self,(void *)i);}
    else if (self->fscalars[i].type == PyArray_OBJECT) {
      return Forthon_getscalarderivedtype(self,(void *)i);}
    else {
      return Forthon_getscalarinteger(self,(void *)i);}
    }

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItem(self->arraydict,oname);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"l",&i);
    return Forthon_getarray(self,(void *)i);}

  /* Now convert oname into the actual string, checking for errors. */
  name = PyString_AsString(oname);
  if (name == NULL) return NULL;

  /* Check if asking for one of the dictionaries */
  /* Note that these should probably not be accessable */
  if (strcmp(name,"scalardict") == 0) return self->scalardict;
  if (strcmp(name,"arraydict") == 0) return self->arraydict;

  /* # Look through the method lists */
  meth = Py_FindMethod(self->fmethods,(PyObject *)self,name);
  if (meth == NULL) {
    PyErr_Clear();
    meth = Py_FindMethod(ForthonPackage_methods,(PyObject *)self,name);
    }
  return meth;
}

/* ######################################################################### */
/* # Set attribute handler                                                   */
static int Forthon_setattro(ForthonObject *self,PyObject *oname,PyObject *v)
{
  long i;
  PyObject *pyi;

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItem(self->scalardict,oname);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"l",&i);
    if (self->fscalars[i].type == PyArray_DOUBLE) {
      return Forthon_setscalardouble(self,v,(void *)i);}
    else if (self->fscalars[i].type == PyArray_CDOUBLE) {
      return Forthon_setscalarcdouble(self,v,(void *)i);}
    else if (self->fscalars[i].type == PyArray_OBJECT) {
      return Forthon_setscalarderivedtype(self,v,(void *)i);}
    else {
      return Forthon_setscalarinteger(self,v,(void *)i);}
    }

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItem(self->arraydict,oname);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"l",&i);
    return Forthon_setarray(self,v,(void *)i);}

  PyErr_SetString(ErrorObject,"no such attribute");
  return -1;
}

/* ######################################################################### */
/* # Create output routines                                                  */
static int Forthon_print(ForthonObject *self, FILE *fp, int flags)
{
  fprintf(fp,"<%s instance at address = %ld>",self->name,(long)self);
  return 0;
}

static PyObject *Forthon_repr(ForthonObject *self)
{
  PyObject *s;
  s = Py_BuildValue("s","<fortran object instance>");
  return s;
}

/* ######################################################################### */
/* # Package object declaration                                              */
static PyTypeObject ForthonType = {
  PyObject_HEAD_INIT(NULL)
  0,                                     /*ob_size*/
  "Forthon",                             /*tp_name*/
  sizeof(ForthonObject),                 /*tp_basicsize*/
  0,                                     /*tp_itemsize*/
  (destructor)Forthon_dealloc,           /*tp_dealloc*/
  (printfunc)Forthon_print,              /*tp_print*/
  0,                                     /*tp_getattr*/
  0,                                     /*tp_setattr*/
  0,                                     /*tp_compare*/
  (reprfunc)Forthon_repr,                /*tp_repr*/
  0,                                     /*tp_as_number*/
  0,                                     /*tp_as_sequence*/
  0,                                     /*tp_as_mapping*/
  0,                                     /*tp_hash*/
  0,                                     /*tp_call*/
  0,                                     /*tp_str*/
  (getattrofunc)Forthon_getattro,        /*tp_getattro*/
  (setattrofunc)Forthon_setattro,        /*tp_setattro*/
  0,                                     /*tp_as_buffer*/
  Py_TPFLAGS_DEFAULT|Py_TPFLAGS_HAVE_GC, /*tp_flags*/
  "Forthon objects",                     /*tp_doc*/
  (traverseproc)Forthon_traverse,        /* tp_traverse */
  (inquiry)Forthon_clear,                /* tp_clear */

};
