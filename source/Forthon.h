/* Created by David P. Grote, March 6, 1998 */

#include <Python.h>

#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#endif

#define NPY_NO_DEPRECATED_API 8
#include <numpy/arrayobject.h>

/* The NPY_ARRAY_ versions were created in numpy version 1.7 */
/* This is needed for backwards compatibility. */
#ifndef NPY_ARRAY_BEHAVED_NS
#define NPY_ARRAY_BEHAVED_NS NPY_BEHAVED_NS
#define NPY_ARRAY_C_CONTIGUOUS NPY_C_CONTIGUOUS
#define NPY_ARRAY_F_CONTIGUOUS NPY_F_CONTIGUOUS
#define NPY_ARRAY_FARRAY NPY_FARRAY
typedef PyArrayObject PyArrayObject_fields;
#endif

#include <pythonrun.h>
#include "forthonf2c.h"

static PyObject *ErrorObject;

#define returnnone {Py_INCREF(Py_None);return Py_None;}

/* This converts a python object into a python array,         */
/* requesting fortran ordering.                               */
static PyArrayObject* FARRAY_FROMOBJECT(PyObject *A2, int ARRAY_TYPE) {
  PyArrayObject *A1;
  PyArray_Descr *descr;
  descr = PyArray_DescrFromType(ARRAY_TYPE);
  A1 = (PyArrayObject *)PyArray_CheckFromAny(A2,descr,0,0,NPY_ARRAY_BEHAVED_NS|NPY_ARRAY_F_CONTIGUOUS,NULL);
  return A1;
}

/* ####################################################################### */
/* # Write definition of scalar and array structures. Note that if these   */
/* # are changed, then the declarations written out in wrappergenerator    */
/* # must also be changed. */
struct ForthonObject_;
typedef struct {
  int type;
  char *typename;
  char* name;
  char* data;
  char* group;
  char* attributes;
  char* comment;
  int dynamic;
  int parameter;
  void (*setscalarpointer)(char *,char *,npy_intp *);
  void (*getscalarpointer)(struct ForthonObject_ **,char *,int *);
  void (*setaction)();
  void (*getaction)();
  } Fortranscalar;

struct Fortranarray_;
typedef struct Fortranarray_{
  int type;
  int dynamic;
  int nd;
  npy_intp* dimensions;
  char* name;
  union {char* s;char** d;} data;
  void (*setarraypointer)(char *,char *,npy_intp *);
  void (*getarraypointer)(struct Fortranarray_ *,char*);
  void (*setaction)();
  void (*getaction)();
  double initvalue;
  PyArrayObject* pya;
  char* group;
  char* attributes;
  char* comment;
  char* dimstring;
  } Fortranarray;

/* ######################################################################### */
/* # Write definition of fortran package type */
typedef struct ForthonObject_ {
  PyObject_HEAD
  char *name;
  char *typename;
  int nscalars;
  Fortranscalar *fscalars;
  int narrays;
  Fortranarray *farrays;
  void (*setdims)(char *,struct ForthonObject_ *,long);
  void (*setstaticdims)(struct ForthonObject_ *);
  PyMethodDef *fmethods;
  PyObject *scalardict,*arraydict;
  PyObject *__module__;
  char *fobj;
  void (*fobjdeallocate)(char *);
  void (*nullifycobj)(char *);
  int allocated;
  int garbagecollected;
} ForthonObject;
static PyTypeObject ForthonType;

/* This is needed here to settle circular dependencies */
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
  ls = (int)strlen(s);
  lv = (int)strlen(v);
  while (ls >= lv) {
    if (strncmp(s,v,strlen(v))==0) return ind;
    s++;
    ind++;
    ls--;
    }
  return -1;
}

/* ###################################################################### */
/* Utility routines used in wrapping the subroutines                      */
/* It checks if the argument can be cast to the desired type.             */
static int Forthon_checksubroutineargtype(PyObject *pyobj,int type_num)
{
  int ret;
  if (PyArray_Check(pyobj)) {
    /* If the input argument is an array, make sure that it is of the */
    /* correct type. */
    ret = (PyArray_TYPE((PyArrayObject *)pyobj) == type_num);
    if (!ret) {
      /* If it is not, do some more checking. In some cases, LONG and INT */
      /* or DOUBLE and FLOAT may be equivalent. */
      if (type_num == NPY_LONG &&
          PyArray_EquivTypenums(NPY_LONG,NPY_INT)) {
        ret = (PyArray_TYPE((PyArrayObject *)pyobj) == NPY_INT);
        }
      else if (type_num == NPY_DOUBLE &&
          PyArray_EquivTypenums(NPY_DOUBLE,NPY_FLOAT)) {
        ret = (PyArray_TYPE((PyArrayObject *)pyobj) == NPY_FLOAT);
        }
      }
    }
  else {
    /* Scalars can always be cast. Note that the data won't be returned */
    /* after the call. */
    ret = 1;
    }
  return ret;
}

/* ###################################################################### */
/* In some cases, when an array is passed from Python to Fortran, for example */
/* if the Python array is not contiguous or not in Fortran ordering, a temporary */
/* copy of the array is made and passed into Fortrh. This routine copies */
/* the data back into the original Python array after the Fortran routine finishes. */
static void Forthon_restoresubroutineargs(int n,PyObject **pyobj,
                                          PyArrayObject **ax)
{
  int i,ret;
  /* Loop over the arguments */
  for (i=0;i<n;i++) {
    /* For each input value that is an array... */
    if (PyArray_Check(pyobj[i])) {
      /* ... check if a copy was made to pass into the wrapped subroutine... */
      if (pyobj[i] != (PyObject *)ax[i]) {
        /* ... If so, copy it back. */
        ret = PyArray_CopyInto((PyArrayObject *)pyobj[i],ax[i]);
        /* Look for errors */
        if (ret == -1) {
          /* If there was one, print the message and clear it */
          if (PyErr_Occurred()) {
            printf("Error restoring argument number %d\n",i);
            PyErr_Print();
            PyErr_Clear();
            }
          else {
            printf("Unsupported problem restoring argument number %d, bad value returned but no error raised. This should never happan.\n",i);
            }
          }
        }
      }
    /* Make sure the temporary references are removed */
    if (ax[i] != NULL) {Py_XDECREF(ax[i]);}
    }
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
/* # Create a new PyArrayObject based on a FortranArray object.              */
/* # If data is NULL, new space is allocated.                                */
static PyArrayObject *ForthonPackage_PyArrayFromFarray(Fortranarray *farray,void *data)
{
  int j,nd,itemsize;
  npy_intp *dimensions;
  PyArrayObject *pya;

  /* Strings need special treatment. */
  if (farray->type == NPY_STRING) {
    /* First, note that strings are always treated as arrays. */
    /* The numpy array type is set so that the element size is the */
    /* length of the associated fortran character variable. If the */
    /* string is not an array, then it is made into a 1-D array of */
    /* length 1. */
    nd = farray->nd;
    /* The dimensions[0] holds the character length. */
    itemsize = (int)(farray->dimensions[0]);
    /* If this is really an array, remove the first dimension, which */
    /* is the character length. Otherwise, make it a 1-D array. */
    if (nd > 1) {nd -= 1;}
    else        {nd = 1;}
    /* Allocate the appropriate amount of space for the dimensions. */
    dimensions = (npy_intp*)PyMem_Malloc(sizeof(npy_intp)*nd);
    if (farray->nd == 1) {
      /* This is really a scalar, so make its length 1. */
      dimensions[0] = (npy_intp)1;
      }
    else {
      /* Copy over the rest of the dimensions. */
      for (j=1;j<farray->nd;j++)
        dimensions[j-1] = farray->dimensions[j];
      }
    }
  else {
    /* Set the appropriate variables that have different values */
    /* for strings. */
    nd = farray->nd;
    itemsize = 0;
    dimensions = farray->dimensions;
    }

  pya = (PyArrayObject *)PyArray_New(&PyArray_Type,
                                     nd,dimensions,
                                     farray->type,NULL,
                                     data,itemsize,NPY_ARRAY_FARRAY,NULL);

  if (farray->type == NPY_STRING) PyMem_Free(dimensions);

  return pya;
}

/* ######################################################################### */
/* # Update the data element of a dynamic, fortran assignable array.         */
/* ------------------------------------------------------------------------- */
/* ######################################################################### */
/* Check if the dimensions as saved in the farray match the dimension of the */
/* associated numpy array. */
static int dimensionsmatch(Fortranarray *farray)
{
  int i;
  int result = 1;
  for (i=0;i<farray->nd;i++) {
    if (farray->dimensions[i] != PyArray_DIMS(farray->pya)[i])
      result = 0;}
  return result;
}

/* ######################################################################### */
/* Check if the array that the Fortran variable points to has been changed. */
/* If it points to a new array, create a numpy array to refer to it. If it */
/* was deallocated, leave the attribute unassociated. */
static void ForthonPackage_updatearray(ForthonObject *self,long i)
{
  Fortranarray *farray = &(self->farrays[i]);
  int j;
  /* If the getarraypointer routine exists, call it to assign a value to data.s */
  if (farray->getarraypointer != NULL) {
    /* Force the pointer to be null, since if the array is not associated, */
    /* the getarraypointer routine just returns and does nothing. This ensures  */
    /* that when the fortan array has been nullified, that garbage data    */
    /* will not be returned when the array is accessed from python.          */
    /* If the array is associated, then farray->data.s will be set         */
    /* appropriately by getarraypointer.                                        */
    farray->data.s = NULL;
    (farray->getarraypointer)(farray,self->fobj);
    /* If the data.s is NULL, then the fortran array is not associated. */
    /* Decrement the python object counter if there is one. */
    /* Set the pointer to the python object to NULL and clear out the */
    /* dimensions. */
    if (farray->data.s == NULL) {
      if (farray->pya != NULL) {Py_XDECREF(farray->pya);}
      farray->pya = NULL;
      for (j=0;j<farray->nd;j++) farray->dimensions[j] = 0;
      }
    else if (farray->pya == NULL ||
             farray->data.s != PyArray_BYTES(farray->pya) ||
             !dimensionsmatch(farray)) {
      /* If data.s is not NULL and there is no python object or its */
      /* data is different, then create a new one. */
      if (farray->pya != NULL) {Py_XDECREF(farray->pya);}
      farray->pya = ForthonPackage_PyArrayFromFarray(farray,farray->data.s);
    }
  }
}

/* ######################################################################### */
/* # Allocate the dimensions element for all of the farrays in the object.   */
static void ForthonPackage_allotdims(ForthonObject *self)
{
  int i;
  for (i=0;i<self->narrays;i++) {
    self->farrays[i].dimensions = (npy_intp *)PyMem_Malloc(self->farrays[i].nd*sizeof(npy_intp));
    if (self->farrays[i].dimensions == NULL) {
      printf("Failure allocating space for dimensions of array %s.\n",self->farrays[i].name);
      exit(EXIT_FAILURE);
      }
    /* Fill the dimensions with zeros. This is only needed for arrays with */
    /* unspecified shape, since setdims won't fill the dimensions. */
    memset(self->farrays[i].dimensions,0,self->farrays[i].nd*sizeof(npy_intp));
    }
}

/* ######################################################################### */
/* Static array initialization routines. Create a numpy array for each */
/* static array. */
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
      self->farrays[i].pya = ForthonPackage_PyArrayFromFarray(&(self->farrays[i]),
                                                              self->farrays[i].data.s);
      /* Check if the allocation was unsuccessful. */
      if (self->farrays[i].pya==NULL) {
        PyErr_Print();
        printf("Failure creating python object for static array %s\n",
               self->farrays[i].name);
        exit(EXIT_FAILURE);}
      /* For strings, replace nulls with blank spaces */
      if (self->farrays[i].type == NPY_STRING) {
        long itemsize = (long)PyArray_ITEMSIZE(self->farrays[i].pya);
        if ((c=memchr(self->farrays[i].data.s,0,
                      PyArray_SIZE(self->farrays[i].pya)*itemsize)))
          /* Note that long is used since c and data.s are addresses. */
          memset(c,(int)' ',
                 (int)(PyArray_SIZE(self->farrays[i].pya)*itemsize-(long)c+
                 (long)self->farrays[i].data.s));
        /* Add the array size to totmembytes. */
        totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya)*itemsize;
        }
      else {
        /* Add the array size to totmembytes. */
        totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya);
        }
      }
    }
}

/* ######################################################################### */
/* # Update the data element of a derived type.                              */
/* Check if the object that the Fortran variable referred to has changed. */
static void ForthonPackage_updatederivedtype(ForthonObject *self,long i,
                                             int createnew)
{
  ForthonObject *objid;
  PyObject *oldobj;
  if (self->fscalars[i].type == NPY_OBJECT && self->fscalars[i].dynamic) {
    /* If dynamic, use getscalarpointer to get the current address of the */
    /* python object from the fortran variable. */
    /* This is needed since the association may have changed in fortran. */
    (self->fscalars[i].getscalarpointer)(&objid,self->fobj,&createnew);
    /* If the address has changed, that means that a reassignment was done */
    /* in fortran. The data needs to be updated and the reference */
    /* count possibly incremented. */
    if (self->fscalars[i].data != (char *)objid) {
      oldobj = (PyObject *)self->fscalars[i].data;
      /* Make sure that the correct python object is pointed to. */
      /* The pointer is redirected before the DECREF is done to avoid */
      /* infinite loops. */
      self->fscalars[i].data = (char *)objid;
      /* Increment the reference count since a new thing points to it. */
      Py_XINCREF((PyObject *)self->fscalars[i].data);
      Py_XDECREF(oldobj);
      }
    }
}

/* ######################################################################### */
static void Forthon_updatederivedtypeelements(ForthonObject *self,
                                              ForthonObject *value)
{
  int i;
  PyObject *oldobj;

  /* Loop over scalars, and update the references to derived types. Note   */
  /* that this pattern matches the way assignment of derived types is done */
  /* in fortran. For pointers (dynamic), discard the old object and point  */
  /* to the new. For static, the data is copied (and the structure is      */
  /* unchanged), but check for pointers in the subelements.                */
  for (i=0;i<self->nscalars;i++) {
    if (self->fscalars[i].type == NPY_OBJECT) {
      if (self->fscalars[i].dynamic) {
        oldobj = (PyObject *)self->fscalars[i].data;
        self->fscalars[i].data = value->fscalars[i].data;
        Py_XINCREF((PyObject *)value->fscalars[i].data);
        Py_XDECREF(oldobj);
        }
      else {
        Forthon_updatederivedtypeelements(
                       (ForthonObject *)self->fscalars[i].data,
                       (ForthonObject *)value->fscalars[i].data);
        }
      }
    }

  /* Also, update any dynamic arrays, since the old will be discarded and */
  /* the new one will be pointed to.                                      */
  (*self->setdims)(self->typename,self,-1);
  for (i=0;i<self->narrays;i++) {
    if (value->farrays[i].dynamic) {
      Py_XINCREF(value->farrays[i].pya);
      Py_XDECREF(self->farrays[i].pya);
      self->farrays[i].pya = value->farrays[i].pya;
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
static PyObject *Forthon_getscalarfloat(ForthonObject *self,void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  return Py_BuildValue("f",*((float *)(fscalar->data)));
}
/* ------------------------------------------------------------------------- */
static PyObject *Forthon_getscalarcfloat(ForthonObject *self,void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  /* THIS IS NOT RIGHT!!!! */
  return PyComplex_FromDoubles((double)(((float *)fscalar->data)[0]),
                               (double)(((float *)fscalar->data)[1]));
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
  int createnew=1;
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
  if (PyArray_NDIM(farray->pya)==1 &&
      PyArray_STRIDES(farray->pya)[0]==PyArray_ITEMSIZE(farray->pya))
    PyArray_UpdateFlags(farray->pya,(NPY_ARRAY_C_CONTIGUOUS | NPY_ARRAY_F_CONTIGUOUS));
  return (PyObject *)farray->pya;
}
/* ------------------------------------------------------------------------- */
/* --- Currently unused - needed for getset scheme
static PyObject *Forthon_getscalardict(ForthonObject *self,void *closure)
{
  Py_INCREF(self->scalardict);
  return self->scalardict;
}
*/
/* ------------------------------------------------------------------------- */
/* --- Currently unused - needed for getset scheme
static PyObject *Forthon_getarraydict(ForthonObject *self,void *closure)
{
  Py_INCREF(self->arraydict);
  return self->arraydict;
}
*/

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
      (farray->setarraypointer)(0,(self->fobj),farray->dimensions);
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
    if (fscalar->setaction != NULL) {
      if (self->fobj == NULL) fscalar->setaction(&lv);
      else                    fscalar->setaction((self->fobj),&lv);
      }
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
    if (fscalar->setaction != NULL) {
      if (self->fobj == NULL) fscalar->setaction(&lv);
      else                    fscalar->setaction((self->fobj),&lv);
      }
    memcpy((fscalar->data),&lv,2*sizeof(double));}
  else {
    PyErr_SetString(ErrorObject,"Right hand side has incorrect type");
    return -1;}
  return 0;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setscalarfloat(ForthonObject *self,PyObject *value,
                                   void *closure)
{
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  float lv;
  int e;
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
    return -1;}
  e = PyArg_Parse(value,"f",&lv);
  if (e) {
    if (fscalar->setaction != NULL) {
      if (self->fobj == NULL) fscalar->setaction(&lv);
      else                    fscalar->setaction((self->fobj),&lv);
      }
    memcpy((fscalar->data),&lv,sizeof(float));}
  else {
    PyErr_SetString(ErrorObject,"Right hand side has incorrect type");
    return -1;}
  return 0;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setscalarcfloat(ForthonObject *self,PyObject *value,
                                    void *closure)
{
  /* This is probably not correct!!! */
  Fortranscalar *fscalar = &(self->fscalars[(long)closure]);
  Py_complex lv;
  int e;
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
    return -1;}
  e = PyArg_Parse(value,"D",&lv);
  if (e) {
    if (fscalar->setaction != NULL) {
      if (self->fobj == NULL) fscalar->setaction(&lv);
      else                    fscalar->setaction((self->fobj),&lv);
      }
    memcpy((fscalar->data),&lv,2*sizeof(float));}
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
  if (value == NULL) {
    PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
    return -1;}
  /* This will convert floats to ints if needed */
#if PY_MAJOR_VERSION >= 3
  lv = PyLong_AsLong(value);
#else
  lv = PyInt_AsLong(value);
#endif
  if (!PyErr_Occurred()) {
    if (fscalar->setaction != NULL) {
      if (self->fobj == NULL) fscalar->setaction(&lv);
      else                    fscalar->setaction((self->fobj),&lv);
      }
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
  long i = (long)closure;
  Fortranscalar *fscalar = &(self->fscalars[i]);
  int createnew;
  npy_intp nullit;
  PyObject *oldobj;

  /* Only create a new instance if a non-NULL value is passed in. */
  /* With a NULL or None value, the object will be decref'ed so there's no */
  /* point creating a new one. */
  createnew = (value != NULL);
  ForthonPackage_updatederivedtype(self,i,createnew);

  if (value == NULL || value == Py_None) {
    if (fscalar->dynamic) {
      if (fscalar->data != NULL) {
        /* Decrement the reference counter and nullify the fortran pointer. */
        oldobj = (PyObject *)fscalar->data;
        nullit = 1;
        (fscalar->setscalarpointer)(0,(self->fobj),&nullit);
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

  if (strcmp("Forthon",Py_TYPE(value)->tp_name) != 0 ||
      strcmp(((ForthonObject *)value)->typename,fscalar->typename) != 0) {
    PyErr_SetString(ErrorObject,"Right hand side has incorrect type");
    return -1;}

  if (fscalar->dynamic) {
    /* Only swap pointers if the object is dynamic. In fortran, if the     */
    /* object is not dynamic, the data is copied and the structure doesn't */
    /* change. If its dynamic, the old data is discarded and the new data  */
    /* is pointed to.                                                      */
    oldobj = (PyObject *)fscalar->data;
    fscalar->data = (char *)value;
    Py_INCREF(value);
    Py_XDECREF(oldobj);
  }

  if (fscalar->setaction != NULL) {
    if (self->fobj == NULL)
      fscalar->setaction(((ForthonObject *)value)->fobj);
    else
      fscalar->setaction((self->fobj),((ForthonObject *)value)->fobj);
    }

  /* This does the assignment in Fortran. */
  nullit = 0;
  (fscalar->setscalarpointer)(((ForthonObject *)value)->fobj,(self->fobj),&nullit);

  /* Call this after the setscalarpointer so that the updates will refer to the */
  /* data. */
  if (!(fscalar->dynamic)) {
    Forthon_updatederivedtypeelements((ForthonObject *)(fscalar->data),
                                      (ForthonObject *)value);
    }

  return 0;
}
/* ------------------------------------------------------------------------- */
static int Forthon_setarray(ForthonObject *self,PyObject *value,
                            void *closure)
{
  Fortranarray *farray = &(self->farrays[(long)closure]);
  int j,r,d,setit;
  PyObject *pyobj;
  PyArrayObject *ax;

  if (value == NULL || value == Py_None) {
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
  ax = FARRAY_FROMOBJECT(pyobj,farray->type);
  if ((farray->dynamic && PyArray_NDIM(ax) == farray->nd) ||
      (farray->dynamic == 3 && farray->nd == 1 && PyArray_NDIM(ax) == 0 &&
       farray->pya == NULL)) {
    /* The long list of checks above looks for the special case of assigning */
    /* a scalar to a 1-D deferred-shape array that is unallocated. In that   */
    /* case, a 1-D array is created with the value of the scalar. If the     */
    /* array is already allocated, the code below broadcasts the scalar over */
    /* the array. */
    if (farray->dynamic == 3) {
      /* This type of dynamic array (deferred-shape) does not have the */
      /* dimensions specified so they can take on whatever is input. */
      for (j=0;j<PyArray_NDIM(ax);j++) {
        farray->dimensions[j] = PyArray_DIMS(ax)[j];
        }
      }
      if (PyArray_NDIM(ax) == 0) {
        /* Special handling is needed when a 0-D array is assigned to a      */
        /* 1-D deferred-shape array. The numpy routine used by               */
        /* FARRAY_FROMOBJECT returns a 0-D array when the input is a scalar. */
        /* This can cause problems elsewhere, so it is replaced with a 1-D   */
        /* array of length 1.                                                */
        farray->dimensions[0] = (npy_intp)1;
        Py_XDECREF(ax);
        ax = (PyArrayObject *)PyArray_SimpleNew(1,farray->dimensions,
                                                farray->type);
        PyArray_SETITEM(ax,PyArray_BYTES(ax),pyobj);
        }
    else {
      /* Call the routine which sets the dimensions */
      (*self->setdims)(farray->group,self,(long)closure);
      }
    setit = 1;
    for (j=0;j<PyArray_NDIM(ax);j++) {
      if (PyArray_DIMS(ax)[j] != farray->dimensions[j])
        setit=0;
      }
    if (setit) {
      if (farray->setaction != NULL) {
        if (self->fobj == NULL)
          farray->setaction(PyArray_BYTES(ax));
        else
          farray->setaction((self->fobj),PyArray_BYTES(ax));
        }
      if (farray->pya != NULL) {Py_XDECREF(farray->pya);}
      farray->pya = ax;
      (farray->setarraypointer)(PyArray_BYTES(farray->pya),(self->fobj),
                                PyArray_DIMS(farray->pya));
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
    /* For strings, allow the length of the input to be different than
     * variable. If the input is shorter, set the length of the variable
     * to be the same. This prevents the Copy from filling in the rest
     * of the variable with nulls - this must be done so that fortan
     * comparisons of strings still work, since fortran does not seem
     * to ignore nulls. If the length of the input is longer, the Copy
     * will do the appropriate truncation. This also fills in the string
     * with spaces to clear out any existing characters. */
    d = -1;
    if (farray->type == NPY_STRING) {
      PyArray_FILLWBYTE(farray->pya,(int)' ');
      if (PyArray_ITEMSIZE(ax) < PyArray_ITEMSIZE(farray->pya)){
        d = (int)PyArray_ITEMSIZE(farray->pya);
        /* PyArray_ITEMSIZE(farray->pya) = PyArray_ITEMSIZE(ax); */
        /* Is there a better way to do this? */
        (((PyArrayObject_fields *)(farray->pya))->descr->elsize) = (int)PyArray_ITEMSIZE(ax);
        }
      }
    /* Copy input data into the array. This does the copy */
    /* for static arrays and also does any broadcasting   */
    /* when the dimensionality of the input is different  */
    /* than the array.                                    */
    r = PyArray_CopyInto(farray->pya,ax);
    /* Reset the value of the itemsize if it was          */
    /* changed to accomodate a string.                    */
    /* if (d > -1) PyArray_ITEMSIZE(farray->pya) = d; */
    if (d > -1) (((PyArrayObject_fields *)(farray->pya))->descr->elsize) = d;
    Py_XDECREF(ax);
  }
  return r;
}
/* ------------------------------------------------------------------------- */
/* --- Currently unused - needed for getset scheme
static int Forthon_setscalardict(ForthonObject *self,void *closure)
{
  PyErr_SetString(PyExc_TypeError, "Cannot set the scalardict attribute");
  return -1;
}
*/
/* ------------------------------------------------------------------------- */
/* --- Currently unused - needed for getset scheme
static int Forthon_setarraydict(ForthonObject *self,void *closure)
{
  PyErr_SetString(PyExc_TypeError, "Cannot set the arraydict attribute");
  return -1;
}
*/

/* ######################################################################### */
static int Forthon_traverse(ForthonObject *self,visitproc visit,void *arg)
{
  int i;
  int createnew=0;
  for (i=0;i<self->nscalars;i++) {
    if (self->fscalars[i].type == NPY_OBJECT &&
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
  int createnew=0;
  npy_intp nullit=1;
  void *d;
  PyObject *oldobj;

  for (i=0;i<self->nscalars;i++) {
    if (self->fscalars[i].type == NPY_OBJECT)
      {
      ForthonPackage_updatederivedtype(self,i,createnew);
      if (self->fscalars[i].data != NULL) {
        d = (void *)((ForthonObject *)self->fscalars[i].data)->fobjdeallocate;
        oldobj = (PyObject *)self->fscalars[i].data;
        self->fscalars[i].data = NULL;
        if (d != NULL && self->fscalars[i].dynamic) {
          (self->fscalars[i].setscalarpointer)(0,(self->fobj),&nullit);
          }
        /* Only delete the object after deleting references to it. */
        Py_DECREF(oldobj);
        }
      }
    }
  for (i=0;i<self->narrays;i++) {
    /* ForthonPackage_updatearray(self,(long)i); */
    if (self->farrays[i].pya != NULL) {
      /* Subtract the array size from totmembytes. */
      totmembytes -= (long)PyArray_NBYTES(self->farrays[i].pya);
      Py_DECREF(self->farrays[i].pya);
      }
    PyMem_Free(self->farrays[i].dimensions);
    }
  if (self->fobj != NULL) {
    /* Note that for package instance (as opposed to derived type */
    /* instances), the fscalars and farrays are statically defined and */
    /* can't be freed. */
    if (self->fscalars != NULL) PyMem_Free(self->fscalars);
    if (self->farrays  != NULL) PyMem_Free(self->farrays);
    }
  if (self->fobj != NULL) {
    if (self->fobjdeallocate != NULL) {(self->fobjdeallocate)(self->fobj);}
    else                              {(self->nullifycobj)(self->fobj);}
    }

  Py_DECREF(self->__module__);

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
  int createnew=1;
  char *name;
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* Check for scalar of derived type which could be dynamic */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    if (self->fscalars[i].type == NPY_OBJECT) {
      ForthonPackage_updatederivedtype(self,(long)i,createnew);
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
    /* Update the array if it is dynamic and fortran assignable. */
    ForthonPackage_updatearray(self,(long)i);
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
  /* There is something wrong with this code XXX */
  /* n = PyUnicode_FromString(self->name); */
  /* PyDict_SetItemString(dict,"_name",n); */
  /* Py_DECREF(n); */
  for (j=0;j<self->nscalars;j++) {
    s = self->fscalars + j;
    if (s->type == NPY_DOUBLE) {
      v = Forthon_getscalardouble(self,(void *)j);}
    else if (s->type == NPY_CDOUBLE) {
      v = Forthon_getscalarcdouble(self,(void *)j);}
    else if (s->type == NPY_FLOAT) {
      v = Forthon_getscalarfloat(self,(void *)j);}
    else if (s->type == NPY_CFLOAT) {
      v = Forthon_getscalarcfloat(self,(void *)j);}
    else if (s->type == NPY_OBJECT) {
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
static char getfunctions_doc[] = "Builds a list containing all of the function names in the package.";
static PyObject *ForthonPackage_getfunctions(PyObject *_self_,PyObject *args)
{
  /* Note, that this used to return a dictionary, mapping names to python callable */
  /* functions, using PyCFunction_New. However, PyCFunction_New would grab a reference */
  /* to self, unnecessarily increasing its reference count. This caused a severe memory */
  /* leak. So now, use of a dictionary of functions is avoided. */
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *list,*name;
  PyMethodDef *ml;
  if (!PyArg_ParseTuple(args,"")) return NULL;
  list = PyList_New((Py_ssize_t)0);
  ml = getForthonPackage_methods();
  for (; ml->ml_name != NULL; ml++) {
    name = Py_BuildValue("s",ml->ml_name);
    PyList_Append(list,name);
    Py_DECREF(name);
    }
  ml = self->fmethods;
  for (; ml->ml_name != NULL; ml++) {
    name = Py_BuildValue("s",ml->ml_name);
    PyList_Append(list,name);
    Py_DECREF(name);
    }
  return list;
}

/* ######################################################################### */
/* # Create routine to force assignment of arrays                            */
/* The routines forces assignment of arrays. It takes two                    */
/* arguments, a string (the array name) and a NPY_object.                    */
/* For dynamic arrays, the array is pointed to the input array               */
/* regardless of its dimension sizes. For static arrays, a copy              */
/* is done similar to that done in gchange, where what ever                  */
/* fits into the space is copied. The main purpose of this                   */
/* routine is to allow a restore from a dump file to work.                   */
static char forceassign_doc[] = "Forces assignment to a dynamic array, resizing it if necessary";
static PyObject *ForthonPackage_forceassign(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  long i;
  int j,r=-1;
  npy_intp *pyadims,*axdims;
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
    ax = FARRAY_FROMOBJECT(pyobj,self->farrays[i].type);
    if (self->farrays[i].dynamic && PyArray_NDIM(ax) == self->farrays[i].nd) {
      /* Free the existing array */
      Forthon_freearray(self,(void *)i);
      /* Point to the new one */
      self->farrays[i].pya = ax;
      (self->farrays[i].setarraypointer)(PyArray_BYTES(self->farrays[i].pya),(self->fobj),
                                         PyArray_DIMS(self->farrays[i].pya));
      totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya);
      returnnone;}
    else if (PyArray_NDIM(ax) == self->farrays[i].nd) {
      /* Copy input data into the array. This does a copy   */
      /* even if the dimensions do not match. If an input   */
      /* dimension is larger, the extra data is truncated.  */
      /* This code ensures that the dimensions of ax        */
      /* remain intact since there may be other references  */
      /* to it.                                             */
      pyadims = PyDimMem_NEW(PyArray_NDIM(ax));
      axdims = PyDimMem_NEW(PyArray_NDIM(ax));
      for (j=0;j<PyArray_NDIM(ax);j++) {
        pyadims[j] = PyArray_DIM(self->farrays[i].pya,j);
        axdims[j] = PyArray_DIM(ax,j);
        if (PyArray_DIM(ax,j) < PyArray_DIM(self->farrays[i].pya,j)) {
          PyArray_DIMS(self->farrays[i].pya)[j] = PyArray_DIM(ax,j);}
        else {
          PyArray_DIMS(ax)[j] = PyArray_DIM(self->farrays[i].pya,j);}
        }
      r = PyArray_CopyInto(self->farrays[i].pya,ax);
      for (j=0;j<PyArray_NDIM(ax);j++) {
        PyArray_DIMS(self->farrays[i].pya)[j] = pyadims[j];
        PyArray_DIMS(ax)[j] = axdims[j];
        }
      PyDimMem_FREE(pyadims);
      PyDimMem_FREE(axdims);
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
  long i;
  int j,r=0,allotit,iverbose=0;
  PyObject *star;
  if (!PyArg_ParseTuple(args,"|si",&s,&iverbose)) return NULL;
  self->allocated = 1;
  if (s == NULL) s = "*";

  /* Check for any scalars of derived type. These must also be allocated */
  for (i=0;i<self->nscalars;i++) {
    if (strcmp(s,self->fscalars[i].group)==0 || strcmp(s,"*")==0) {
      if (!(self->fscalars[i].dynamic)) {
        if (self->fscalars[i].type == NPY_OBJECT &&
            self->fscalars[i].data != NULL) {
          r = 1;
          star = Py_BuildValue("(s)","*");
          ForthonPackage_gallot((PyObject *)self->fscalars[i].data,star);
          Py_DECREF(star);
      }}}}

  /* Process the arrays now that the dimensions are set */
  for (i=0;i<self->narrays;i++) {
   if (strcmp(s,self->farrays[i].group)==0 || strcmp(s,"*")==0) {
    /* Update the array if it is dynamic and fortran assignable. */
    ForthonPackage_updatearray(self,i);
    /* Call the routine which sets the dimensions */
    /* Call this after updatearray since updatearray might change */
    /* farrays[i].dimensions. */
    (*self->setdims)(s,self,i);
    r = 1;
    /* Note that deferred-shape arrays shouldn't be allocated in this way */
    /* since they have no specified dimensions. */
    if (self->farrays[i].dynamic && self->farrays[i].dynamic != 3) {
      /* First, free the existing array */
      Forthon_freearray(self,(void *)i);
      /* Make sure the dimensions are all greater then zero. */
      /* If not, then don't allocate the array. */
      allotit = 1;
      for (j=0;j<self->farrays[i].nd;j++)
        if (self->farrays[i].dimensions[j] <= 0) allotit = 0;
      if (allotit) {
        self->farrays[i].pya = ForthonPackage_PyArrayFromFarray(&(self->farrays[i]),NULL);
        /* Check if the allocation was unsuccessful. */
        if (self->farrays[i].pya==NULL) {
          long arraysize=1;
          for (j=0;j<self->farrays[i].nd;j++)
            arraysize *= self->farrays[i].dimensions[j];
          printf("GALLOT: allocation failure for %s to size %ld\n",
                 self->farrays[i].name,arraysize);
          exit(EXIT_FAILURE);
          }
        /* Point fortran pointer to new space */
        (self->farrays[i].setarraypointer)(PyArray_BYTES(self->farrays[i].pya),
                                           (self->fobj),
                                           PyArray_DIMS(self->farrays[i].pya));
        /* Fill array with initial value. A check could probably be made */
        /* of whether the initial value is zero since the initialization */
        /* doesn't need to be done then. Not having the check gaurantees */
        /* that it is set correctly, but is slower. */
        if (self->farrays[i].type == NPY_STRING) {
          PyArray_FILLWBYTE(self->farrays[i].pya,(int)' ');
          }
        else if (self->farrays[i].type == NPY_LONG) {
          for (j=0;j<PyArray_SIZE(self->farrays[i].pya);j++)
            *((long *)(PyArray_BYTES(self->farrays[i].pya))+j) = self->farrays[i].initvalue;
          }
        else if (self->farrays[i].type == NPY_DOUBLE) {
          for (j=0;j<PyArray_SIZE(self->farrays[i].pya);j++)
            *((double *)(PyArray_BYTES(self->farrays[i].pya))+j) = self->farrays[i].initvalue;
          }
        else if (self->farrays[i].type == NPY_FLOAT) {
          for (j=0;j<PyArray_SIZE(self->farrays[i].pya);j++)
            *((float *)(PyArray_BYTES(self->farrays[i].pya))+j) = (float)self->farrays[i].initvalue;
          }
        /* Add the array size to totmembytes. */
        totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya);
        if (iverbose) printf("Allocating %s.%s %d\n",self->name,self->farrays[i].name,
                                       (int)PyArray_SIZE(self->farrays[i].pya));
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
  long i;
  int r=0;
  PyArrayObject *ax;
  int j,rt,changeit,freeit,iverbose=0;
  npy_intp *pyadims,*axdims;
  PyObject *star;

  if (!PyArg_ParseTuple(args,"|si",&s,&iverbose)) return NULL;
  self->allocated = 1;
  if (s == NULL) s = "*";

  /* Check for any scalars of derived type. These must also be allocated */
  for (i=0;i<self->nscalars;i++) {
    if (strcmp(s,self->fscalars[i].group)==0 || strcmp(s,"*")==0) {
      if (!(self->fscalars[i].dynamic)) {
        if (self->fscalars[i].type == NPY_OBJECT &&
            self->fscalars[i].data != NULL) {
          r = 1;
          star = Py_BuildValue("(s)","*");
          ForthonPackage_gchange((PyObject *)self->fscalars[i].data,star);
          Py_DECREF(star);
      }}}}

  /* Process the arrays now that the dimensions are set */
  for (i=0;i<self->narrays;i++) {
   if (strcmp(s,self->farrays[i].group)==0 || strcmp(s,"*")==0) {
    r = 1;
    if (self->farrays[i].dynamic) {
      /* Update the array if it is dynamic and fortran assignable. */
      ForthonPackage_updatearray(self,i);
      /* Call the routine which sets the dimensions */
      /* Call this after updatearray since updatearray might change */
      /* farrays[i].dimensions. */
      (*self->setdims)(s,self,i);
      /* Check if any of the dimensions have changed or if array is */
      /* unallocated. In either case, change it. */
      changeit = 0;
      if (self->farrays[i].pya == NULL) {
        changeit = 1;}
      else {
        for (j=0;j<self->farrays[i].nd;j++) {
          if (self->farrays[i].dimensions[j] !=
              PyArray_DIMS(self->farrays[i].pya)[j])
            changeit = 1;}
        }
      /* Make sure all of the dimensions are >= 0. If not, then free it. */
      freeit = 0;
      for (j=0;j<self->farrays[i].nd;j++)
        if (self->farrays[i].dimensions[j] <= 0) freeit = 1;
      if (freeit) Forthon_freearray(self,(void *)i);
      /* Only allocate new space and copy old data if */
      /* any dimensions are different. */
      if (changeit && !freeit) {
        /* Use array routine to create new space */
        ax = ForthonPackage_PyArrayFromFarray(&(self->farrays[i]),NULL);
        /* Check if the allocation was unsuccessful. */
        if (ax==NULL) {
          long arraysize=1;
          for (j=0;j<self->farrays[i].nd;j++)
            arraysize *= self->farrays[i].dimensions[j];
          printf("GCHANGE: allocation failure for %s to size %ld\n",
                 self->farrays[i].name,arraysize);
          exit(EXIT_FAILURE);
          }
        /* Fill array with initial value. A check could probably be made */
        /* of whether the initial value is zero since the initialization */
        /* doesn't need to be done then. Not having the check gaurantees */
        /* that it is set correctly, but is slower. */
        if (self->farrays[i].type == NPY_STRING) {
          PyArray_FILLWBYTE(ax,(int)' ');
          }
        else if (self->farrays[i].type == NPY_LONG) {
          for (j=0;j<PyArray_SIZE(ax);j++)
            *((long *)(PyArray_BYTES(ax))+j) = self->farrays[i].initvalue;
          }
        else if (self->farrays[i].type == NPY_DOUBLE) {
          for (j=0;j<PyArray_SIZE(ax);j++)
            *((double *)(PyArray_BYTES(ax))+j) = self->farrays[i].initvalue;
          }
        else if (self->farrays[i].type == NPY_FLOAT) {
          for (j=0;j<PyArray_SIZE(ax);j++)
            *((float *)(PyArray_BYTES(ax))+j) = (float)self->farrays[i].initvalue;
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
          pyadims = PyDimMem_NEW(PyArray_NDIM(ax));
          axdims = PyDimMem_NEW(PyArray_NDIM(ax));
          for (j=0;j<PyArray_NDIM(ax);j++) {
            pyadims[j] = PyArray_DIM(self->farrays[i].pya,j);
            axdims[j] = PyArray_DIM(ax,j);
            if (PyArray_DIM(ax,j) < PyArray_DIM(self->farrays[i].pya,j)) {
              PyArray_DIMS(self->farrays[i].pya)[j] = PyArray_DIM(ax,j);}
            else {
              PyArray_DIMS(ax)[j] = PyArray_DIM(self->farrays[i].pya,j);}
            }
          rt = PyArray_CopyInto(ax,self->farrays[i].pya);
          if (rt)
            printf("gchange: error copying data for the array %s",
                   self->farrays[i].name);
          for (j=0;j<PyArray_NDIM(ax);j++) {
            PyArray_DIMS(self->farrays[i].pya)[j] = pyadims[j];
            PyArray_DIMS(ax)[j] = axdims[j];
            }
          PyDimMem_FREE(pyadims);
          PyDimMem_FREE(axdims);
          }
        /* Free the old array */
        Forthon_freearray(self,(void *)i);
        /* Point pointers to new space. */
        self->farrays[i].pya = ax;
        (self->farrays[i].setarraypointer)(PyArray_BYTES(self->farrays[i].pya),
                                           (self->fobj),
                                           PyArray_DIMS(self->farrays[i].pya));
        /* Add the array size to totmembytes. */
        totmembytes += (long)PyArray_NBYTES(self->farrays[i].pya);
        if (iverbose) printf("Allocating %s.%s %d\n",self->name,self->farrays[i].name,
                                       (int)PyArray_SIZE(self->farrays[i].pya));
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
    newattr = (char *)PyMem_Malloc(strlen(self->fscalars[i].attributes) +
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
    /* PyMem_Free(self->fscalars[i].attributes); */
    self->fscalars[i].attributes = newattr;
    returnnone;}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    newattr = (char *)PyMem_Malloc(strlen(self->farrays[i].attributes) +
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
    /* PyMem_Free(self->farrays[i].attributes); */
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
    /* PyMem_Free(self->fscalars[i].attributes); */
    self->fscalars[i].attributes = (char *)PyMem_Malloc(strlen(attr) + 1);
    strcpy(self->fscalars[i].attributes,attr);
    returnnone;}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    /* See comments in addvarattr why the free is commented out */
    /* PyMem_Free(self->farrays[i].attributes); */
    self->farrays[i].attributes = (char *)PyMem_Malloc(strlen(attr) + 1);
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
    newattr = (char *)PyMem_Malloc(strlen(self->fscalars[i].attributes) -
                                   strlen(attr) + 1);
    ind = strfind(attr,self->fscalars[i].attributes);
    /* Check if attr was found, and make sure it is surrounded by spaces. */
    if (ind == -1 ||
        (ind > 0 && self->fscalars[i].attributes[ind-1] != ' ') ||
        (ind < (int)strlen(self->fscalars[i].attributes) &&
         self->fscalars[i].attributes[ind+strlen(attr)] != ' ')) {
      PyErr_SetString(ErrorObject,"Variable has no such attribute");
      return NULL;
      }
    strncpy(newattr,self->fscalars[i].attributes,ind);
    newattr[ind] = '\0';
    if ((ind+strlen(attr)) < strlen(self->fscalars[i].attributes))
      strcat(newattr,self->fscalars[i].attributes+ind+strlen(attr));
    /* See comments in addvarattr why the free is commented out */
    /* PyMem_Free(self->fscalars[i].attributes); */
    self->fscalars[i].attributes = newattr;
    returnnone;}

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    newattr = (char *)PyMem_Malloc(strlen(self->farrays[i].attributes) -
                             strlen(attr) + 1);
    ind = strfind(attr,self->farrays[i].attributes);
    /* Check if attr was found, and make sure it is surrounded by spaces. */
    if (ind == -1 ||
        (ind > 0 && self->farrays[i].attributes[ind-1] != ' ') ||
        (ind < (int)strlen(self->farrays[i].attributes) &&
         self->farrays[i].attributes[ind+strlen(attr)] != ' ')) {
      PyErr_SetString(ErrorObject,"Variable has no such attribute");
      return NULL;
      }
    strncpy(newattr,self->farrays[i].attributes,ind);
    newattr[ind] = '\0';
    if ((ind+strlen(attr)) < strlen(self->farrays[i].attributes))
    strcat(newattr,self->farrays[i].attributes+ind+strlen(attr));
    /* See comments in addvarattr why the free is commented out */
    /* PyMem_Free(self->farrays[i].attributes); */
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
static char getstrides_doc[] = "Returns the strides of the input array. The input must be an array (no lists or tuples).";
static PyObject *ForthonPackage_getstrides(PyObject *_self_,PyObject *args)
{
  PyObject *pyobj;
  PyArrayObject *ax;
  PyObject *result;
  npy_intp *dims;
  int i;
  long *strides;

  if (!PyArg_ParseTuple(args,"O",&pyobj)) return NULL;
  if (!PyArray_Check(pyobj)) {
    PyErr_SetString(PyExc_TypeError,"Input argument must be an array");
    return NULL;
    }

  ax = (PyArrayObject *)pyobj;

  /* Note that the second argument gives the dimensions of the 1-d array. */
  dims = (npy_intp *)PyMem_Malloc(sizeof(npy_intp));
  dims[0] = (npy_intp)(PyArray_NDIM(ax));
  result = (PyObject *)PyArray_SimpleNew((int)1,dims,NPY_LONG);
  PyMem_Free(dims);

  strides = (long *)PyArray_BYTES((PyArrayObject *)result);
  for (i=0;i<PyArray_NDIM(ax);i++)
    strides[i] = (long)(PyArray_STRIDES(ax)[i]);

  return result;
}

/* ######################################################################### */
static char printtypenum_doc[] = "Prints the typenum of the array. The input must be an array (no lists or tuples).";
static PyObject *ForthonPackage_printtypenum(PyObject *_self_,PyObject *args)
{
  PyObject *pyobj;

  if (!PyArg_ParseTuple(args,"O",&pyobj)) return NULL;
  if (!PyArray_Check(pyobj)) {
    PyErr_SetString(PyExc_TypeError,"Input argument must be an array");
    return NULL;
    }

  printf("Typenum = %d\n",PyArray_TYPE((PyArrayObject *)pyobj));

  returnnone;
}

/* ######################################################################### */
#ifdef WITH_FEENABLEEXCEPT
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <fenv.h>
#endif
static char feenableexcept_doc[] = "Turns on or off trapping of floating point exceptions";
static PyObject *ForthonPackage_feenableexcept(PyObject *_self_,PyObject *args)
{
  long flag;

  if (!PyArg_ParseTuple(args,"l",&flag)) return NULL;

#ifdef WITH_FEENABLEEXCEPT
    if (flag) {
      feenableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
    }
    else {
      fedisableexcept(FE_DIVBYZERO | FE_OVERFLOW | FE_INVALID);
    }
#endif

  returnnone;
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
        if (self->fscalars[i].type == NPY_OBJECT &&
            self->fscalars[i].data != NULL) {
          r = 1;
          star = Py_BuildValue("(s)","*");
          ForthonPackage_gfree((PyObject *)self->fscalars[i].data,star);
          Py_DECREF(star);
      }}}}

  for (i=0;i<self->narrays;i++) {
    if (strcmp(s,self->farrays[i].group)==0 || strcmp(s,"*")==0) {
      r = 1;
      /* Update the array if it is dynamic and fortran assignable. */
      ForthonPackage_updatearray(self,i);
      /* Then free it */
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
        if (self->fscalars[i].type == NPY_OBJECT &&
            self->fscalars[i].data != NULL) {
          star = Py_BuildValue("(s)","*");
          ForthonPackage_gsetdims((PyObject *)self->fscalars[i].data,star);
          Py_DECREF(star);
      }}}}

  /* Call the routine which sets the dimensions */
  (*self->setdims)(s,self,-1);

  returnnone;
}

/* ######################################################################### */
/* # Print information about the variable name.                              */
static char getvartype_doc[] = "Returns the fortran type of a variable";
static PyObject *ForthonPackage_getvartype(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  int i,charsize;
  char *name;
  char charstring[50];
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* The PyUnicode stuff is done to avoid having to deal with strings at the
     C level, which would require explicit memory allocations (yuck!) */

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    if (self->fscalars[i].type == NPY_STRING) {
      return PyUnicode_FromString("character");}
    else if (self->fscalars[i].type == NPY_LONG) {
      return PyUnicode_FromString("integer");}
    else if (self->fscalars[i].type == NPY_DOUBLE) {
      return PyUnicode_FromString("double");}
    else if (self->fscalars[i].type == NPY_CDOUBLE) {
      return PyUnicode_FromString("double complex");}
    else if (self->fscalars[i].type == NPY_FLOAT) {
      return PyUnicode_FromString("float");}
    else if (self->fscalars[i].type == NPY_CFLOAT) {
      return PyUnicode_FromString("float complex");}
    }

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    if (self->farrays[i].type == NPY_STRING) {
      charsize = (int)(self->farrays[i].dimensions[0]);
      sprintf(charstring,"character(%d)",charsize);
      return PyUnicode_FromString(charstring);}
    else if (self->farrays[i].type == NPY_LONG) {
      return PyUnicode_FromString("integer");}
    else if (self->farrays[i].type == NPY_DOUBLE) {
      return PyUnicode_FromString("double");}
    else if (self->farrays[i].type == NPY_CDOUBLE) {
      return PyUnicode_FromString("double complex");}
    else if (self->farrays[i].type == NPY_FLOAT) {
      return PyUnicode_FromString("float");}
    else if (self->farrays[i].type == NPY_CFLOAT) {
      return PyUnicode_FromString("float complex");}
    }

  returnnone;

}

/* ######################################################################### */
static void stringconcatanddel(PyObject **left,char *right)
{
  /* This is needed in order to properly handle the creation and destruction */
  /* of python string objects. */
  PyObject *pyright;
  PyObject *result;
  pyright = PyUnicode_FromString(right);
  result = PyUnicode_Concat(*left,pyright);
  Py_DECREF(pyright);
  Py_DECREF(*left);
  *left = result;
}
static void stringconcatanddellong(PyObject **left,long right)
{
  /* This is needed in order to properly handle the creation and destruction */
  /* of python string objects. */
  PyObject *pylong;
  PyObject *pyright;
  PyObject *result;
#if PY_MAJOR_VERSION >= 3
  pylong = PyLong_FromLong(right);
#else
  pylong = PyInt_FromLong(right);
#endif
  pyright = PyObject_Str(pylong);
  result = PyUnicode_Concat(*left,pyright);
  Py_DECREF(pylong);
  Py_DECREF(pyright);
  Py_DECREF(*left);
  *left = result;
}

/* ######################################################################### */
/* # Print information about the variable name.                              */
static char listvar_doc[] = "Returns information about a variable";
static PyObject *ForthonPackage_listvar(PyObject *_self_,PyObject *args)
{
  ForthonObject *self = (ForthonObject *)_self_;
  PyObject *pyi;
  PyObject *doc;
  int i,j,charsize;
  char *name;
  char charstring[50];
  if (!PyArg_ParseTuple(args,"s",&name)) return NULL;

  /* The PyUnicode stuff is done to avoid having to deal with strings at the
     C level, which would require explicit memory allocations (yuck!) */

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->scalardict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    doc = PyUnicode_FromString("");
    stringconcatanddel(&doc,"Package:    ");
    stringconcatanddel(&doc,self->name);
    stringconcatanddel(&doc,"\nGroup:      ");
    stringconcatanddel(&doc,self->fscalars[i].group);
    stringconcatanddel(&doc,"\nAttributes:");
    stringconcatanddel(&doc,self->fscalars[i].attributes);
    stringconcatanddel(&doc,"\nType:       ");
    if (self->fscalars[i].type == NPY_STRING) {
      stringconcatanddel(&doc,"character");}
    else if (self->fscalars[i].type == NPY_LONG) {
      stringconcatanddel(&doc,"integer");}
    else if (self->fscalars[i].type == NPY_DOUBLE) {
      stringconcatanddel(&doc,"double");}
    else if (self->fscalars[i].type == NPY_CDOUBLE) {
      stringconcatanddel(&doc,"double complex");}
    else if (self->fscalars[i].type == NPY_FLOAT) {
      stringconcatanddel(&doc,"float");}
    else if (self->fscalars[i].type == NPY_CFLOAT) {
      stringconcatanddel(&doc,"float complex");}
    stringconcatanddel(&doc,"\nAddress:    ");
    if (self->fscalars[i].type == NPY_OBJECT)
      ForthonPackage_updatederivedtype(self,i,1);
    stringconcatanddellong(&doc,(long)(self->fscalars[i].data));
    stringconcatanddel(&doc,"\nComment:\n");
    stringconcatanddel(&doc,self->fscalars[i].comment);
    return doc;
    }

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItemString(self->arraydict,name);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"i",&i);
    doc = PyUnicode_FromString("");
    stringconcatanddel(&doc,"Package:    ");
    stringconcatanddel(&doc,self->name);
    stringconcatanddel(&doc,"\nGroup:      ");
    stringconcatanddel(&doc,self->farrays[i].group);
    stringconcatanddel(&doc,"\nAttributes:");
    stringconcatanddel(&doc,self->farrays[i].attributes);
    stringconcatanddel(&doc,"\nDimension:  ");
    stringconcatanddel(&doc,self->farrays[i].dimstring);
    stringconcatanddel(&doc,"\n            (");
    for (j=0;j<self->farrays[i].nd;j++) {
      stringconcatanddellong(&doc,(long)(self->farrays[i].dimensions[j]));
      if (j < self->farrays[i].nd-1)
        stringconcatanddel(&doc,", ");
      }
    stringconcatanddel(&doc,")");

    stringconcatanddel(&doc,"\nType:       ");
    if (self->farrays[i].type == NPY_STRING) {
      charsize = (int)(self->farrays[i].dimensions[0]);
      sprintf(charstring,"character(%d)",charsize);
      stringconcatanddel(&doc,charstring);
      /* stringconcatanddellong(&doc,(long)(self->farrays[i].dimensions[0])); */
      }
    else if (self->farrays[i].type == NPY_LONG) {
      stringconcatanddel(&doc,"integer");}
    else if (self->farrays[i].type == NPY_DOUBLE) {
      stringconcatanddel(&doc,"double");}
    else if (self->farrays[i].type == NPY_CDOUBLE) {
      stringconcatanddel(&doc,"double complex");}
    else if (self->farrays[i].type == NPY_FLOAT) {
      stringconcatanddel(&doc,"float");}
    else if (self->farrays[i].type == NPY_CFLOAT) {
      stringconcatanddel(&doc,"float complex");}

    stringconcatanddel(&doc,"\nAddress:    ");
    if (self->farrays[i].pya == NULL) {
      stringconcatanddel(&doc,"unallocated");}
    else {
      stringconcatanddellong(&doc,(long)(PyArray_BYTES(self->farrays[i].pya)));}

    stringconcatanddel(&doc,"\nPyaddress:  ");
    if ((self->farrays[i].pya) == 0)
      stringconcatanddel(&doc,"unallocated");
    else
      stringconcatanddellong(&doc,(long)(self->farrays[i].pya));

    stringconcatanddel(&doc,"\nComment:\n");
    stringconcatanddel(&doc,self->farrays[i].comment);
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
  Py_ssize_t pos=0;
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
  PyObject *key, *value, *pyi;
  Py_ssize_t pos=0;
  int e;
  if (!PyArg_ParseTuple(args,"O",&dict)) return NULL;
  /* There is something wrong with this code XXX */
  /* Set the object name if it is now "pointee"*/
  /* if (strcmp(self->name,"pointee") == 0) { */
    /* PyMem_Free(self->name); */
    /* self->name = PyUnicode_AS_DATA(PyDict_GetItemString(dict,"_name")); */
    /* } */
  /* First set the scalars so that the array dimensions are set. */
  while (PyDict_Next(dict,&pos,&key,&value)) {
    if (value == Py_None) continue;
    pyi = PyDict_GetItem(self->scalardict,key);
    if (pyi != NULL) {
      e = Forthon_setattro(self,key,value);
      if (e==0) continue;
      PyErr_Clear();
      }
    }
  /* Now arrays can be set. */
  /* This is done this way since the setarraypointer routine uses the farray */
  /* dimensions instead of the dimensions from the PyArrayObject.       */
  pos = 0;
  while (PyDict_Next(dict,&pos,&key,&value)) {
    if (value == Py_None) continue;
    pyi = PyDict_GetItem(self->arraydict,key);
    if (pyi != NULL) {
      e = Forthon_setattro(self,key,value);
      if (e==0) continue;
      PyErr_Clear();
      }
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
/* Methods which are callable as attributes of a Forthon object            */
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
  {"__setstate__",(PyCFunction)ForthonPackage_setdict,1,setdict_doc},
  {"totmembytes" ,(PyCFunction)ForthonPackage_totmembytes,1,totmembytes_doc},
  {"varlist"     ,(PyCFunction)ForthonPackage_varlist,1,varlist_doc},
  {"getstrides"  ,(PyCFunction)ForthonPackage_getstrides,1,getstrides_doc},
  {"printtypenum"  ,(PyCFunction)ForthonPackage_printtypenum,1,printtypenum_doc},
  {"feenableexcept"  ,(PyCFunction)ForthonPackage_feenableexcept,1,feenableexcept_doc},
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
  /* Py_TYPE(self)->tp_free((PyObject*)self); */
}

/* ######################################################################### */
/* # Get attribute handler                                                   */
#if PY_MAJOR_VERSION >= 3
#define CMPSTR(s) PyUnicode_CompareWithASCIIString(oname,s)
#else
#define CMPSTR(s) strcmp(name,s)
#endif
static PyObject *Forthon_getattro(ForthonObject *self,PyObject *oname)
{
  long i;
  PyObject *pyi;
  PyMethodDef *ml;
#if PY_MAJOR_VERSION < 3
  char *name;
#endif

  /* Get index for variable from scalar dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItem(self->scalardict,oname);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"l",&i);
    if (self->fscalars[i].getaction != NULL) {
      if (self->fobj == NULL) self->fscalars[i].getaction();
      else                    self->fscalars[i].getaction((self->fobj));
      }
    if (self->fscalars[i].type == NPY_DOUBLE) {
      return Forthon_getscalardouble(self,(void *)i);}
    else if (self->fscalars[i].type == NPY_CDOUBLE) {
      return Forthon_getscalarcdouble(self,(void *)i);}
    else if (self->fscalars[i].type == NPY_FLOAT) {
      return Forthon_getscalarfloat(self,(void *)i);}
    else if (self->fscalars[i].type == NPY_CFLOAT) {
      return Forthon_getscalarcfloat(self,(void *)i);}
    else if (self->fscalars[i].type == NPY_OBJECT) {
      return Forthon_getscalarderivedtype(self,(void *)i);}
    else {
      return Forthon_getscalarinteger(self,(void *)i);}
    }

  /* Get index for variable from array dictionary */
  /* If it is not found, the pyi is returned as NULL */
  pyi = PyDict_GetItem(self->arraydict,oname);
  if (pyi != NULL) {
    PyArg_Parse(pyi,"l",&i);
    if (self->farrays[i].getaction != NULL) {
      if (self->fobj == NULL) self->farrays[i].getaction();
      else                    self->farrays[i].getaction((self->fobj));
      }
    return Forthon_getarray(self,(void *)i);}

  /* Now convert oname into the actual string, checking for errors. */
#if PY_MAJOR_VERSION < 3
  name = PyString_AsString(oname);
  if (name == NULL) return NULL;
#endif

  /* Check if asking for one of the dictionaries or other names*/
  /* Note that these should probably not be accessable */
  if (CMPSTR("scalardict") == 0) {
    Py_INCREF(self->scalardict);
    return self->scalardict;
    }
  if (CMPSTR("arraydict") == 0) {
    Py_INCREF(self->arraydict);
    return self->arraydict;
    }
  if (CMPSTR("__module__") == 0) {
    Py_INCREF(self->__module__);
    return self->__module__;
    }

  /* The code here used to be handled by calling Py_FindMethod, but */
  /* that is not defined in python3 */
  /* Look through the Forthon generic methods */
  ml = getForthonPackage_methods();
  for (; ml->ml_name != NULL; ml++) {
    if (CMPSTR(ml->ml_name) == 0) {
      return (PyObject *)PyCFunction_New(ml,(PyObject *)self);
    }
  }
  /* Look through the object specific methods */
  ml = self->fmethods;
  for (; ml->ml_name != NULL; ml++) {
    if (CMPSTR(ml->ml_name) == 0) {
      return (PyObject *)PyCFunction_New(ml,(PyObject *)self);
    }
  }

  /* The last resort, the standard getattr */
  return PyObject_GenericGetAttr((PyObject *)self,oname);
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
    if (self->fscalars[i].parameter) {
      PyErr_SetString(PyExc_TypeError, "Cannot set a parameter");
      return -1;
      }
    if (self->fscalars[i].type == NPY_DOUBLE) {
      return Forthon_setscalardouble(self,v,(void *)i);}
    else if (self->fscalars[i].type == NPY_CDOUBLE) {
      return Forthon_setscalarcdouble(self,v,(void *)i);}
    else if (self->fscalars[i].type == NPY_FLOAT) {
      return Forthon_setscalarfloat(self,v,(void *)i);}
    else if (self->fscalars[i].type == NPY_CFLOAT) {
      return Forthon_setscalarcfloat(self,v,(void *)i);}
    else if (self->fscalars[i].type == NPY_OBJECT) {
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
  char v[120];
  PyObject *s;
  sprintf(v,"<%s instance at address = %ld>",self->name,(long)self);
  s = Py_BuildValue("s",v);
  return s;
}

/* ######################################################################### */
/* # Package object declaration                                              */
static PyTypeObject ForthonType = {
  PyVarObject_HEAD_INIT(NULL, 0)         /*ob_size*/
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
