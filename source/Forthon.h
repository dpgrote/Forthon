/* Created by David P. Grote, March 6, 1998 */

#include <Python.h>

#if PY_VERSION_HEX < 0x02050000 && !defined(PY_SSIZE_T_MIN)
typedef int Py_ssize_t;
#define PY_SSIZE_T_MAX INT_MAX
#define PY_SSIZE_T_MIN INT_MIN
#endif

#define NPY_NO_DEPRECATED_API 8
#define PY_ARRAY_UNIQUE_SYMBOL Forthon_ARRAY_API
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

extern PyObject *ErrorObject;

PyMODINIT_FUNC Forthon_import_array();

#define returnnone {Py_INCREF(Py_None);return Py_None;}

extern PyTypeObject ForthonType;

/* This converts a python object into a python array,         */
/* requesting fortran ordering.                               */
extern PyArrayObject* FARRAY_FROMOBJECT(PyObject *A2, int ARRAY_TYPE);

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
  char* unit;
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
  char* unit;
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

/* ######################################################################### */
/* ######################################################################### */
/* ######################################################################### */
/* This variable is used to keep track of the total amount of memory         */
/* dynamically allocated in the package.                                     */
/* This is intentionally static so that each package will have its own copy. */
static long totmembytes=0;

/* ###################################################################### */
/* Utility routines used in wrapping the subroutines                      */
/* It checks if the argument can be cast to the desired type.             */
extern int Forthon_checksubroutineargtype(PyObject *pyobj,int type_num);

/* ###################################################################### */
/* In some cases, when an array is passed from Python to Fortran, for example */
/* if the Python array is not contiguous or not in Fortran ordering, a temporary */
/* copy of the array is made and passed into Fortrh. This routine copies */
/* the data back into the original Python array after the Fortran routine finishes. */
extern void Forthon_restoresubroutineargs(int n,PyObject **pyobj, PyArrayObject **ax);

/* ###################################################################### */
/* Builds a scalar and an array dictionary for the package. The           */
/* dictionaries are then used in the getattr and setattr to look up the   */
/* indices given a variable name. That lookup is faster than a linear     */
/* scan through the list of variables.                                    */
extern void Forthon_BuildDicts(ForthonObject *self);
extern void Forthon_DeleteDicts(ForthonObject *self);

/* ######################################################################### */
/* # Allocate the dimensions element for all of the farrays in the object.   */
extern void ForthonPackage_allotdims(ForthonObject *self);

/* ######################################################################### */
/* Static array initialization routines. Create a numpy array for each */
/* static array. */
extern void ForthonPackage_staticarrays(ForthonObject *self);

extern PyObject *ForthonPackage_gallot(PyObject *_self_,PyObject *args);

#include <setjmp.h>
extern jmp_buf stackenvironment;
extern int lstackenvironmentset;

