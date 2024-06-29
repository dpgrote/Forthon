/* ######################################################################### */
/* There are routines which are callable from Fortran                         */

#include "Python.h"
#include "forthonf2c.h"

static char* cstrfromfstr(char *fstr,int fstrlen)
{
  char* cname;
  cname = (char *) PyMem_Malloc((fstrlen+1)*sizeof(char));
  cname[fstrlen] = (char)0;
  memcpy(cname,fstr,fstrlen);
  return cname;
}

void
%fname('gallot')+'(FSTRING name,long *iverbose SL1)'
{
  char *cname;
  PyObject *m, *d, *f, *r;
  cname = cstrfromfstr(FSTRPTR(name),FSTRLEN1(name));

  m = PyImport_ImportModule("Forthon");
  if (m != NULL) {
    d = PyModule_GetDict(m);
    if (d != NULL) {
      f = PyDict_GetItemString(d,"gallot");
      if (f != NULL) {
        r = PyObject_CallFunction(f,"si",cname,*iverbose);
        Py_XDECREF(r);
  }}}
  Py_XDECREF(m);

  PyMem_Free(cname);
  if (PyErr_Occurred()) PyErr_Print();
}

void
%fname('gchange')+'(FSTRING name,long *iverbose SL1)'
{
  char *cname;
  PyObject *m, *d, *f, *r;
  cname = cstrfromfstr(FSTRPTR(name),FSTRLEN1(name));

  m = PyImport_ImportModule("Forthon");
  if (m != NULL) {
    d = PyModule_GetDict(m);
    if (d != NULL) {
      f = PyDict_GetItemString(d,"gchange");
      if (f != NULL) {
        r = PyObject_CallFunction(f,"si",cname,*iverbose);
        Py_XDECREF(r);
  }}}
  Py_XDECREF(m);

  PyMem_Free(cname);
}

void
%fname('gsetdims')+'(FSTRING name SL1)'
{
  char *cname;
  PyObject *m, *d, *f, *r;
  cname = cstrfromfstr(FSTRPTR(name),FSTRLEN1(name));

  m = PyImport_ImportModule("Forthon");
  if (m != NULL) {
    d = PyModule_GetDict(m);
    if (d != NULL) {
      f = PyDict_GetItemString(d,"gsetdims");
      if (f != NULL) {
        r = PyObject_CallFunction(f,"s",cname);
        Py_XDECREF(r);
  }}}
  Py_XDECREF(m);

  PyMem_Free(cname);
}

void
%fname('gfree')+'(FSTRING name SL1)'
{
  char *cname;
  PyObject *m, *d, *f, *r;
  cname = cstrfromfstr(FSTRPTR(name),FSTRLEN1(name));

  m = PyImport_ImportModule("Forthon");
  if (m != NULL) {
    d = PyModule_GetDict(m);
    if (d != NULL) {
      f = PyDict_GetItemString(d,"gfree");
      if (f != NULL) {
        r = PyObject_CallFunction(f,"s",cname);
        Py_XDECREF(r);
  }}}
  Py_XDECREF(m);

  PyMem_Free(cname);
}

/* The following routines are used when dealing with fortran derived types. */
void
%fname('tallot')+'(PyObject **self)'
{
  PyObject *pname, *f, *r;
  pname = Py_BuildValue("s","gallot");
  f = PyObject_GetAttr(*self,pname);
  if (f != NULL) {
    r = PyObject_CallFunction(f,"s","*");
    Py_XDECREF(f);
    Py_XDECREF(r);
  }
  Py_DECREF(pname);
}

void
%fname('tchange')+'(PyObject **self)'
{
  PyObject *pname, *f, *r;
  pname = Py_BuildValue("s","gchange");
  f = PyObject_GetAttr(*self,pname);
  if (f != NULL) {
    r = PyObject_CallFunction(f,"s","*");
    Py_XDECREF(f);
    Py_XDECREF(r);
  }
  Py_DECREF(pname);
}

void
%fname('tfree')+'(PyObject **self)'
{
  PyObject *pname, *f, *r;
  pname = Py_BuildValue("s","gfree");
  f = PyObject_GetAttr(*self,pname);
  if (f != NULL) {
    r = PyObject_CallFunction(f,"s","*");
    Py_XDECREF(f);
    Py_XDECREF(r);
  }
  Py_DECREF(pname);
}

/* ---------------------------------------------------------------------- */

void
%fname('remark')+'(FSTRING text SL1)'
{
  char *ctext;
  PyObject *pystdout;
  ctext = cstrfromfstr(FSTRPTR(text),FSTRLEN1(text));
  pystdout = PySys_GetObject("stdout");
  PyFile_WriteString(ctext,pystdout);
  PyFile_WriteString("\n",pystdout);
  PyMem_Free(ctext);
}

void
%fname('parsestr')+'(FSTRING fstr SL1)'
{
  char *cfstr;
  cfstr = (char *) PyMem_Malloc((FSTRLEN1(fstr)+1)*sizeof(char));
  memcpy(cfstr,FSTRPTR(fstr),FSTRLEN1(fstr));
  cfstr[FSTRLEN1(fstr)+0] = (char)0;
  PyRun_SimpleString(cfstr);
  PyMem_Free(cfstr);
}

void
%fname('execuser')+'(FSTRING fstr SL1)'
{
  char *cfstr;
  cfstr = (char *) PyMem_Malloc((FSTRLEN1(fstr)+3)*sizeof(char));
  memcpy(cfstr,FSTRPTR(fstr),FSTRLEN1(fstr));
  if (cfstr[FSTRLEN1(fstr)-1]==')') {
    cfstr[FSTRLEN1(fstr)+0] = (char)0;
    }
  else {
    cfstr[FSTRLEN1(fstr)+0] = '(';
    cfstr[FSTRLEN1(fstr)+1] = ')';
    cfstr[FSTRLEN1(fstr)+2] = (char)0;
    }
  PyRun_SimpleString(cfstr);
  PyMem_Free(cfstr);
}

int
%fname('utgetcl')+'(FSTRING w SL1)'
{
  char *s;
  int l;
  s = FSTRPTR(w);
  l = FSTRLEN1(w);
  while (l > 0) {
    if (s[l-1] != ' ') return l;
    l--;}
  return l+1;
}

void
%fname('glbheadi')+'(long dum1,long dum2,long runtime,long rundate,long dum3,long dum4)'
{}

void
%fname('slbasis')+'(FSTRING s,long *v SL1)'
{}

void
%fname('hstall')+'(FSTRING s,long *v SL1)'
{
/* collector.sample (cycle, time) */
}

#include <sys/times.h>
#include <unistd.h>

void
%fname('ostime')
(double *cpu, double *io, double *sys, double *mem)
{
  /*
  double utime,stime;
  struct tms usage;
  long hardware_ticks_per_second;
  (void) times(&usage);
  hardware_ticks_per_second = sysconf(_SC_CLK_TCK);
  utime = (double) usage.tms_utime/hardware_ticks_per_second;
  stime = (double) usage.tms_stime/hardware_ticks_per_second;

  *cpu = (double) utime;
  *io  = (double) 0.;
  *sys = (double) stime;
  *mem = (double) 0.;
  */

  /* This should be more robust, since it lets python take care of the */
  /* portability. */
  PyObject *m, *d, *f, *r;
  m = PyImport_ImportModule("time");
  if (m != NULL) {
    d = PyModule_GetDict(m);
    if (d != NULL) {
      f = PyDict_GetItemString(d,"clock");
      if (f != NULL) {
        r = PyObject_CallFunction(f,NULL);
        *cpu = PyFloat_AS_DOUBLE(r);
        Py_XDECREF(r);
  }}}
  Py_XDECREF(m);
  *io  = (double) 0.;
  *sys = (double) 0.;
  *mem = (double) 0.;
}

double
%fname('py_tremain')+'(void)'
{
/* This is obsolete */
  return (double)(1.e36);
}

void
%fname('setshape')+'(FSTRING s,long *v SL1)'
{}
void
%fname('parsetuf')+'(void)'
{}
void
%fname('rtserv')+'(void)'
{}
void
%fname('glbpknam')+'(void)'
{}
void
%fname('change')+'(void)'
{}
void
%fname('edit')+'(long *iunit,FSTRING s SL1)'
{
/* place holder for now - this should probably be written */
}
void
%fname('outfile')+'(long *iunit,FSTRING s SL1)'
{
/* place holder for now - this should probably be written */
}

/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */
/* This routine is for error handling. setjmp is called from               */
/* the wrappers of the fortran routines, just before the fortran routines  */
/* are called. kaboom is called by the fortran routines if there is an     */
/* error. It raises a python exception and calls longjmp so that it        */
/* returns to the point where setjmp was called. setjmp will then return   */
/* an error value which the wrapper checks. It there was an exception,     */
/* the the wrapper returns.                                                */
/* Note that stackenvironment is used as a global and is what is passed */
/* into the calls to setjmp. */

#include <setjmp.h>
jmp_buf stackenvironment;
int lstackenvironmentset;

void
%fname('kaboom')+'(FSTRING message SL1)'
{
  char *errormessage;
  errormessage = cstrfromfstr(FSTRPTR(message),FSTRLEN1(message));
  PyErr_SetString(PyExc_RuntimeError,errormessage);
  PyMem_Free(errormessage);
  lstackenvironmentset = 0;
  longjmp(stackenvironment,1);
  /* exit(1); */
}

/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */

void
%fname('callpythonfunc')+'(FSTRING fname,FSTRING mname SL1 SL2)'
{
  char *cfname,*cmname,*cpname;
  PyObject *modules;
  PyObject *m=NULL;
  PyObject *d=NULL;
  PyObject *f=NULL;
  PyObject *r=NULL;
  int m_is_borrowed = 1;
  char *errormessage=NULL;
  cfname = (char *) PyMem_Malloc((FSTRLEN1(fname)+1)*sizeof(char));
  cmname = (char *) PyMem_Malloc((FSTRLEN2(mname)+1)*sizeof(char));

  // Copy the module and function names from the fortran data. Note that
  // strings in fortran do not have a null termination and it must be
  // explicitly added here.
  memcpy(cfname,FSTRPTR(fname),FSTRLEN1(fname));
  memcpy(cmname,FSTRPTR(mname),FSTRLEN2(mname));
  cfname[FSTRLEN1(fname)] = (char)0;
  cmname[FSTRLEN2(mname)] = (char)0;

  // Some fancy footwork is needed to find the module to import. Depending
  // on how the modules are setup, the module's name in python could either
  // be modulename or pkg.modulename. The latter if using an egg or if the
  // pkg scripts are installed in site-packages. The module should have
  // already been imported, so get it from ModuleDict (same as sys.modules).
  // First look for it under the name modulename. If it is not found try
  // pkg.modulename. If it is still not found, then try importing it.
  // Note that the module reference from the module dict is borrowed (and
  // should not be decremented), but the one from ImportModule is new (and
  // should be decremented). Once the module is found, the function can
  // be called.
  modules = PyImport_GetModuleDict();
  m = PyDict_GetItemString(modules,cmname);

  if (m == NULL) {
    // Try pkg.modulename
    cpname = (char *) PyMem_Malloc((FSTRLEN2(mname)+strlen(FORTHON_PKGNAME)+2)*sizeof(char));
    strcpy(cpname,FORTHON_PKGNAME);
    strcat(cpname,".");
    strcat(cpname,cmname);
    m = PyDict_GetItemString(modules,cpname);
    PyMem_Free(cpname);
    }

  if (m == NULL) {
    // Still not found, so try directly importing
    m = PyImport_ImportModule(cmname);
    m_is_borrowed = 0;
    }

  // Call the function (checking for errors)
  if (m != NULL) {
    d = PyModule_GetDict(m);
    if (d != NULL) {
      f = PyDict_GetItemString(d,cfname);
      if (f != NULL) {
        r = PyObject_CallFunction(f,NULL);
        }
      }
    }

  if (m == NULL) {
    // Module not found, so raise an exception and return the same way that kaboom does
    if (PyErr_Occurred() == NULL) {
      char *errorstring = "callpythonfunc: %s module could not be found";
      errormessage = (char *) PyMem_Malloc((strlen(errorstring)+strlen(cmname)+1)*sizeof(char));
      sprintf(errormessage,errorstring,cmname);
      }
    goto err;
    }

  if (d == NULL) {
    // Module dictionary not found, so raise an exception and return the same way that kaboom does
    // This should never happen.
    if (PyErr_Occurred() == NULL) {
      char *errorstring = "callpythonfunc: %s module's dictionary could not be found";
      errormessage = (char *) PyMem_Malloc((strlen(errorstring)+strlen(cmname)+1)*sizeof(char));
      sprintf(errormessage,errorstring,cmname);
      }
    goto err;
    }

  if (f == NULL) {
    // Function not found, so raise an exception and return the same way that kaboom does
    if (PyErr_Occurred() == NULL) {
      char *errorstring = "callpythonfunc: %s.%s function could not be found";
      errormessage = (char *) PyMem_Malloc((strlen(errorstring)+strlen(cfname)+strlen(cmname)+1)*sizeof(char));
      sprintf(errormessage,errorstring,cmname,cfname);
      }
    goto err;
    }

  if (r == NULL) {
    // Function returned an error, so raise an exception and return the same way that kaboom does
    if (PyErr_Occurred() == NULL) {
      char *errorstring = "callpythonfunc: %s.%s function had an error";
      errormessage = (char *) PyMem_Malloc((strlen(errorstring)+strlen(cfname)+strlen(cmname)+1)*sizeof(char));
      sprintf(errormessage,errorstring,cmname,cfname);
      }
    goto err;
    }

  PyMem_Free(cfname);
  PyMem_Free(cmname);
  if (!m_is_borrowed) {
    Py_XDECREF(m);
    }
  Py_XDECREF(r);

  return;

err:
  if (errormessage != NULL) {
    PyErr_SetString(PyExc_RuntimeError,errormessage);
    PyMem_Free(errormessage);
    }
  PyMem_Free(cfname);
  PyMem_Free(cmname);
  lstackenvironmentset = 0;
  longjmp(stackenvironment,1);
}

