/* ######################################################################### */
/* There are routines which are callable from Fortran                         */

#include "Python.h"
#include "forthonf2c.h"

static char* cstrfromfstr(char *fstr,int fstrlen)
{
  char* cname;
  cname = (char *) malloc((fstrlen+1)*sizeof(char));
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

  free(cname);
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

  free(cname);
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

  free(cname);
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

  free(cname);
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
  free(ctext);
}

void
%fname('parsestr')+'(FSTRING fstr SL1)'
{
  char *cfstr;
  cfstr = (char *) malloc((FSTRLEN1(fstr)+1)*sizeof(char));
  memcpy(cfstr,FSTRPTR(fstr),FSTRLEN1(fstr));
  cfstr[FSTRLEN1(fstr)+0] = (char)0;
  PyRun_SimpleString(cfstr);
  free(cfstr);
}

void
%fname('execuser')+'(FSTRING fstr SL1)'
{
  char *cfstr;
  cfstr = (char *) malloc((FSTRLEN1(fstr)+3)*sizeof(char));
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
  free(cfstr);
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
%py_ifelse(f90,1,'(double *cpu, double *io, double *sys, double *mem)')
%py_ifelse(f90,0,py_ifelse(machine,'T3E','(double *cpu, double *io, double *sys, double *mem)','(float *cpu, float *io, float *sys, float *mem)'))
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


%py_ifelse(machine,'T3E','#include <sys/types.h>','')
%py_ifelse(machine,'T3E','#include <sys/jtab.h>','')
%py_ifelse(machine,'T3E','#include <errno.h>','')

double
%fname('py_tremain')+'(void)'
{
%py_ifelse(machine,'T3E','','/*')
  job_t job;
  if (job_cntl(0, J_GET_ALL, (int)&job) != -1)
    return (double)(job.j_mpptimelimit - job.j_mpptimeused);
%py_ifelse(machine,'T3E','','*/')
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
char *errormessage;

void
%fname('kaboom')+'(FSTRING message SL1)'
{
  errormessage = cstrfromfstr(FSTRPTR(message),FSTRLEN1(message));
  PyErr_SetString(PyExc_RuntimeError,errormessage);
  free(errormessage);
  longjmp(stackenvironment,1);
  /* exit(1); */
}

/* ----------------------------------------------------------------------- */
/* ----------------------------------------------------------------------- */

void
%fname('callpythonfunc')+'(FSTRING fname,FSTRING mname SL1 SL2)'
{
  char *cfname,*cmname;
  PyObject *m,*d,*f,*r;
  cfname = (char *) malloc((FSTRLEN1(fname)+1)*sizeof(char));
  cmname = (char *) malloc((FSTRLEN2(mname)+1)*sizeof(char));
  memcpy(cfname,FSTRPTR(fname),FSTRLEN1(fname));
  memcpy(cmname,FSTRPTR(mname),FSTRLEN2(mname));
  cfname[FSTRLEN1(fname)] = (char)0;
  cmname[FSTRLEN2(mname)] = (char)0;
  m = PyImport_ImportModule(cmname);
  r = NULL;
  if (m != NULL) {
    d = PyModule_GetDict(m);
    if (d != NULL) {
      f = PyDict_GetItemString(d,cfname);
      if (f != NULL) {
        r = PyObject_CallFunction(f,NULL);
  }}}
  free(cfname);
  free(cmname);
  Py_XDECREF(m);
  if (r == NULL) {
    longjmp(stackenvironment,1);
  }
  else {
    Py_XDECREF(r);
  }
}

