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
%fname('gallot')+'(FSTRING name,int *iverbose SL1)'
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
%fname('gchange')+'(FSTRING name,int *iverbose SL1)'
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
%fname('kaboom')+'(int *e)'
{
printf("KABOOM! Something bad happened\n");
exit(1);
}

void
%fname('glbheadi')+'(int dum1,int dum2,int runtime,int rundate,int dum3,int dum4)'
{}

void
%fname('slbasis')+'(FSTRING s,int *v SL1)'
{}

void
%fname('hstall')+'(FSTRING s,int *v SL1)'
{
/* collector.sample (cycle, time) */
}

#include <sys/times.h>
#include <unistd.h>

void
%fname('ostime')
%py_ifelse(f90 or f90f,1,'(double *cpu, double *io, double *sys, double *mem)')
%py_ifelse(f90 or f90f,0,py_ifelse(machine,'T3E','(double *cpu, double *io, double *sys, double *mem)','(float *cpu, float *io, float *sys, float *mem)'))
{
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
%fname('setshape')+'(FSTRING s,int *v SL1)'
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
%fname('edit')+'(int *iunit,FSTRING s SL1)'
{
/* place holder for now - this should probably be written */
}
void
%fname('outfile')+'(int *iunit,FSTRING s SL1)'
{
/* place holder for now - this should probably be written */
}
