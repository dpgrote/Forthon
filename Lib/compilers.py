"""Class which determines which fortran compiler to use and sets defaults
for it.
"""

import sys,os,re
import string

class FCompiler:
  """
Determines which compiler to use and sets up description of how to use it.
To add a new compiler, create a new method which a name using the format
machine_compiler. The first line of the function must be of the following form
    if (self.findfile(compexec) and
        (self.fcompiler==compname or self.fcompiler is None)):
where compexec is the executable name of the compiler and compname is a
descriptive (or company) name for the compiler. They can be the same.
In the function, the two attributes f90free and f90fixed must be defined. Note
that the final linking is done with gcc, so any fortran libraries will need to
be added to libs (and their locations to libdirs).
Also, the new function must be included in the while loop below in the
appropriate block for the machine.
  """

  def __init__(self,machine=None,debug=0,fcompiler=None,static=0):
    if machine is None: machine = sys.platform
    self.machine = machine

    self.paths = string.split(os.environ['PATH'],os.pathsep)

    self.fcompiler = fcompiler
    self.static = static
    self.defines = []
    self.fopt = ''
    self.popts = ''
    self.libs = []
    self.libdirs = []
    self.pywrapperargs = ''

    # --- Pick the fortran compiler
    # --- When adding a new compiler, it must be listed here under the correct
    # --- machine name.
    while 1:
      if self.machine == 'linux2':
        if self.linux_intel8() is not None: break
        if self.linux_intel() is not None: break
        if self.linux_pg() is not None: break
        if self.linux_absoft() is not None: break
      elif self.machine == 'darwin':
        if self.macosx_absoft() is not None: break
        if self.macosx_nag() is not None: break
      elif self.machine == 'win32':
        if self.win32_pg() is not None: break
        if self.win32_intel() is not None: break
      elif self.machine == 'aix4':
        self.static = 1
        if self.aix_mpxlf() is not None: break
        if self.aix_xlf() is not None: break
        if self.aix_xlf_r() is not None: break
      else:
        raise SystemExit,'Machine type %s is unknown'%self.machine
      raise SystemExit,'Fortran compiler not found'

    # --- The following two quantities must be defined.
    try:
      self.f90free
      self.f90fixed
    except:
      # --- Note that this error should never happed (except during debugging)
      raise "The fortran compiler definition is not correct, f90free and f90fixed must be defined."

    if debug: self.fopt = '-g'

  def findfile(self,file):
    if self.machine == 'win32': file = file + '.exe'
    for path in self.paths:
      try:
        if file in os.listdir(path): return path
      except:
        pass
    return None

  #-----------------------------------------------------------------------------
  # --- LINUX
  def linux_intel8(self):
    if (self.findfile('ifort') and
        (self.fcompiler=='intel8' or self.fcompiler is None)):
      # --- Intel8
      self.f90free  = 'ifort -nofor_main -free -r8 -DIFC -fpp -implicitnone -C90 -Zp8'
      self.f90fixed = 'ifort -nofor_main -132 -r8 -DIFC -fpp -implicitnone -C90 -Zp8'
      self.popts = '-O'
      flibroot,b = os.path.split(self.findfile('ifort'))
      self.libdirs = [flibroot+'/lib']
      self.libs = ['ifcore','ifport','imf','svml','cxa','irc','unwind']
      cpuinfo = open('/proc/cpuinfo','r').read()
      if re.search('Pentium III',cpuinfo):
        self.fopt = '-O3 -xK -tpp6 -ip -unroll -prefetch'
      else:
        self.fopt = '-O3 -xW -tpp7 -ip -unroll -prefetch'
      return 1

  def linux_intel(self):
    if (self.findfile('ifc') and
        (self.fcompiler=='intel' or self.fcompiler is None)):
      # --- Intel
      self.f90free  = 'ifc -132 -r8 -DIFC -fpp -implicitnone -C90 -Zp8'
      self.f90fixed = 'ifc -132 -r8 -DIFC -fpp -implicitnone -C90 -Zp8'
      self.popts = '-O'
      flibroot,b = os.path.split(self.findfile('ifc'))
      self.libdirs = [flibroot+'/lib']
      self.libs = ['IEPCF90','CEPCF90','F90','intrins','imf','svml','irc','cxa']
      cpuinfo = open('/proc/cpuinfo','r').read()
      if re.search('Pentium III',cpuinfo):
        self.fopt = '-O3 -xK -tpp6 -ip -unroll -prefetch'
      else:
        self.fopt = '-O3 -xW -tpp7 -ip -unroll -prefetch'
      return 1

  def linux_pg(self):
    if (self.findfile('pgf90') and
        (self.fcompiler=='pg' or self.fcompiler is None)):
      # --- Portland group
      self.f90free  = 'pgf90 -Mextend -Mdclchk -r8'
      self.f90fixed = 'pgf90 -Mextend -Mdclchk -r8'
      self.popts = '-Mcache_align'
      flibroot,b = os.path.split(self.findfile('pgf90'))
      self.libdirs = [flibroot+'/lib']
      self.libs = ['pgf90'] # ???
      self.fopt = '-fast -Mcache_align'

  def linux_absoft(self):
    if (self.findfile('f90') and
        (self.fcompiler=='absoft' or self.fcompiler is None)):
      # --- Absoft
      self.f90free  = 'f90 -B108 -N113 -W132 -YCFRL=1 -YEXT_NAMES=ASIS'
      self.f90fixed = 'f90 -B108 -N113 -W132 -YCFRL=1 -YEXT_NAMES=ASIS'
      self.pywrapperargs = '--2underscores'
      flibroot,b = os.path.split(self.findfile('f90'))
      self.libdirs = [flibroot+'/lib']
      self.libs = ['U77','V77','f77math','f90math','fio']
      self.fopt = '-O'
      return 1

  #-----------------------------------------------------------------------------
  # --- MAC OSX
  def macosx_absoft(self):
    if (self.findfile('f90') and
        (self.fcompiler=='absoft' or self.fcompiler is None)):
      # --- Absoft
      self.f90free  = 'f90 -N11 -N113 -YEXT_NAMES=LCS -YEXT_SFX=_'
      self.f90fixed = 'f90 -f fixed -W 132 -N11 -N113 -YEXT_NAMES=LCS -YEXT_SFX=_'
      flibroot,b = os.path.split(self.findfile('pgf90'))
      self.libdirs = [flibroot+'/lib']
      self.libs = ['fio','f77math','f90math','f90math_altivec']
      self.fopt = '-O2'
      return 1

  def macosx_nag(self):
    if (self.findfile('f95') and
        (self.fcompiler=='nag' or self.fcompiler is None)):
      # --- NAG
      self.f90free  = 'f95 -132 -fpp -Wp,-macro=no_com -free -PIC -w -mismatch_all -kind=byte -r8'
      self.f90fixed = 'f95 -132 -fpp -Wp,-macro=no_com -Wp,-fixed -fixed -PIC -w -mismatch_all -kind=byte -r8'
      flibroot,b = os.path.split(self.findfile('f95'))
      self.libdirs = [flibroot+'/lib']
      self.libs = ['???']
      self.fopt = '-Wc,-O3 -Wc,-funroll-loops -O3 -Ounroll=2'
      return 1

  #-----------------------------------------------------------------------------
  # --- WIN32
  def win32_pg(self):
    if (self.findfile('pgf90') and
        (self.fcompiler=='pg' or self.fcompiler is None)):
      # --- Portland group
      self.f90free  = 'pgf90 -Mextend -Mdclchk -r8'
      self.f90fixed = 'pgf90 -Mextend -Mdclchk -r8'
      self.popts = '-Mcache_align'
      flibroot,b = os.path.split(self.findfile('pgf90'))
      self.libdirs = [flibroot+'/Lib']
      self.libs = ['???']
      self.fopt = '-fast -Mcache_align'
      return 1

  def win32_intel(self):
    if (self.findfile('ifl') and
        (self.fcompiler=='intel' or self.fcompiler is None)):
      # --- Intel
      self.f90free  = 'ifl -Qextend_source -Qautodouble -DIFC -FR -Qfpp -4Yd -C90 -Zp8 -Qlowercase -us -MT -Zl -static'
      self.f90fixed = 'ifl -Qextend_source -Qautodouble -DIFC -FI -Qfpp -4Yd -C90 -Zp8 -Qlowercase -us -MT -Zl -static'
      flibroot,b = os.path.split(self.findfile('ifl'))
      self.libdirs = [flibroot+'/Lib']
      self.libs = ['CEPCF90MD','F90MD','intrinsMD']
      self.fopt = '-O3'
      return 1

  #-----------------------------------------------------------------------------
  # --- AIX
  def aix_mpxlf(self):
    if (self.fcompiler=='mpxlf' or
        (self.fcompiler is None and self.findfile('mpxlf95'))):
      # --- IBM SP, parallel
      self.f90free  = 'mpxlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -WF,-DMPIPARALLEL -qsuffix=f=f90:cpp=F90 -qfree=f90 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
      self.f90fixed = 'mpxlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -WF,-DMPIPARALLEL -qfixed=132 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
      self.popts = '-O'
      self.ld = 'mpxlf_r -bmaxdata:0x70000000 -bmaxstack:0x10000000 -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp'
      self.libs = ' $(PYMPI)/driver.o $(PYMPI)/patchedmain.o -L$(PYMPI) -lpympi -lpthread'
      self.defines = ['PYMPI=/usr/common/homes/g/grote/pyMPI']
      self.fopt = '-O3 -qstrict -qarch=pwr3 -qtune=pwr3'
      return 1

  def aix_xlf(self):
    if (self.fcompiler=='xlf' or
        (self.fcompiler is None and self.findfile('xlf95'))):
      # --- IBM SP, serial
      self.f90free  = 'xlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -qsuffix=f=f90:cpp=F90 -qfree=f90 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
      self.f90fixed = 'xlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -qfixed=132 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
      self.popts = '-O'
      self.ld = 'xlf -bmaxdata:0x70000000 -bmaxstack:0x10000000 -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp'
      self.libs = ['pthread']
      self.fopt = '-O3 -qstrict -qarch=pwr3 -qtune=pwr3'
      return 1

  def aix_xlf_r(self):
    if (self.fcompiler=='xlf_r' or
        (self.fcompiler is None and self.findfile('xlf90_r'))):
      # --- IBM SP, OpenMP
      self.f90free  = 'xlf90_r -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -qsuffix=f=f90:cpp=F90 -qfree=f90 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
      self.f90fixed = 'xlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -qfixed=132 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
      self.popts = '-O'
      self.ld = 'xlf90_r -bmaxdata:0x70000000 -bmaxstack:0x10000000 -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp'
      self.libs = ' -lpthread -lxlf90_r -lxlopt -lxlf -lxlsmp'
      self.fopt = '-O3 -qstrict -qarch=pwr3 -qtune=pwr3 -qsmp=omp'
      return 1

