#! python
"""Does everything that is needed to build an individual package.
"""

import sys,os,re
import getopt
import string
import distutils
import distutils.sysconfig
from distutils.core import setup, Extension

optlist,args = getopt.getopt(sys.argv[1:],'agd:t:F:C:D:L:l:i:f:',
                         ['f90','f77','f90f','nowritemodules','macros=',
                          'FOPTS','COPTS','static'])

if len(sys.argv) == 1:
  print """
Forthon [options] pkgname [extra files to be compiled]

pkgname is the name of the package.
A complete package will have at least two files, the interface description
file and the fortran file. The default name for the interface file is
pkgname.v.  Note that the first line of the interface file must be the package
name. The default name for the fortran file is pkgname.F

One or more of the following options can be specified.

 -a
    When specified, all groups will be allocated when package is imported
    into python. By default, groups are not allocated.
 -C command
    C compiler, one of gcc, icc.  Will automatically be determined if
    not supplied.
 -d package
    Specifies that a package that the package being built depends upon.
    This option can be specified multiple times.
 -D VAR=value
    Defines a macro which will be inserted into the makefile. This is required
    in some cases where a third party library must be specified. This can
    be specified multiple times.
 -F command
    Fortran compiler.  Will automatically be determined if not supplied.
    It can be one of the following, depending on the machine
    intel8, intel, pg, absort, nag, xlf, mpxlf, xlf_r
 -g
    Turns off optimization for fortran compiler
 -i filename
    Specify full name of interface file. It defaults to pkgname.v
 -f filename
    Specifiy full name of main fortran file. It defaults to pkgname.F
 -L path
    Addition library directories
 -l library
    Additional libraries that are needed. Note that the prefix 'lib' and any
    suffixes should not be included.
 -t type
    Machine type. Will automatically be determined if not supplied.
    Can be one of linux2, aix4, darwin, win32.
 --f90
    Writes wrapper code using f90, which means that python accessible
    variables are defined in f90 modules. This is the default.
 --f77
    Writes wrapper code using f77, which means that python accessible
    variables are defined in common blocks. This is obsolete and is not
    supported.
 --f90f
    Writes wrapper code using a variant of the f90 method. This is under
    development and should not be used.
 --nowritemodules
    Don't write out the module definitions. Useful if the modules have
    been written already. Note that if variables of derived type are used, the
    original code will need to be modified. See example2. Also note that
    if this option is used, no checks are made to ensure the consistency
    between the interface file description and the actual module.
 --macros pkg.v
    Other interface files whose macros are needed
 --FOPTS
    Additional options to the fortran compiler line. For example to turn on
    profiling.
 --COPTS
    Additional options to the C compiler line. For example to turn on
    profiling.
 --static
    Build the static version of the code by default, rather than the
    dynamically linker version. Not yet supported.
  """
  sys.exit(0)

# --- Get the package name
pkg = args[0]

# --- Get any other extra fortan files
if len(args) > 1:
  extrafiles = args[1:]
else:
  extrafiles = []

# --- Set default values for command line options
interfacefile = pkg + '.v'
fortranfile = pkg + '.F'
initialgallot = ''
dependencies = []
defines = []
machine = None
fcompiler = None
f90 = '--f90'
f90f = 0
writemodules = 1
othermacros = []
debug = 0
fopts = ''
copts = ''
libs = []
libdirs = []
static = 0

for o in optlist:
  if o[0]=='-a': initialgallot = '-a'
  elif o[0] == '-g': debug = 1
  elif o[0] == '-t': machine = o[1]
  elif o[0] == '-F': fcompiler = o[1]
  elif o[0] == '-d': dependencies.append(o[1])
  elif o[0] == '-D': defines.append(o[1])
  elif o[0] == '-L': libdirs.append(o[1])
  elif o[0] == '-l': libs.append(o[1])
  elif o[0] == '-i': interfacefile = o[1]
  elif o[0] == '-f': fortranfile = o[1]
  elif o[0] == '--f90': f90 = '--f90'
  elif o[0] == '--f77': f90 = ''
  elif o[0] == '--f90f': f90f = 1
  elif o[0] == '--2underscores': twounderscores = 1
  elif o[0] == '--FOPTS': fopts = o[1]
  elif o[0] == '--COPTS': copts = o[1]
  elif o[0] == '--static': static = 1
  elif o[0] == '--macros': othermacros.append(o[1])

if machine is None:
  machine = sys.platform

paths = string.split(os.environ['PATH'],os.pathsep)
def findfile(file,paths):
  if machine == 'win32': file = file + '.exe'
  for path in paths:
    try:
      if file in os.listdir(path): return path
    except:
      pass
  return None

def fixpath(path):
  if machine == 'win32':
    # --- Cygwin does path mangling, requiring two back slashes
    p = re.sub(r'\\',r'/',path)
    if p[1:2] == ':': p = '/c'+p[2:]
    return p
  else:
    return path

# --- Find place where packages are placed. Use the facility
# --- from distutils to be robust.
pywrapperhome = os.path.join(distutils.sysconfig.get_python_lib(),'Forthon')
pywrapperhome = fixpath(pywrapperhome)
#pywrapperhome = os.path.join(sys.prefix,'lib','python'+sys.version[:3],'site-packages','Forthon')


# --- f90free is set here so that a check can be made later to determine if
# --- it was reset. If it wasn't, that means that a fortran compiler was not
# --- found.
f90free = None
pywrapperargs = ''

# --- Pick the fortran compiler
#-----------------------------------------------------------------------------
if machine == 'linux2':
  if findfile('ifort',paths) and (fcompiler=='intel8' or fcompiler is None):
    # --- Intel
    f90free  = 'ifort -nofor_main -free -r8 -DIFC -fpp -implicitnone -C90 -Zp8'
    f90fixed = 'ifort -nofor_main -132 -r8 -DIFC -fpp -implicitnone -C90 -Zp8'
    popt = '-O'
    flibroot,b = os.path.split(findfile('ifort',paths))
    libdirs.append(flibroot+'/lib')
    libs = libs + ['ifcore','ifport','imf','svml','cxa','irc','unwind']
    if not fopts:
      cpuinfo = open('/proc/cpuinfo','r').read()
      if re.search('Pentium III',cpuinfo):
        fopts = '-O3 -xK -tpp6 -ip -unroll -prefetch'
      else:
        fopts = '-O3 -xW -tpp7 -ip -unroll -prefetch'
  elif findfile('ifc',paths) and (fcompiler=='intel' or fcompiler is None):
    # --- Intel
    f90free  = 'ifc -132 -r8 -DIFC -fpp -implicitnone -C90 -Zp8'
    f90fixed = 'ifc -132 -r8 -DIFC -fpp -implicitnone -C90 -Zp8'
    popt = '-O'
    flibroot,b = os.path.split(findfile('ifc',paths))
    libdirs.append(flibroot+'/lib')
    libs = libs + ['IEPCF90','CEPCF90','F90','intrins','imf','svml','irc','cxa']
    if not fopts:
      cpuinfo = open('/proc/cpuinfo','r').read()
      if re.search('Pentium III',cpuinfo):
        fopts = '-O3 -xK -tpp6 -ip -unroll -prefetch'
      else:
        fopts = '-O3 -xW -tpp7 -ip -unroll -prefetch'
  elif findfile('pgf90',paths) and (fcompiler=='pg' or fcompiler is None):
    # --- Portland group
    f90free  = 'pgf90 -Mextend -Mdclchk -r8'
    f90fixed = 'pgf90 -Mextend -Mdclchk -r8'
    popt = '-Mcache_align'
    flibroot,b = os.path.split(findfile('pgf90',paths))
    libdirs.append(flibroot+'/lib')
    libs = libs + ['pgf90'] # ???
    if not fopts: fopts = '-fast -Mcache_align'
  elif findfile('f90',paths) and (fcompiler=='absoft' or fcompiler is None):
    # --- Absoft
    f90free  = 'f90 -B108 -N113 -W132 -YCFRL=1 -YEXT_NAMES=ASIS'
    f90fixed = 'f90 -B108 -N113 -W132 -YCFRL=1 -YEXT_NAMES=ASIS'
    popt = ''
    pywrapperargs = '--2underscores'
    flibroot,b = os.path.split(findfile('f90',paths))
    libdirs.append(flibroot+'/lib')
    libs = libs + ['U77','V77','f77math','f90math','fio']
    if not fopts: fopts = '-O'

#-----------------------------------------------------------------------------
elif machine == 'darwin':
  # --- MAC OSX
  if findfile('f90',paths) and (fcompiler=='absoft' or fcompiler is None):
    # --- Absoft
    f90free  = 'f90 -N11 -N113 -YEXT_NAMES=LCS -YEXT_SFX=_'
    f90fixed = 'f90 -f fixed -W 132 -N11 -N113 -YEXT_NAMES=LCS -YEXT_SFX=_'
    popt = ''
    pywrapperargs = ''
    flibroot,b = os.path.split(findfile('pgf90',paths))
    libdirs.append(flibroot+'/lib')
    libs = libs + ['fio','f77math','f90math','f90math_altivec']
    if not fopts: fopts = '-O2'

  elif findfile('f95',paths) and (fcompiler=='nag' or fcompiler is None):
    # --- NAG
    f90free  = 'f95 -132 -fpp -Wp,-macro=no_com -free -PIC -w -mismatch_all -kind=byte -r8'
    f90fixed = 'f95 -132 -fpp -Wp,-macro=no_com -Wp,-fixed -fixed -PIC -w -mismatch_all -kind=byte -r8'
    popt = ''
    flibroot,b = os.path.split(findfile('f95',paths))
    libdirs.append(flibroot+'/lib')
    pywrapperargs = ''
    libdirs.append('???')
    libs = libs + ['???']
    if not fopts: fopts = '-Wc,-O3 -Wc,-funroll-loops -O3 -Ounroll=2'

#-----------------------------------------------------------------------------
elif machine == 'win32':
  if findfile('pgf90',paths) and (fcompiler=='pg' or fcompiler is None):
    # --- Portland group
    f90free  = 'pgf90 -Mextend -Mdclchk -r8'
    f90fixed = 'pgf90 -Mextend -Mdclchk -r8'
    popt = '-Mcache_align'
    flibroot,b = os.path.split(findfile('pgf90',paths))
    libdirs.append(flibroot+'/Lib')
    libs = libs + ['???']
    if not fopts: fopts = '-fast -Mcache_align'
  elif findfile('ifl',paths) and (fcompiler=='intel' or fcompiler is None):
    # --- Intel
    f90free  = 'ifl -Qextend_source -Qautodouble -DIFC -FR -Qfpp -4Yd -C90 -Zp8 -Qlowercase -us -MT -Zl -static'
    f90fixed = 'ifl -Qextend_source -Qautodouble -DIFC -FI -Qfpp -4Yd -C90 -Zp8 -Qlowercase -us -MT -Zl -static'
    popt = ''
    flibroot,b = os.path.split(findfile('ifl',paths))
    libdirs.append(flibroot+'/Lib')
    libs = libs + ['CEPCF90MD','F90MD','intrinsMD']
    if not fopts: fopts = '-O3'

#-----------------------------------------------------------------------------
elif machine == 'aix4':
  static = 1
  if fcompiler=='mpxlf' or (fcompiler is None and findfile('mpxlf95',paths)):
    # --- IBM SP, parallel
    f90free  = 'mpxlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -WF,-DMPIPARALLEL -qsuffix=f=f90:cpp=F90 -qfree=f90 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
    f90fixed = 'mpxlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -WF,-DMPIPARALLEL -qfixed=132 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
    popt = '-O'
    ld = 'mpxlf_r -bmaxdata:0x70000000 -bmaxstack:0x10000000 -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp'
    libs = libs + ' $(PYMPI)/driver.o $(PYMPI)/patchedmain.o -L$(PYMPI) -lpympi -lpthread'
    if len(defines) == 0:
      defines.append('PYMPI=/usr/common/homes/g/grote/pyMPI')
    if not fopts:
      fopts = '-O3 -qstrict -qarch=pwr3 -qtune=pwr3'

  elif fcompiler=='xlf' or (fcompiler is None and findfile('xlf95',paths)):
    # --- IBM SP, serial
    f90free  = 'xlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -qsuffix=f=f90:cpp=F90 -qfree=f90 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
    f90fixed = 'xlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -qfixed=132 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
    popt = '-O'
    ld = 'xlf -bmaxdata:0x70000000 -bmaxstack:0x10000000 -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp'
    libs = libs + ' -lpthread'
    if not fopts:
      fopts = '-O3 -qstrict -qarch=pwr3 -qtune=pwr3'

  elif fcompiler=='xlf_r' or (fcompiler is None and findfile('xlf90_r',paths)):
    # --- IBM SP, OpenMP
    f90free  = 'xlf90_r -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -qsuffix=f=f90:cpp=F90 -qfree=f90 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
    f90fixed = 'xlf95 -c -qmaxmem=8192 -u -qdpc=e -qintsize=4 -qsave=defaultinit -qfixed=132 -bmaxdata:0x70000000 -bmaxstack:0x10000000 -WF,-DESSL'
    popt = '-O'
    ld = 'xlf90_r -bmaxdata:0x70000000 -bmaxstack:0x10000000 -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp'
    libs = libs + ' -lpthread -lxlf90_r -lxlopt -lxlf -lxlsmp'
    if not fopts:
      fopts = '-O3 -qstrict -qarch=pwr3 -qtune=pwr3 -qsmp=omp'

else:
  raise SystemExit,'Machine type %s is unknown'%machine


if f90free is None:
  raise SystemExit,'Fortran compiler not found'

# --- Force debugging option if specified
if debug: fopts = '-g'

#-------------------------------------------------------------------------
prefix = fixpath(sys.prefix)
pyvers = sys.version[:3]
python = fixpath(sys.executable)

# --- Generate list of package dependencies
dep = ''
for d in dependencies:
  dep = dep + ' -d %s.scalars'%d

# --- Loop over extrafiles. For each fortran file, append the object name
# --- to be used in the makefile. For each C file, add to a list to be
# --- included in the setup command.
extraobjects = ''
extraobjectslist = []
extracfiles = []
for f in extrafiles:
  root,suffix = os.path.splitext(f)
  if suffix in ['.F','.F90','.f']:
    extraobjects = extraobjects + root + '.o '
    extraobjectslist = extraobjectslist + [root + '.o']
  elif suffix in ['.c']:
    extracfiles.append(f)

# --- Make string containing other macros files
othermacstr = ''
for f in othermacros:
  othermacstr = othermacstr + ' --macros ' + f

# --- Define default rule. Note that static doesn't work yet.
if static:
  default = 'static:'
else:
  fortranroot,suffix = os.path.splitext(fortranfile)
  default = 'dynamic: %(pkg)s_p.o %(fortranroot)s.o %(pkg)spymodule.c Forthon.h Forthon.c %(extraobjects)s'%locals()

definesstr = ''
for d in defines: definesstr = definesstr + d + '\n'

# --- First, create Makefile.pkg which has all the needed definitions
makefiletext = """
%(definesstr)s
PYTHON = %(prefix)s
PYVERS = %(pyvers)s
PYPREPROC = %(python)s -c "from Forthon.preprocess import main;main()" %(f90)s -t%(machine)s %(pywrapperargs)s

%(default)s

%%.o: %%.F %(pkg)s_p.o
	%(f90fixed)s %(fopts)s -c $<
%%.o: %%.F90 %(pkg)s_p.o
	%(f90free)s %(fopts)s -c $<
Forthon.h:%(pywrapperhome)s/Forthon.h
	$(PYPREPROC) %(pywrapperhome)s/Forthon.h Forthon.h
Forthon.c:%(pywrapperhome)s/Forthon.c
	$(PYPREPROC) %(pywrapperhome)s/Forthon.c Forthon.c

%(pkg)s_p.o:%(pkg)s_p.F90
	%(f90free)s %(popt)s -c %(pkg)s_p.F90
%(pkg)spymodule.c %(pkg)s_p.F90:%(interfacefile)s
	%(python)s -c "from Forthon.wrappergenerator import wrappergenerator_main;wrappergenerator_main()" \\
	%(f90)s -t %(machine)s %(pywrapperargs)s %(initialgallot)s \\
        %(othermacstr)s %(interfacefile)s %(dep)s
clean:
	rm -rf *.o *_p.F90 *.mod *module.c *.scalars *.so Forthon.c Forthon.h forthonf2c.h build
"""%(locals())
makefile = open('Makefile.%s'%pkg,'w')
makefile.write(makefiletext)
makefile.close()

# --- Now, execuate the make command.
os.system('make -f Makefile.%s'%pkg)

# --- Make sure that the shared object is deleted. This is needed since
# --- distutils doesn't seem to check if objects passed in are newer
# --- than the shared object. The 'try' is used in case the file doesn't
# --- exist (like when the code is built the first time).
try:
  os.remove(pkg+'py.so')
except:
  pass

sys.argv = ['Forthon','build','--build-platlib','.']
setup(name = pkg,
      ext_modules = [Extension(pkg+'py',
                        [pkg+'pymodule.c','Forthon.c']+extracfiles,
                        include_dirs=[pywrapperhome,'.'],
                        extra_objects=[pkg+'.o',pkg+'_p.o']+extraobjectslist,
                        library_dirs=libdirs,
                        libraries=libs)]
     )

