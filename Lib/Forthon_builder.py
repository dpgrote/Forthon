"""
Forthon [options] pkgname [extra files to be compiled]

pkgname is the name of the package.
A complete package will have at least two files, the interface description
file and the fortran file. The default name for the interface file is
pkgname.v.  Note that the first line of the interface file must be the package
name. The default name for the fortran file is pkgname.F

Extra files can for fortran or C files that are to be compiled and included
in the package.

One or more of the following options can be specified.

 -a
    When specified, all groups will be allocated when package is imported
    into python. By default, groups are not allocated.
 -d package
    Specifies that a package that the package being built depends upon.
    This option can be specified multiple times.
 -D VAR=value
    Defines a macro which will be inserted into the makefile. This is required
    in some cases where a third party library must be specified. This can
    be specified multiple times.
 -F command
    Fortran compiler.  Will automatically be determined if not supplied.
    It can be one of the following, depending on the machine:
    intel8, intel, pg, absort, nag, xlf, mpxlf, xlf_r.
 -g
    Turns off optimization for fortran compiler.
 -i filename
    Specify full name of interface file. It defaults to pkgname.v.
 -f filename
    Specifiy full name of main fortran file. It defaults to pkgname.F.
 -L path
    Addition library paths
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
 --FOPTS options
    Additional options to the fortran compiler line. For example to turn on
    profiling. If there are any spaces in options, it must be surrounded in
    double quotes.
 --COPTS options
    Additional options to the C compiler line. For example to turn on
    profiling. If there are any spaces in options, it must be surrounded in
    double quotes.
 --static
    Build the static version of the code by default, rather than the
    dynamically linker version. Not yet supported.
 --free_suffix suffix
    Suffix used for fortran files in free format. Defaults to F90
 --fixed_suffix suffix
    Suffix used for fortran files in fixed format. Defaults to F
"""

import sys,os,re
import getopt
import string
import distutils
import distutils.sysconfig
from distutils.core import setup, Extension
from Forthon.compilers import FCompiler

# --- Print help and then exit if not arguments are given
if len(sys.argv) == 1:
  print __doc__
  sys.exit(0)

# --- Process command line arguments
optlist,args = getopt.getopt(sys.argv[1:],'agd:t:F:D:L:l:i:f:',
                         ['f90','f77','f90f','nowritemodules','macros=',
                          'FOPTS=','COPTS=','static',
                          'free_suffix=','fixed_suffix='])

# --- Get the package name
pkg = args[0]

# --- Get any other extra files
if len(args) > 1:
  extrafiles = args[1:]
else:
  extrafiles = []

# --- Set default values for command line options
machine = sys.platform
interfacefile = pkg + '.v'
fortranfile = pkg + '.F'
initialgallot = ''
dependencies = []
defines = []
fcomp = None
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
free_suffix = 'F90'
fixed_suffix = 'F'

for o in optlist:
  if o[0]=='-a': initialgallot = '-a'
  elif o[0] == '-g': debug = 1
  elif o[0] == '-t': machine = o[1]
  elif o[0] == '-F': fcomp = o[1]
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
  elif o[0] == '--free_suffix': free_suffix = o[1]
  elif o[0] == '--fixed_suffix': fixed_suffix = o[1]


# --- Fix path - needed for Cygwin
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

# --- Pick the fortran compiler
fcompiler = FCompiler(machine=machine,
                      debug=debug,
                      fcompiler=fcomp,
                      static=static)

# --- Create some locals which are needed for strings below.
f90free = fcompiler.f90free
f90fixed = fcompiler.f90fixed
fopts = fopts + ' ' + fcompiler.fopts
popts = fcompiler.popts
pywrapperargs = fcompiler.pywrapperargs

# --- Find location of the python libraries and executable.
prefix = fixpath(sys.prefix)
pyvers = sys.version[:3]
python = fixpath(sys.executable)

# --- Generate list of package dependencies
dep = ''
for d in dependencies:
  dep = dep + ' -d %s.scalars'%d

# --- Loop over extrafiles. For each fortran file, append the object name
# --- to be used in the makefile and for setup. For each C file, add to a
# --- list to be included in setup.
extraobjectsstr = ''
extraobjectslist = []
extracfiles = []
for f in extrafiles:
  root,suffix = os.path.splitext(f)
  if suffix[1:] in ['F','F90','f',fixed_suffix,free_suffix]:
    extraobjectsstr = extraobjectsstr + root + '.o '
    extraobjectslist = extraobjectslist + [root + '.o']
  elif suffix[1:] in ['c']:
    extracfiles.append(f)

# --- Make string containing other macros files
othermacstr = ''
for f in othermacros:
  othermacstr = othermacstr + ' --macros ' + f

# --- Put any defines in a string that will appear at the beginning of the
# --- makefile.
definesstr = ''
for d in (defines + fcompiler.defines):
  definesstr = definesstr + d + '\n'

# --- Define default rule. Note that static doesn't work yet.
if fcompiler.static:
  defaultrule = 'static:'
  raise "Static linking not supported at this time"
else:
  fortranroot,suffix = os.path.splitext(fortranfile)
  defaultrule = 'dynamic: %(pkg)s_p.o %(fortranroot)s.o %(pkg)spymodule.c Forthon.h Forthon.c %(extraobjectsstr)s'%locals()

# --- First, create Makefile.pkg which has all the needed definitions
makefiletext = """
%(definesstr)s
PYTHON = %(prefix)s
PYVERS = %(pyvers)s
PYPREPROC = %(python)s -c "from Forthon.preprocess import main;main()" %(f90)s -t%(machine)s %(pywrapperargs)s

%(defaultrule)s

%%.o: %%.%(fixed_suffix)s %(pkg)s_p.o
	%(f90fixed)s %(fopts)s -c $<
%%.o: %%.%(free_suffix)s %(pkg)s_p.o
	%(f90free)s %(fopts)s -c $<
Forthon.h:%(pywrapperhome)s/Forthon.h
	$(PYPREPROC) %(pywrapperhome)s/Forthon.h Forthon.h
Forthon.c:%(pywrapperhome)s/Forthon.c
	$(PYPREPROC) %(pywrapperhome)s/Forthon.c Forthon.c

%(pkg)s_p.o:%(pkg)s_p.%(free_suffix)s
	%(f90free)s %(popts)s -c %(pkg)s_p.%(free_suffix)s
%(pkg)spymodule.c %(pkg)s_p.%(free_suffix)s:%(interfacefile)s
	%(python)s -c "from Forthon.wrappergenerator import wrappergenerator_main;wrappergenerator_main()" \\
	%(f90)s -t %(machine)s %(pywrapperargs)s %(initialgallot)s \\
        %(othermacstr)s %(interfacefile)s %(dep)s
clean:
	rm -rf *.o *_p.%(free_suffix)s *.mod *module.c *.scalars *.so Forthon.c Forthon.h forthonf2c.h build
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
                        library_dirs=fcompiler.libdirs+libdirs,
                        libraries=fcompiler.libs+libs)]
     )

