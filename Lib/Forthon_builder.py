"""
Forthon [options] pkgname [extra Fortran or C files to be compiled or objects to link]

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
    Additional library paths
 -l library
    Additional libraries that are needed. Note that the prefix 'lib' and any
    suffixes should not be included.
 -I path
    Additional include paths
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
 --timeroutines
    When specified, timers are added which accumulate the runtime in all
    routines called directly from python.
 --macros pkg.v
    Other interface files whose macros are needed
 --fopt options
    Optimization option for the fortran compiler. This will replace the
    default optimization options. If there are any spaces in options, it
    must be surrounded in double quotes.
 --fargs options
    Additional options for the fortran compiler. For example to turn on
    profiling.  If there are any spaces in options, it must be surrounded in
    double quotes.
 --static
    Build the static version of the code by default, rather than the
    dynamically linker version. Not yet supported.
 --free_suffix suffix
    Suffix to use for fortran files in free format. Defaults to F90
 --fixed_suffix suffix
    Suffix to use for fortran files in fixed format. Defaults to F
 --compile_first file
"""

import sys,os,re
import getopt
import string
import distutils
import distutils.sysconfig
from distutils.core import setup, Extension
from distutils.dist import Distribution
from distutils.command.build import build
from Forthon.compilers import FCompiler

# --- Print help and then exit if not arguments are given
if len(sys.argv) == 1:
  print __doc__
  sys.exit(0)

# --- Process command line arguments
optlist,args = getopt.getopt(sys.argv[1:],'agd:t:F:D:L:l:I:i:f:',
                         ['f90','f77','f90f','nowritemodules',
                          'timeroutines','macros=',
                          'fopt=','fargs=','static',
                          'free_suffix=','fixed_suffix=',
                          'compile_first='])

# --- Get the package name and any other extra files
pkg = args[0]
extrafiles = args[1:]

# --- Default values for command line options
machine        = sys.platform
interfacefile  = pkg + '.v'
fortranfile    = None # --- Can only be set after finding fixed_suffix
initialgallot  = ''
dependencies   = []
defines        = []
fcomp          = None
f90            = '--f90'
f90f           = 0
writemodules   = 1
timeroutines   = ''
othermacros    = []
debug          = 0
twounderscores = 0
fopt           = None
fargs          = ''
libs           = []
libdirs        = []
includedirs    = []
static         = 0
free_suffix    = 'F90'
fixed_suffix   = 'F'
compile_first  = ''

for o in optlist:
  if o[0]=='-a': initialgallot = '-a'
  elif o[0] == '-g': debug = 1
  elif o[0] == '-t': machine = o[1]
  elif o[0] == '-F': fcomp = o[1]
  elif o[0] == '-d': dependencies.append(o[1])
  elif o[0] == '-D': defines.append(o[1])
  elif o[0] == '-L': libdirs.append(o[1])
  elif o[0] == '-l': libs.append(o[1])
  elif o[0] == '-I': includedirs.append(o[1])
  elif o[0] == '-i': interfacefile = o[1]
  elif o[0] == '-f': fortranfile = o[1]
  elif o[0] == '--f90': f90 = '--f90'
  elif o[0] == '--f77': f90 = ''
  elif o[0] == '--f90f': f90f = 1
  elif o[0] == '--2underscores': twounderscores = 1
  elif o[0] == '--fopt': fopt = o[1]
  elif o[0] == '--fargs': fargs = fargs + ' ' + o[1]
  elif o[0] == '--static': static = 1
  elif o[0] == '--nowritemodules': writemodules = 0
  elif o[0] == '--timeroutines': timeroutines = 1
  elif o[0] == '--macros': othermacros.append(o[1])
  elif o[0] == '--free_suffix': free_suffix = o[1]
  elif o[0] == '--fixed_suffix': fixed_suffix = o[1]
  elif o[0] == '--compile_first': compile_first = o[1]

if fortranfile is None: fortranfile = pkg + '.' + fixed_suffix

# --- Set arguments to Forthon, based on defaults and any inputs.
forthonargs = []
if twounderscores: forthonargs.append('--2underscores')
if not writemodules: forthonargs.append('--nowritemodules')
if timeroutines: forthonargs.append('--timeroutines')

# --- Fix path - needed for Cygwin
def fixpath(path,dos=1):
  if machine == 'win32':
    # --- Cygwin does path mangling, requiring two back slashes
    if(dos):
      p = re.sub(r'\\',r'\\\\',path)
      p = re.sub(r':',r':\\\\',p)
    else:
      p = re.sub(r'\\',r'/',path)
#      p = re.sub(r'C:',r'/c',p)
#      if p[1:2] == ':': p = '/cygdrive/'+string.lower(p[0])+p[2:]
    return p
  else:
    return path

# --- Define the seperator to use in paths.
pathsep = os.sep
if machine == 'win32': pathsep = r'\\'

# --- Find place where packages are placed. Use the facility
# --- from distutils to be robust.
forthonhome = os.path.join(distutils.sysconfig.get_python_lib(),'Forthon')
forthonhome = fixpath(forthonhome)

# --- Find the location of the build directory. There must be a better way
# --- of doing this.
dummydist = Distribution()
dummybuild = build(dummydist)
dummybuild.finalize_options()
builddir = dummybuild.build_temp
bb = string.split(builddir,os.sep)
upbuilddir = len(bb)*(os.pardir + os.sep)
del dummydist,dummybuild,bb

# --- Add prefix to interfacefile since it will only be referenced from
# --- the build directory.
interfacefile = fixpath(os.path.join(upbuilddir,interfacefile))

# --- Pick the fortran compiler
fcompiler = FCompiler(machine=machine,
                      debug=debug,
                      fcompiler=fcomp,
                      static=static)

# --- Create some locals which are needed for strings below.
f90free = fcompiler.f90free
f90fixed = fcompiler.f90fixed
popt = fcompiler.popt
forthonargs = forthonargs + fcompiler.forthonargs
if fopt is None: fopt = fcompiler.fopt
extra_link_args = fcompiler.extra_link_args

# --- Create path to fortran files for the Makefile since they will be
# --- referenced from the build directory.
freepath = os.path.join(upbuilddir,'%%.%(free_suffix)s'%locals())
fixedpath = os.path.join(upbuilddir,'%%.%(fixed_suffix)s'%locals())

# --- Find location of the python libraries and executable.
prefix = fixpath(sys.prefix,dos=0)
pyvers = sys.version[:3]
python = fixpath(sys.executable,dos=0)

# --- Generate list of package dependencies
dep = ''
for d in dependencies:
  dep = dep + ' -d %s.scalars'%d

# --- Loop over extrafiles. For each fortran file, append the object name
# --- to be used in the makefile and for setup. For each C file, add to a
# --- list to be included in setup. For each object file, add to the list
# --- of extra objects passed to setup.
extraobjectsstr = ''
extraobjectslist = []
extracfiles = []
if machine == 'win32':
  osuffix = '.obj'
else:
  osuffix = '.o'
for f in extrafiles:
  root,suffix = os.path.splitext(f)
  if suffix[1:] in ['F','F90','f',fixed_suffix,free_suffix,'o']:
    extraobjectsstr = extraobjectsstr + ' ' + root + osuffix
    extraobjectslist = extraobjectslist + [root + osuffix]
  elif suffix[1:] in ['c']:
    extracfiles.append(f)

if compile_first != '':
  compile_first = compile_first + osuffix

# --- Make string containing other macros files
othermacstr = ''
for f in othermacros:
  othermacstr = othermacstr + ' --macros ' + os.path.join(upbuilddir,f)

# --- Put any defines in a string that will appear at the beginning of the
# --- makefile.
definesstr = ''
for d in (defines + fcompiler.defines):
  definesstr = definesstr + d + '\n'

# --- Define default rule. Note that static doesn't work yet.
fortranroot,suffix = os.path.splitext(fortranfile)
if fcompiler.static:
  defaultrule = 'static:'
  raise "Static linking not supported at this time"
else:
  defaultrule = 'dynamic: %(compile_first)s %(pkg)s_p%(osuffix)s %(fortranroot)s%(osuffix)s %(pkg)spymodule.c Forthon.h Forthon.c %(extraobjectsstr)s'%locals()

if writemodules:
  # --- Fortran modules are written by the wrapper to the _p file.
  # --- The rest of the fortran files will depend in this ones obhect file.
  # --- This file doesn't depend on any fortran files.
  modulecontainer = pkg+'_p'
  wrapperdependency = ''
else:
  # --- If nowritemodules is set, then it is assumed that modules are contained
  # --- in the main fortran file. In this case, set the dependencies in the
  # --- makefile so that all files depend on the main file, rather than the
  # --- wrapper fortran file.
  modulecontainer = fortranroot
  wrapperdependency = fortranroot+osuffix

# --- convert list of fortranargs into a string
forthonargs = string.join(forthonargs,' ')

# --- Add any includedirs to fargs
for i in includedirs:
  fargs = fargs + '-I'+i+' '

# --- First, create Makefile.pkg which has all the needed definitions
makefiletext = """
%(definesstr)s
PYTHON = %(prefix)s
PYVERS = %(pyvers)s
PYPREPROC = %(python)s -c "from Forthon.preprocess import main;main()" %(f90)s -t%(machine)s %(forthonargs)s

%(defaultrule)s

%%%(osuffix)s: %(fixedpath)s %(modulecontainer)s%(osuffix)s
	%(f90fixed)s %(fopt)s %(fargs)s -c $<
%%%(osuffix)s: %(freepath)s %(modulecontainer)s%(osuffix)s
	%(f90free)s %(fopt)s %(fargs)s -c $<
Forthon.h:%(forthonhome)s%(pathsep)sForthon.h
	$(PYPREPROC) %(forthonhome)s%(pathsep)sForthon.h Forthon.h
Forthon.c:%(forthonhome)s%(pathsep)sForthon.c
	$(PYPREPROC) %(forthonhome)s%(pathsep)sForthon.c Forthon.c

%(pkg)s_p%(osuffix)s:%(pkg)s_p.%(free_suffix)s %(wrapperdependency)s
	%(f90free)s %(popt)s %(fargs)s -c %(pkg)s_p.%(free_suffix)s
%(pkg)spymodule.c %(pkg)s_p.%(free_suffix)s:%(interfacefile)s
	%(python)s -c "from Forthon.wrappergenerator import wrappergenerator_main;wrappergenerator_main()" \\
	%(f90)s -t %(machine)s %(forthonargs)s %(initialgallot)s \\
        %(othermacstr)s %(interfacefile)s %(dep)s
clean:
	rm -rf *%(osuffix)s *_p.%(free_suffix)s *.mod *module.c *.scalars *.so Forthon.c Forthon.h forthonf2c.h build
"""%(locals())
builddir=fixpath(builddir,0)
try: os.makedirs(builddir)
except: pass
makefile = open(os.path.join(builddir,'Makefile.%s'%pkg),'w')
makefile.write(makefiletext)
makefile.close()

# --- Now, execuate the make command.
os.chdir(builddir)
os.system('make -f Makefile.%(pkg)s'%locals())
os.chdir(upbuilddir)

# --- Make sure that the shared object is deleted. This is needed since
# --- distutils doesn't seem to check if objects passed in are newer
# --- than the shared object. The 'try' is used in case the file doesn't
# --- exist (like when the code is built the first time).
try:
  os.remove(pkg+'py.so')
except:
  pass

addbuilddir = lambda p:os.path.join(builddir,p)
cfiles = map(addbuilddir,[pkg+'pymodule.c','Forthon.c'])
ofiles = map(addbuilddir,[pkg+osuffix,pkg+'_p'+osuffix]+extraobjectslist)
sys.argv = ['Forthon','build','--build-platlib','.']

# --- DOS requires an extra argument and include directory to build properly
if machine == 'win32': sys.argv.append('--compiler=mingw32')
if machine == 'win32': includedirs+=['/usr/include']

setup(name = pkg,
      ext_modules = [Extension(pkg+'py',
                               cfiles+extracfiles,
                               include_dirs=[forthonhome]+includedirs,
                               extra_objects=ofiles,
                               library_dirs=fcompiler.libdirs+libdirs,
                               libraries=fcompiler.libs+libs,
                               define_macros=fcompiler.define_macros,
                               extra_link_args=extra_link_args)]
     )

