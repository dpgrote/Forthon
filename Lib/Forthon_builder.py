"""Process and build a Forthon package"""

import sys,os,re
import distutils
import distutils.sysconfig
from distutils.core import setup, Extension
from distutils.dist import Distribution
from distutils.command.build import build

from Forthon_options import options,args
from Forthon.compilers import FCompiler

# --- Get the package name, which is assumed to be the first argument.
pkg = args[0]
del args[0]

print "Building package " + pkg

# --- Get any extra fortran, C or object files listed.
# --- This scans through args until the end or until it finds an option
# --- argument (that begins with a '-'). Any remaining option arguments
# --- are passed to distutils.
extrafiles = []
while len(args) > 0:
    if args[0][0] != '-':
        extrafiles += [args[0]]
        del args[0]
    else:
        break

# --- Default values for command line options
machine        = options.machine
interfacefile  = options.interfacefile or (pkg + '.v')
fortranfile    = options.fortranfile
initialgallot  = options.initialgallot
dependencies   = options.dependencies
defines        = options.defines
fcomp          = options.fcomp
fcompexec      = options.fcompexec
f90            = options.f90
writemodules   = options.writemodules
timeroutines   = options.timeroutines
othermacros    = options.othermacros
debug          = options.debug
underscoring   = options.underscoring
twounderscores = options.twounderscores
fopt           = options.fopt
fargslist      = options.fargslist
cargs          = options.cargs
realsize       = options.realsize
libs           = options.libs
libdirs        = options.libdirs
includedirs    = options.includedirs
static         = options.static
free_suffix    = options.free_suffix
fixed_suffix   = options.fixed_suffix
compile_first  = options.compile_first
builddir       = options.builddir
implicitnone   = options.implicitnone
build_base     = options.build_base
build_temp     = options.build_temp
verbose        = options.verbose
dobuild        = options.dobuild
with_feenableexcept = options.with_feenableexcept
pkgbase        = options.pkgbase
pkgdir         = options.pkgdir

# --- There options require special handling

if initialgallot:
    initialgallot = '-a'
else:
    initialgallot = ''

if f90:
    f90 = '--f90'
else:
    f90 = '--f77'

fargs = ' '.join(fargslist)

if not fortranfile:
    # --- Find the main fortran file, which should have a name like pkg.suffix
    # --- where suffix is one of the free or fixed suffices.
    if os.access(pkg + '.' + options.fixed_suffix,os.F_OK):
        fortranfile = pkg + '.' + options.fixed_suffix
    elif os.access(pkg + '.' + options.free_suffix,os.F_OK):
        fortranfile = pkg + '.' + options.free_suffix
    else:
        raise Exception('Main fortran file can not be found, please specify using the fortranfile option')

# --- Set arguments to Forthon, based on defaults and any inputs.
forthonargs = []
if underscoring: forthonargs.append('--underscoring')
else:            forthonargs.append('--nounderscoring')
if twounderscores: forthonargs.append('--2underscores')
else:              forthonargs.append('--no2underscores')
if not writemodules: forthonargs.append('--nowritemodules')
if timeroutines: forthonargs.append('--timeroutines')

# --- Get the numpy headers path
import numpy
if numpy.__version__ < '1.1.0':
    # --- Older versions of numpy hacked into distutils, changing things
    # --- in such a way to mangle things like object file names. To avoid
    # --- this, the code from get_include is included here explicitly,
    # --- avoiding the importing of numpy.distutils.
    import os
    if numpy.show_config is None:
        # running from numpy source directory
        d = os.path.join(os.path.dirname(numpy.__file__), 'core', 'include')
    else:
        # using installed numpy core headers
        import numpy.core as core
        d = os.path.join(os.path.dirname(core.__file__), 'include')
    includedirs.append(d)
else:
    includedirs.append(numpy.get_include())

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

# --- Find place where packages are placed. This imports one of the
# --- Forthon files and gets the path from that. It uses fvars.py since
# --- that is a small file which doesn't have other dependencies.
import fvars
forthonhome = os.path.dirname(fvars.__file__)
forthonhome = fixpath(forthonhome)
del fvars

# --- Reset sys.argv removing the Forthon options and appending any extra
# --- distutils options remaining in args.
# --- This needs to be done for generating builddir since the distutils
# --- options may affect its value.
sys.argv = ['Forthon','build']
if build_base:
    sys.argv += ['--build-base',build_base]
if not dobuild:
    sys.argv += ['install']
sys.argv += args

# --- Find the location of the build directory. There must be a better way
# --- of doing this.
if builddir is None:
    dummydist = Distribution()
    dummydist.parse_command_line()
    dummybuild = dummydist.get_command_obj('build')
    dummybuild.finalize_options()
    builddir = dummybuild.build_temp
    bb = builddir.split(os.sep)
    upbuilddir = len(bb)*(os.pardir + os.sep)
    del dummydist,dummybuild,bb
else:
    upbuilddir = os.getcwd()

if dobuild:
    # --- Add the build-temp option. This is needed since distutils would otherwise
    # --- put the object files from compiling the pkgnamepy.c file in a temp
    # --- directory relative to the file. build_temp defaults to an empty string,
    # --- so the .o files are put in the same place as the .c files.
    sys.argv += ['--build-temp',build_temp]

# --- Add prefix to interfacefile since it will only be referenced from
# --- the build directory.
interfacefile = fixpath(os.path.join(upbuilddir,interfacefile))

# --- Get path to fortranfile relative to the build directory.
upfortranfile = os.path.join(upbuilddir,fortranfile)

# --- Pick the fortran compiler
fcompiler = FCompiler(machine=machine,
                      debug=debug,
                      fcompname=fcomp,
                      fcompexec=fcompexec,
                      static=static,
                      implicitnone=implicitnone,
                      twounderscores=twounderscores)

# --- Create some locals which are needed for strings below.
f90free = fcompiler.f90free
f90fixed = fcompiler.f90fixed
popt = fcompiler.popt
forthonargs = forthonargs + fcompiler.forthonargs
if fopt is None: fopt = fcompiler.fopt
extra_link_args = fcompiler.extra_link_args
extra_compile_args = fcompiler.extra_compile_args
define_macros = fcompiler.define_macros

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

sourcedirs = []
def getpathbasename(f):
    dirname,basename = os.path.split(f)
    rootname,suffix = os.path.splitext(basename)
    if dirname not in sourcedirs:
        sourcedirs.append(dirname)
    return rootname,suffix

# --- Loop over extrafiles. For each fortran file, append the object name
# --- to be used in the makefile and for setup. For each C file, add to a
# --- list to be included in setup. For each object file, add to the list
# --- of extra objects passed to setup. Make a string of the extra files,
# --- with paths relative to the build directory.
# --- Also, for fortran files, keep a list of suffices so that the appropriate
# --- build rules can be added to the makefile.
extrafilesstr = ''
extraobjectsstr = ''
extraobjectslist = []
extracfiles = []
fortransuffices = [fixed_suffix,free_suffix]
if machine == 'win32':
    osuffix = '.obj'
else:
    osuffix = '.o'
for f in extrafiles:
    extrafilesstr = extrafilesstr + ' ' + os.path.join(upbuilddir,f)
    root,suffix = getpathbasename(f)
    if suffix[1:] in ['o','obj']:
        extraobjectsstr = extraobjectsstr + ' ' + root + osuffix
        extraobjectslist = extraobjectslist + [root + osuffix]
    elif suffix[1:] in ['F','F90','f','f90','for',fixed_suffix,free_suffix]:
        extraobjectsstr = extraobjectsstr + ' ' + root + osuffix
        extraobjectslist = extraobjectslist + [root + osuffix]
        if suffix[1:] not in fortransuffices:
            fortransuffices.append(suffix[1:])
    elif suffix[1:] in ['c', 'cc', 'cpp', 'cxx']:
        extracfiles.append(f)

if compile_first != '':
    compile_firstroot,compile_firstsuffix = getpathbasename(compile_first)
    compile_firstobject = compile_firstroot + osuffix
else:
    compile_firstobject = ''

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
fortranroot,fortransuffix = getpathbasename(fortranfile)
if fcompiler.static:
    defaultrule = 'static:'
    raise InputError('Static linking not supported at this time')
else:
    defaultrule = 'dynamic: %(compile_firstobject)s %(pkg)s_p%(osuffix)s %(fortranroot)s%(osuffix)s %(pkg)spymodule.c Forthon.h Forthon.c %(extraobjectsstr)s'%locals()

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
forthonargs = ' '.join(forthonargs)

# --- Add any includedirs to fargs
for i in includedirs:
    fargs = fargs + ' -I'+i+' '

# --- Add in any user supplied cargs
if cargs is not None:
    extra_compile_args.append(cargs)

pypreproc = '%(python)s -c "from Forthon.preprocess import main;main()" %(f90)s -t%(machine)s %(forthonargs)s'%locals()
forthon = '%(python)s -c "from Forthon.wrappergenerator import wrappergenerator_main;wrappergenerator_main()"'%locals()
noprintdirectory = ''
if not verbose:
    # --- Set so that the make doesn't produce any output
    f90fixed = "@echo ' ' F90Fixed $(<F);" + f90fixed
    f90free = "@echo ' ' F90Free $(<F);" + f90free
    pypreproc = "@echo ' ' Preprocess $(<F);" + pypreproc
    forthon = "@echo ' ' Forthon $(<F);" + forthon
    noprintdirectory = '--no-print-directory'

# --- Create a separate rule to compile the compiler_first file, setting it up
# --- so that it doesn't have any dependencies beyond itself.
compile_firstrule = ''
if compile_first != '' and compile_firstsuffix != '':
    suffixpath = os.path.join(upbuilddir,'%(compile_first)s'%locals())
    if compile_firstsuffix[-2:] == '90': ff = f90free
    else:                                ff = f90fixed
    compile_firstrule = """
%(compile_firstobject)s: %(suffixpath)s
	%(ff)s %(fopt)s %(fargs)s -c $<
  """%locals()

# --- Add build rules for fortran files with suffices other than the
# --- basic fixed and free ones. Those first two suffices are included
# --- explicitly in the makefile. Note that this depends on fargs.
extrafortranrules = ''
if len(fortransuffices) > 2:
    for suffix in fortransuffices[2:]:
        suffixpath = os.path.join(upbuilddir,'%%.%(suffix)s'%locals())
        if suffix[-2:] == '90': ff = f90free
        else:                   ff = f90fixed
        extrafortranrules += """
%%%(osuffix)s: %(suffixpath)s %(modulecontainer)s%(osuffix)s
	%(ff)s %(fopt)s %(fargs)s -c $<
    """%locals()
        del suffix,suffixpath,ff

compilerulestemplate = """
%%%(osuffix)s: %(fixedpath)s %(modulecontainer)s%(osuffix)s
	%(f90fixed)s %(fopt)s %(fargs)s -c $<
%%%(osuffix)s: %(freepath)s %(modulecontainer)s%(osuffix)s
	%(f90free)s %(fopt)s %(fargs)s -c $<
"""

if not writemodules and not fortranfile == compile_first:
    # --- If not writing modules, create a rule to compile the main fortran file
    # --- that does not have a dependency on itself. That self dependency causes
    # --- problems when the fortran file is in a subdirectory.
    # --- Skip this if the fortranfile is the same as compile_first to avoid the
    # --- redundant dependency in the makefile.
    if fortransuffix[-2:] == '90': ff = f90free
    else:                          ff = f90fixed
    compilerules = """
%(wrapperdependency)s: %(upfortranfile)s
	%(ff)s %(fopt)s %(fargs)s -c $<
"""%locals()
else:
    compilerules = ''

for sourcedir in sourcedirs:
    # --- Create path to fortran files for the Makefile since they will be
    # --- referenced from the build directory.
    freepath = os.path.join(os.path.join(upbuilddir,sourcedir),'%%.%(free_suffix)s'%locals())
    fixedpath = os.path.join(os.path.join(upbuilddir,sourcedir),'%%.%(fixed_suffix)s'%locals())
    compilerules += compilerulestemplate%locals()

# --- First, create Makefile.pkg which has all the needed definitions
# --- Note the two rules for the pymodule file. The first specifies that the
# --- pymodule.c file should be recreated if the .v file was updatd.
# --- The second changes the timestamp of the pymodule.c to force a rebuild
# --- of the .so during the distutils setup if any of the source files have
# --- been updated.
makefiletext = """
%(definesstr)s

%(defaultrule)s

%(compile_firstrule)s
%(compilerules)s
%(extrafortranrules)s
Forthon.h:%(forthonhome)s%(pathsep)sForthon.h
	%(pypreproc)s %(forthonhome)s%(pathsep)sForthon.h Forthon.h
Forthon.c:%(forthonhome)s%(pathsep)sForthon.c
	%(pypreproc)s %(forthonhome)s%(pathsep)sForthon.c Forthon.c

%(pkg)s_p%(osuffix)s:%(pkg)s_p.%(free_suffix)s %(wrapperdependency)s
	%(f90free)s %(popt)s %(fargs)s -c %(pkg)s_p.%(free_suffix)s
%(pkg)spymodule.c %(pkg)s_p.%(free_suffix)s::%(interfacefile)s
	%(forthon)s --realsize %(realsize)s %(f90)s -t %(machine)s %(forthonargs)s %(initialgallot)s %(othermacstr)s %(dep)s %(pkg)s %(interfacefile)s
%(pkg)spymodule.c:: %(upfortranfile)s %(extrafilesstr)s
	@touch %(pkg)spymodule.c
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
m = os.system('make -f Makefile.%(pkg)s %(noprintdirectory)s'%locals())
if m != 0:
    # --- If there was a problem with the make, then quite this too.
    # --- The factor of 256 just selects out the higher of the two bytes
    # --- returned by system. The upper has the error number returned by make.
    sys.exit(int(m/256))
os.chdir(upbuilddir)

# --- Make sure that the shared object is deleted. This is needed since
# --- distutils doesn't seem to check if objects passed in are newer
# --- than the shared object. The 'try' is used in case the file doesn't
# --- exist (like when the code is built the first time).
try:
    os.remove(pkg+'py.so')
except:
    pass

cfiles = [os.path.join(builddir,p) for p in [pkg+'pymodule.c','Forthon.c']]
ofiles = [os.path.join(builddir,p) for p in [fortranroot+osuffix,
                                             pkg+'_p'+osuffix] +
                                             extraobjectslist]

# --- DOS requires an extra argument and include directory to build properly
if machine == 'win32': sys.argv.append('--compiler=mingw32')
if machine == 'win32': includedirs+=['/usr/include']

# --- On darwin machines, the python makefile mucks up the -arch argument.
# --- This fixes it.
if machine == 'darwin':
# --- Machines running csh/tcsh seem to have MACHTYPE defined and this is the safest way to set -arch.
    if 'MACHTYPE' in os.environ:
        if os.environ['MACHTYPE'] == 'i386':
            os.environ['ARCHFLAGS'] = '-arch i386'
        elif os.environ['MACHTYPE'] == 'x86_64':
            os.environ['ARCHFLAGS'] = '-arch x86_64'
        elif os.environ['MACHTYPE'] == 'powerpc':
            os.environ['ARCHFLAGS'] = '-arch ppc'
#---  If the shell is bash, MACHTYPE is undefined.  So get what we can from uname. We will assume that if
#---  we are running Snow Leopard we are -arch x86-64 and if running Leopard on intel we are -arch i386.
#---  This can be over-ridden by defining MACHTYPE.
    else:
        archtype = os.uname()[-1]
        if archtype in ['Power Macintosh','ppc']:
            os.environ['ARCHFLAGS'] = '-arch ppc'
        elif archtype in ['i386','x86_64']:
            kernel_major = eval(os.uname()[2].split('.')[0])
            if kernel_major < 10 :
                os.environ['ARCHFLAGS'] = '-arch i386'  # Leopard or earlier
            else:
                os.environ['ARCHFLAGS'] = '-arch x86_64'  # Snow Leopard

if not verbose:
    print "  Setup " + pkg
    sys.stdout = open(os.devnull, 'w')

if with_feenableexcept:
    define_macros.append(('WITH_FEENABLEEXCEPT',1))

if pkgbase is None:
    pkgbase = pkg

define_macros.append(('FORTHON_PKGNAME','"%s"'%pkgbase))

package_dir = None
packages = None
if not dobuild:
    # --- When installing, add the package directory if specified so that the
    # --- other files are installed.
    if pkgdir is not None:
        package_dir = {pkgbase:pkgdir}
        packages = [pkgbase]

setup(name = pkgbase,
      packages = packages,
      package_dir = package_dir,
      ext_modules = [Extension('.'.join([pkgbase,pkg+'py']),
                               cfiles+extracfiles,
                               include_dirs=[forthonhome]+includedirs,
                               extra_objects=ofiles,
                               library_dirs=fcompiler.libdirs+libdirs,
                               libraries=fcompiler.libs+libs,
                               define_macros=define_macros,
                               extra_compile_args=extra_compile_args,
                               extra_link_args=extra_link_args)]
     )

