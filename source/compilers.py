"""Class which determines which fortran compiler to use and sets defaults
for it.
"""

import sys
import os
import re
import platform
import struct
import subprocess
from cfinterface import realsize, intsize


class FCompiler:
    """
    Determines which compiler to use and sets up description of how to use it.
    To add a new compiler, create a new method with a name using the format
    machine_compiler. The first lines of the function must be of the following form

        if usecompiler(fcompname, fcompexec):
          self.fcompname = fcompname
          self.f90free += ' -fortran arguments for free format'
          self.f90fixed += ' -fortran arguments for fixed format'

    where fcompexec is the executable name of the compiler and fcompname is a
    descriptive (or company) name for the compiler. They can be the same.
    Note that the final linking is done with gcc, so any fortran libraries will
    need to be added to libs (and their locations to libdirs).
    Also, the new function must be included in the while loop below in the
    appropriate block for the machine.
    """

    def __init__(self, machine=None, debug=0, fcompname=None, fcompexec=None, implicitnone=1, twounderscores=0):
        if machine is None:
            machine = sys.platform
        self.machine = machine
        if self.machine != 'win32':
            self.processor = os.uname()[4]
        else:
            self.processor = 'i686'
        self.paths = os.environ['PATH'].split(os.pathsep)

        self.fcompname = fcompname
        self.fcompexec = fcompexec
        self.implicitnone = implicitnone
        self.twounderscores = twounderscores
        self.defines = []
        self.fopt = ''
        self.popt = ''
        self.libs = []
        self.libdirs = []
        self.forthonargs = []
        self.extra_link_args = []
        self.extra_compile_args = []
        self.define_macros = []

        if self.fcompexec in ['mpif90', 'mpifort']:
            self.getmpicompilerinfo()

        # --- Pick the fortran compiler
        # --- When adding a new compiler, it must be listed here under the correct
        # --- machine name.
        while 1:
            if self.machine in ['linux', 'linux2', 'linux3']:
                if self.linux_intel() is not None:
                    break
                if self.linux_g95() is not None:
                    break
                if self.linux_gfortran() is not None:
                    break
                if self.linux_pg() is not None:
                    break
                if self.linux_absoft() is not None:
                    break
                if self.linux_lahey() is not None:
                    break
                if self.linux_pathscale() is not None:
                    break
                if self.linux_xlf_r() is not None:
                    break
                if self.linux_cray() is not None:
                    break
            elif self.machine == 'darwin':
                if self.macosx_gfortran() is not None:
                    break
                if self.macosx_xlf() is not None:
                    break
                if self.macosx_g95() is not None:
                    break
                if self.macosx_absoft() is not None:
                    break
                if self.macosx_nag() is not None:
                    break
                if self.macosx_gnu() is not None:
                    break
            elif self.machine == 'cygwin':
                if self.cygwin_g95() is not None:
                    break
            elif self.machine == 'win32':
                if self.win32_pg() is not None:
                    break
                if self.win32_intel() is not None:
                    break
            elif self.machine == 'aix4' or self.machine == 'aix5':
                if self.aix_xlf() is not None:
                    break
                if self.aix_mpxlf() is not None:
                    break
                if self.aix_xlf_r() is not None:
                    break
                if self.aix_mpxlf64() is not None:
                    break
                if self.aix_pghpf() is not None:
                    break
            else:
                raise SystemExit('Machine type %s is unknown'%self.machine)
            raise SystemExit('Fortran compiler not found')

        # --- The following two quantities must be defined.
        try:
            self.f90free
            self.f90fixed
        except:
            # --- Note that this error should never happed (except during debugging)
            raise ValueError("The fortran compiler definition is not correct, f90free and f90fixed must be defined.")

        if debug:
            self.fopt = '-g'
            self.popt = '-g'
            self.extra_link_args += ['-g']
            self.extra_compile_args += ['-g', '-O0']

        # --- Add the compiler name to the forthon arguments
        self.forthonargs += ['-F ' + self.fcompname]

    def getmpicompilerinfo(self):
        # --- Try using mpifort -show to discover the compiler information
        try:
            show = subprocess.check_output(['mpifort', '-show'], universal_newlines=True, stderr=subprocess.STDOUT)
        except (OSError, subprocess.CalledProcessError):
            pass
        else:
            # --- Get mpi linking arguments
            # --- These are needed since the linking will be done with a c compiler which will not know
            # --- about the arguments needed for Fortran.
            for s in show.split():
                if s.startswith('-L') or s.startswith('-l'):
                    self.extra_link_args += [s]
            fcompname = os.path.basename(show.split()[0])
            if self.fcompname is None:
                self.fcompname = fcompname
            else:
                assert self.fcompname == fcompname, Exception('The compiler specified must be the same as the one used by mpifort')

    def usecompiler(self, fcompname, fcompexec):
        'Check if the specified compiler is found'
        if self.fcompexec is None:
            result = (self.findfile(fcompexec) and
                      (self.fcompname == fcompname or self.fcompname is None))
            if result:
                self.fcompexec = fcompexec
        else:
            result = self.fcompname == fcompname and self.findfile(self.fcompexec)
        if result:
            self.f90free = self.fcompexec
            self.f90fixed = self.fcompexec
        return result

    def findfile(self, file, followlinks=1):
        if self.machine == 'win32':
            file = file + '.exe'
        if os.path.isabs(file):
            if os.access(file, os.X_OK):
                return file
            return None
        for path in self.paths:
            try:
                if file in os.listdir(path) and os.path.isfile(os.path.join(path, file)):
                    # --- Check if the path is a link
                    if followlinks:
                        try:
                            link = os.readlink(os.path.join(path, file))
                            result = os.path.dirname(link)
                            if result == '':
                                # --- link is not a full path but a local link, so the
                                # --- path needs to be prepended.
                                result = os.path.join(os.path.dirname(path), link)
                            path = result
                        except OSError:
                            pass
                    return path
            except:
                pass
        return None

    # ----------------------------------------------------------------------------
    # --- Machine generic utilities

    # --- For g95 and gfortran
    def findgnulibroot(self, fcompname, fcompexec):
        # --- Find the lib root for gnu based compilers.
        # --- Get the full name of the compiler executable.
        fcomp = os.path.join(self.findfile(fcompexec, followlinks=0), fcompexec)
        # --- Map the compiler name to the library needed.
        flib = {'gfortran': 'gfortran', 'g95': 'f95'}[fcompname]
        # --- Run it with the appropriate option to return the library path name
        ff = os.popen(fcomp + ' -print-file-name=lib' + flib + '.a')
        gcclib = ff.readline()[:-1]
        ff.close()
        # --- Strip off the actual library name to get the path.
        libroot = os.path.dirname(gcclib)
        return libroot

    # --- Makes sure libdirs is not [''], because that can change the semantics
    # --- of how gcc linking works. A -L'' argument will cause global libraries
    # --- to not be found.
    def findgnulibdirs(self, fcompname, fcompexec):
        libroot = self.findgnulibroot(fcompname, fcompexec)
        if libroot:
            return [libroot]
        return []

    # --- Make sure tha the version of gfortran is new enough. Older versions
    # --- had a bug - array dimensions were not setup properly for array arguments,
    # --- leading to memory out of bounds errors.
    def isgfortranversionok(self, fcompexec):
        fcomp = os.path.join(self.findfile(fcompexec, followlinks=0), fcompexec)
        ff = os.popen(fcomp + ' -dumpversion')
        dumpversion = ff.readline()[:-1]
        ff.close()
        # --- The format of dumpversion is not consistent - really annoying.
        # --- For version 4.4 and older, there was extra text output.
        # --- Search through it to find a version number.
        for version in dumpversion.split():
            if re.match('[0-9]', version[0]):
                break
        else:
            # --- Version not found, set to zero
            version = '0.0'
        vfloat = float('.'.join(version.split('.')[0:2]))
        if vfloat < 4.3:
            print "gfortran will not be used, it's version is too old or unknown - upgrade to a newer version"
            return False
        else:
            return True

    # -----------------------------------------------------------------------------
    # --- LINUX
    def linux_intel(self):
        if self.usecompiler('intel', 'ifort') or self.usecompiler('intel8', 'ifort'):
            self.fcompname = 'ifort'
            self.f90free += ' -nofor_main -free -DIFC -fpp -fPIC'
            self.f90fixed += ' -nofor_main -132 -DIFC -fpp -fPIC'
            self.f90free += ' -DFPSIZE=%s -r%s -Zp%s'%(realsize, realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s -Zp%s'%(realsize, realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            if self.implicitnone:
                self.f90free += ' -implicitnone'
                self.f90fixed += ' -implicitnone'
            self.popt = '-O'
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/lib']
            self.libs = ['ifcore', 'ifport', 'imf', 'svml', 'irc']
            cpuinfo = open('/proc/cpuinfo', 'r').read()
            if re.search('Pentium III', cpuinfo):
                self.fopt = '-O3 -xK -tpp6 -ip -unroll -prefetch'
            elif re.search('AMD Athlon', cpuinfo):
                self.fopt = '-O3 -ip -unroll -prefetch'
            elif self.processor == 'ia64':
                self.fopt = '-O3 -ip -unroll -tpp2'
                self.f90free += ' -fpic'
                self.f90fixed += ' -fpic'
                self.libs.remove('svml')
            elif struct.calcsize('l') == 8:
                self.fopt = '-O3 -ip -unroll'
            else:
                self.fopt = '-O3 -ip -unroll'
            return 1

    def linux_g95(self):
        if self.usecompiler('g95', 'g95'):
            self.fcompname = 'g95'
            self.f90free += ' -ffree-form -fPIC -Wno=155 -fshort-circuit'
            self.f90fixed += ' -ffixed-line-length-132 -fPIC -fshort-circuit'
            self.f90free += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            if self.implicitnone:
                self.f90free += ' -fimplicit-none'
                self.f90fixed += ' -fimplicit-none'
            if self.twounderscores:
                self.f90free += ' -fsecond-underscore'
                self.f90fixed += ' -fsecond-underscore'
            else:
                self.f90free += ' -fno-second-underscore'
                self.f90fixed += ' -fno-second-underscore'
            self.popt = '-O'
            self.libdirs = self.findgnulibdirs(self.fcompname, self.fcompexec)
            self.libs = ['f95']
            cpuinfo = open('/proc/cpuinfo', 'r').read()
            if re.search('Pentium III', cpuinfo):
                self.fopt = '-O3'
            elif re.search('AMD Athlon', cpuinfo):
                self.fopt = '-O3'
            elif struct.calcsize('l') == 8:
                self.fopt = '-O3 -mfpmath=sse -ftree-vectorize -ftree-vectorizer-verbose=0 -funroll-loops -fstrict-aliasing -fsched-interblock -falign-loops=16 -falign-jumps=16 -falign-functions=16 -ffast-math -fstrict-aliasing'
            else:
                self.fopt = '-O3'
            return 1

    def linux_gfortran(self):
        if self.usecompiler('gfortran', 'gfortran'):
            if not self.isgfortranversionok(self.fcompexec):
                return None
            self.fcompname = 'gfortran'
            self.f90free += ' -fPIC'
            self.f90fixed += ' -fPIC -ffixed-line-length-132'
            self.f90free += ' -DFPSIZE=%s'%(realsize)
            self.f90fixed += ' -DFPSIZE=%s'%(realsize)
            if realsize == '8':
                self.f90free += ' -fdefault-real-8 -fdefault-double-8'
                self.f90fixed += ' -fdefault-real-8 -fdefault-double-8'
            self.f90free += ' -DISZ=%s'%(intsize)
            self.f90fixed += ' -DISZ=%s'%(intsize)
            if intsize == '8':
                self.f90free += ' -fdefault-integer-8'
                self.f90fixed += ' -fdefault-integer-8'
            if self.implicitnone:
                self.f90free += ' -fimplicit-none'
                self.f90fixed += ' -fimplicit-none'
            if self.twounderscores:
                self.f90free += ' -fsecond-underscore'
                self.f90fixed += ' -fsecond-underscore'
            else:
                self.f90free += ' -fno-second-underscore'
                self.f90fixed += ' -fno-second-underscore'
            self.libdirs = self.findgnulibdirs(self.fcompname, self.fcompexec)
            self.libs = ['gfortran']
            self.fopt = '-O3 -ftree-vectorize -ftree-vectorizer-verbose=0'
            return 1

    def linux_pg(self):
        if self.usecompiler('pg', 'pgf90'):
            # --- Portland group
            self.fcompname = 'pgi'
            f90opts = ' -fPIC -Mextend'
            if self.implicitnone:
                f90opts += ' -Mdclchk'
            else:
                f90opts += ' -Mnodclchk'
            if self.twounderscores:
                f90opts += ' -Msecond_underscore'
            else:
                f90opts += ' -Mnosecond_underscore'
            f90opts += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            f90opts += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90free += f90opts
            self.f90fixed += f90opts
            self.popt = '-Mcache_align'
            if platform.python_compiler().startswith('GCC'):
                flibroot, b = os.path.split(self.findfile(self.fcompexec))
                self.libdirs = [flibroot + '/lib']
                self.libs = ['pgf90']  # ???
            else:
                # --- When using pgcc for linking, this includes the needed fortran libraries.
                self.extra_link_args += ['-pgf90libs']
            self.fopt = '-fast -Mcache_align'
            return 1

    def linux_absoft(self):
        if self.usecompiler('absoft', 'f90'):
            self.fcompname = 'absoft'
            # --- Absoft
            self.f90free += ' -B108 -N113 -W132 -YCFRL=1 -YEXT_NAMES=ASIS'
            self.f90fixed += ' -B108 -N113 -W132 -YCFRL=1 -YEXT_NAMES=ASIS'
            self.f90free += ' -DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.popt = '-Mcache_align'
            self.forthonargs = ['--2underscores']  # --- This needs to be fixed XXX
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/lib']
            self.libs = ['U77', 'V77', 'f77math', 'f90math', 'fio']
            self.fopt = '-O'
            return 1

    def linux_lahey(self):
        if self.usecompiler('lahey', 'lf95'):
            self.fcompname = 'lahey'
            # --- Lahey
            # in = implicit none
            # dbl = real*8 (variables or constants?)
            # [n]fix = fixed or free form
            # wide = column width longer than 72
            # ap = preserve arithmetic precision
            self.f90free += ' --nfix --dbl --mlcdecl'
            self.f90fixed += ' --fix --wide --dbl --mlcdecl'
            self.f90free += ' -DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.popt = '-Mcache_align'
            if self.implicitnone:
                self.f90free += ' --in'
                self.f90fixed += ' --in'
            self.popt = '-O'
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/lib']
            self.libs = ["fj9i6", "fj9f6", "fj9e6", "fccx86"]
            cpuinfo = open('/proc/cpuinfo', 'r').read()
            if re.search('Pentium III', cpuinfo):
                self.fopt = '--O2 --unroll --prefetch --nap --npca --ntrace --nsav'
            elif re.search('AMD Athlon', cpuinfo):
                self.fopt = '--O2 --unroll --prefetch --nap --npca --ntrace --nsav'
            else:
                self.fopt = '--O2 --unroll --prefetch --nap --npca --ntrace --nsav'
            return 1

    def linux_pathscale(self):
        if self.usecompiler('pathscale', 'pathf95'):
            self.f90free += ' -freeform -DPATHF90 -ftpp -fPIC -woff1615'
            self.f90fixed += ' -fixedform -extend_source -DPATHF90 -ftpp -fPIC -woff1615'
            self.f90free += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            if self.twounderscores:
                self.f90free += ' -fsecond-underscore'
                self.f90fixed += ' -fsecond-underscore'
            else:
                self.f90free += ' -fno-second-underscore'
                self.f90fixed += ' -fno-second-underscore'
            self.popt = '-O'
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/lib/2.1']
            self.libs = ['pathfortran']
            cpuinfo = open('/proc/cpuinfo', 'r').read()
            self.extra_compile_args = ['-fPIC']
            self.extra_link_args += ['-fPIC']
            if re.search('Pentium III', cpuinfo):
                self.fopt = '-Ofast'
            elif re.search('AMD Athlon', cpuinfo):
                self.fopt = '-O3'
            elif struct.calcsize('l') == 8:
                self.fopt = '-O3 -OPT:Ofast -fno-math-errno'
            else:
                self.fopt = '-Ofast'
            return 1

    def linux_xlf_r(self):
        if self.usecompiler('xlf_r', 'xlf95_r'):
            self.fcompname = 'xlf'
            intsize = struct.calcsize('l')
            f90 = ' -c -WF,-DXLF -qmaxmem=8192 -qdpc=e -qautodbl=dbl4 -qsave=defaultinit -WF,-DESSL'%locals()
            self.f90free += f90 + ' -qsuffix=f=f90:cpp=F90 -qfree=f90'
            self.f90fixed += f90 + ' -qfixed=132'
            self.f90free += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.f90fixed += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.ld = 'xlf95_r -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp'%locals()
            if self.implicitnone:
                self.f90free += ' -u'
                self.f90fixed += ' -u'
            self.popt = '-O'
            #self.extra_link_args = []
            self.extra_compile_args = []
            # --- Note that these are specific the machine intrepid at Argonne.
            self.libdirs = ['/gpfs/software/linux-sles10-ppc64/apps/V1R3M0/ibmcmp-sep2008/opt/xlf/bg/11.1/lib', '/gpfs/software/linux-sles10-ppc64/apps/V1R3M0/ibmcmp-sep2008/opt/xlsmp/bg/1.7/lib']
            self.libs = ['xlf90_r', 'xlsmp']
            self.fopt = '-O3 -qstrict -qarch=auto -qtune=auto -qsmp=omp'
            return 1

    def linux_cray(self):
        if self.usecompiler('crayftn', 'crayftn'):
            self.fcompname = 'crayftn'
            self.f90free += ' -f free'
            self.f90fixed += ' -f fixed -N 132'
            self.f90free += ' -DFPSIZE=%s'%(realsize)
            self.f90fixed += ' -DFPSIZE=%s'%(realsize)
            if realsize == '8':
                self.f90free += ' -s real64'
                self.f90fixed += ' -s real64'
            self.f90free += ' -DISZ=%s'%(intsize)
            self.f90fixed += ' -DISZ=%s'%(intsize)
            if intsize == '8':
                self.f90free += ' -s integer64'
                self.f90fixed += ' -s integer64'
            if self.implicitnone:
                self.f90free += ' -e I'
                self.f90fixed += ' -e I'
            if self.twounderscores:
                self.f90free += ' -h second_underscore'
                self.f90fixed += ' -h second_underscore'
            else:
                self.f90free += ' -h nosecond_underscore'
                self.f90fixed += ' -h nosecond_underscore'
            self.libdirs = []
            self.libs = []
            self.fopt = '-O3'
            return 1

    # -----------------------------------------------------------------------------
    # --- CYGWIN
    def cygwin_g95(self):
        if self.usecompiler('g95', 'g95'):
            self.fcompname = 'g95'
            self.f90fixed += ' -ffixed-line-length-132'
            self.f90free += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            if self.twounderscores:
                self.f90free += ' -fsecond-underscore'
                self.f90fixed += ' -fsecond-underscore'
            else:
                self.f90free += ' -fno-second-underscore'
                self.f90fixed += ' -fno-second-underscore'
            self.fopt = '-O3 -ftree-vectorize -ftree-vectorizer-verbose=0'
#      self.fopt = '-O3 -funroll-loops -fstrict-aliasing -fsched-interblock  \
#           -falign-loops=16 -falign-jumps=16 -falign-functions=16 \
#           -falign-jumps-max-skip=15 -falign-loops-max-skip=15 -malign-natural \
#           -ffast-math -mpowerpc-gpopt -force_cpusubtype_ALL \
#           -fstrict-aliasing'
#      self.extra_link_args += ['-flat_namespace', '-undefined suppress', '-lg2c']
            self.extra_link_args += ['-flat_namespace', '--allow-shlib-undefined', '-Wl,--export-all-symbols', '-Wl,-export-dynamic', '-Wl,--unresolved-symbols=ignore-all', '-lg2c']
            self.libdirs = self.findgnulibdirs(self.fcompname, self.fcompexec)
            self.libdirs.append('/lib/w32api')
            self.libs = ['f95']
            return 1

    # -----------------------------------------------------------------------------
    # --- MAC OSX
    def macosx_g95(self):
        if self.usecompiler('g95', 'g95'):
            self.fcompname = 'g95'
            self.f90free += ' -fzero -ffree-form -Wno=155'
            self.f90fixed += ' -fzero -ffixed-line-length-132'
            self.f90free += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            if self.implicitnone:
                self.f90free += ' -fimplicit-none'
                self.f90fixed += ' -fimplicit-none'
            if self.twounderscores:
                self.f90free += ' -fsecond-underscore'
                self.f90fixed += ' -fsecond-underscore'
            else:
                self.f90free += ' -fno-second-underscore'
                self.f90fixed += ' -fno-second-underscore'
            self.fopt = '-O3 -funroll-loops -fstrict-aliasing -fsched-interblock \
                 -falign-loops=16 -falign-jumps=16 -falign-functions=16 \
                 -ftree-vectorize -ftree-vectorizer-verbose=0 \
                 -ffast-math -fstrict-aliasing'
#      self.fopt = '-O3  -mtune=G5 -mcpu=G5 -mpowerpc64'
            self.extra_link_args += ['-flat_namespace']
            self.libdirs = self.findgnulibdirs(self.fcompname, self.fcompexec)
            self.libs = ['f95']
            return 1

    def macosx_gfortran(self):
        if self.usecompiler('gfortran', 'gfortran'):
            if not self.isgfortranversionok(self.fcompexec):
                return None
            self.fcompname = 'gfortran'
            self.f90fixed += ' -ffixed-line-length-132'
            self.f90free += ' -DFPSIZE=%s'%(realsize)
            self.f90fixed += ' -DFPSIZE=%s'%(realsize)
            if realsize == '8':
                self.f90free += ' -fdefault-real-8 -fdefault-double-8'
                self.f90fixed += ' -fdefault-real-8 -fdefault-double-8'
            self.f90free += ' -DISZ=%s'%(intsize)
            self.f90fixed += ' -DISZ=%s'%(intsize)
            if intsize == '8':
                self.f90free += ' -fdefault-integer-8'
                self.f90fixed += ' -fdefault-integer-8'
            if self.implicitnone:
                self.f90free += ' -fimplicit-none'
                self.f90fixed += ' -fimplicit-none'
            if self.twounderscores:
                self.f90free += ' -fsecond-underscore'
                self.f90fixed += ' -fsecond-underscore'
            else:
                self.f90free += ' -fno-second-underscore'
                self.f90fixed += ' -fno-second-underscore'
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.fopt = '-O3 -funroll-loops -fstrict-aliasing -fsched-interblock  \
                 -falign-loops=16 -falign-jumps=16 -falign-functions=16 \
                 -malign-natural \
                 -ffast-math -mpowerpc-gpopt -force_cpusubtype_ALL \
                 -fstrict-aliasing -mtune=G5 -mcpu=G5 -mpowerpc64'
#      self.fopt = '-O3  -mtune=G5 -mcpu=G5 -mpowerpc64'
            self.fopt = '-O3 -ftree-vectorize -ftree-vectorizer-verbose=0'
#      self.extra_link_args += ['-flat_namespace', '-lg2c']
            self.extra_link_args += ['-flat_namespace']
            self.libdirs = self.findgnulibdirs(self.fcompname, self.fcompexec)
            self.libs = ['gfortran']
            return 1

    def macosx_xlf(self):
        if self.usecompiler('xlf', 'xlf95') or self.usecompiler('xlf90', 'xlf95'):
            self.fcompname = 'xlf'
            self.f90free += ' -WF,-DXLF -qsuffix=f=f90:cpp=F90 -qextname -qautodbl=dbl4 -qdpc=e -bmaxdata:0x70000000 -bmaxstack:0x10000000 -qinitauto'
            self.f90fixed += ' -WF,-DXLF -qextname -qfixed=132 -qautodbl=dbl4 -qdpc=e -bmaxdata:0x70000000 -bmaxstack:0x10000000 -qinitauto'
            self.f90free += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.f90fixed += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            if self.implicitnone:
                self.f90free += ' -u'
                self.f90fixed += ' -u'
            self.fopt = '-O5'
            self.extra_link_args += ['-flat_namespace']  # , '-Wl,-undefined, suppress']  # , '-Wl,-stack_size, 10000000']
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/lib']
            self.libs = ['xlf90', 'xl', 'xlfmath']
            return 1

    def macosx_absoft(self):
        if self.usecompiler('absoft', 'f90'):
            self.fcompname = 'absoft'
            self.f90free += ' -N11 -N113 -YEXT_NAMES=LCS -YEXT_SFX=_'
            self.f90fixed += ' -f fixed -W 132 -N11 -N113 -YEXT_NAMES=LCS -YEXT_SFX=_'
            self.f90free += ' -DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/lib']
            self.extra_link_args += ['-flat_namespace', '-Wl,-undefined, suppress']
            self.libs = ['fio', 'f77math', 'f90math', 'f90math_altivec', 'lapack', 'blas']
            self.fopt = '-O3'
            return 1

    def macosx_nag(self):
        if self.usecompiler('nag', 'f95'):
            self.fcompname = 'nag'
            # self.f90free += ' -132 -fpp -Wp,-macro=no_com -Wc,-O3 -Wc,-funroll-loops -free -PIC -u -w -mismatch_all -kind=byte'
            # self.f90fixed += ' -132 -fpp -u -Wp,-macro=no_com -Wp,-fixed -fixed -Wc,-O3 -Wc,-funroll-loops -PIC -w -mismatch_all -kind=byte'
            self.f90free += ' -132 -fpp -Wp,-macro=no_com -free -PIC -u -w -mismatch_all -kind=byte -Oassumed=contig'
            self.f90fixed += ' -132 -fpp -Wp,-macro=no_com -Wp,-fixed -fixed -PIC -u -w -mismatch_all -kind=byte -Oassumed=contig'
            self.f90free += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.extra_link_args += ['-flat_namespace', '-framework vecLib', '/usr/local/lib/NAGWare/quickfit.o', '/usr/local/lib/NAGWare/libf96.a']
            self.libs = ['m']
            self.fopt = '-Wc,-O3 -Wc,-funroll-loops -O3 -Ounroll=2'
            self.fopt = '-O4 -Wc,-fast'
            self.fopt = '-O3 '  # -Wc,-fast'
            self.define_macros.append(('NAG', '1'))
            return 1

    def macosx_gnu(self):
        if self.usecompiler('gnu', 'g95'):
            self.fcompname = 'gnu'
            self.f90fixed += ' -ffixed-form -ffixed-line-length-132'
            self.f90free += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/lib']
            self.libs = ['???']
            self.fopts = '-O3'
            return 1

    # -----------------------------------------------------------------------------
    # --- WIN32
    def win32_pg(self):
        if self.usecompiler('pg', 'pgf90'):
            # --- Portland group
            self.fcompname = 'pgi'
            self.f90free += ' -Mextend -Mdclchk'
            self.f90fixed += ' -Mextend -Mdclchk'
            self.f90free += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.popt = '-Mcache_align'
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/Lib']
            self.libs = ['???']
            self.fopt = '-fast -Mcache_align'
            return 1

    def win32_intel(self):
        if self.usecompiler('intel', 'ifl'):
            self.fcompname = 'ifl'
            self.f90free += ' -Qextend_source -Qautodouble -DIFC -FR -Qfpp -4Yd -C90 -Zp8 -Qlowercase -us -MT -Zl -static'
            self.f90fixed += ' -Qextend_source -Qautodouble -DIFC -FI -Qfpp -4Yd -C90 -Zp8 -Qlowercase -us -MT -Zl -static'
            self.f90free += ' -DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/Lib']
            self.libs = ['CEPCF90MD', 'F90MD', 'intrinsMD']
            self.fopt = '-O3'
            return 1

    # -----------------------------------------------------------------------------
    # --- AIX
    def aix_xlf(self):
        if self.usecompiler('xlf', 'xlf95'):
            self.fcompname = 'xlf'
            intsize = struct.calcsize('l')
            if intsize == '4':
                bmax = '-bmaxdata:0x70000000 -bmaxstack:0x10000000'
            else:
                bmax = '-q64'
            f90 = ' -c -WF,-DXLF -qmaxmem=8192 -qdpc=e -qautodbl=dbl4 -qsave=defaultinit -WF,-DESSL %(bmax)s'%locals()
            self.f90free += f90 + ' -qsuffix=f=f90:cpp=F90 -qfree=f90'
            self.f90fixed += f90 + ' -qfixed=132'
            self.f90free += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.f90fixed += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.ld = 'xlf -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp %(bmax)s'%locals()
            self.popt = '-O'
            self.extra_link_args += [bmax]
            self.extra_compile_args = [bmax]
            if self.implicitnone:
                self.f90free += ' -u'
                self.f90fixed += ' -u'
            self.libs = ['xlf90', 'xlopt', 'xlf', 'xlomp_ser', 'pthread', 'essl']
            self.fopt = '-O3 -qstrict -qarch=auto -qtune=auto'
            return 1

    def aix_mpxlf(self):
        if self.usecompiler('mpxlf', 'mpxlf95'):
            self.fcompname = 'xlf'
            intsize = struct.calcsize('l')
            if intsize == '4':
                bmax = '-bmaxdata:0x70000000 -bmaxstack:0x10000000'
            else:
                bmax = '-q64'
            f90 = ' -c -WF,-DXLF -qmaxmem=8192 -qdpc=e -qautodbl=dbl4 -qsave=defaultinit -WF,-DMPIPARALLEL -WF,-DESSL %(bmax)s'%locals()
            self.f90free += f90 + ' -qsuffix=f=f90:cpp=F90 -qfree=f90'
            self.f90fixed += f90 + ' -qfixed=132'
            self.f90free += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.f90fixed += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.ld = 'mpxlf_r -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp %(bmax)s'%locals()
            if self.implicitnone:
                self.f90free += ' -u'
                self.f90fixed += ' -u'
            self.popt = '-O'
            self.extra_link_args += [bmax]
            self.extra_compile_args = [bmax]
            self.libs = ['xlf90', 'xlopt', 'xlf', 'xlomp_ser', 'pthread', 'essl']
            self.defines = ['PYMPI=/usr/common/homes/g/grote/pyMPI']
            self.fopt = '-O3 -qstrict -qarch=auto -qtune=auto'
            return 1

    def aix_xlf_r(self):
        if self.usecompiler('xlf_r', 'xlf95_r'):
            # --- IBM SP, OpenMP
            self.fcompname = 'xlf'
            intsize = struct.calcsize('l')
            if intsize == '4':
                bmax = '-bmaxdata:0x70000000 -bmaxstack:0x10000000'
            else:
                bmax = '-q64'
            f90 = ' -c -WF,-DXLF -qmaxmem=8192 -qdpc=e -qautodbl=dbl4 -qsave=defaultinit -WF,-DESSL %(bmax)s'%locals()
            self.f90free += f90 + ' -qsuffix=f=f90:cpp=F90 -qfree=f90'
            self.f90fixed += f90 + ' -qfixed=132'
            self.f90free += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.f90fixed += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.ld = 'xlf95_r -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp %(bmax)s'%locals()
            if self.implicitnone:
                self.f90free += ' -u'
                self.f90fixed += ' -u'
            self.popt = '-O'
            self.extra_link_args += [bmax]
            self.extra_compile_args = [bmax]
            self.libs = ['xlf90', 'xlopt', 'xlf', 'xlsmp', 'pthreads', 'essl']
            self.fopt = '-O3 -qstrict -qarch=auto -qtune=auto -qsmp=omp'
            return 1

    def aix_mpxlf64(self):
        if self.usecompiler('mpxlf64', 'mpxlf95'):
            # --- IBM SP, parallel
            self.fcompname = 'xlf'
            intsize = struct.calcsize('l')
            if intsize == '4':
                bmax = '-bmaxdata:0x70000000 -bmaxstack:0x10000000'
            else:
                bmax = '-q64'
            f90 = ' -c -WF,-DXLF -qmaxmem=8192 -qdpc=e -qautodbl=dbl4 -qsave=defaultinit -WF,-DMPIPARALLEL -WF,-DESSL %(bmax)s'%locals()
            self.f90free += f90 + ' -qsuffix=f=f90:cpp=F90 -qfree=f90'
            self.f90fixed += f90 + ' -qfixed=132'
            self.f90free += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90fixed += ' -WF,-DFPSIZE=%s'%(realsize)  # ???
            self.f90free += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.f90fixed += ' -WF,-DISZ=%s -qintsize%s'%(intsize, intsize)
            self.ld = 'mpxlf95_r -bE:$(PYTHON)/lib/python$(PYVERS)/config/python.exp %(bmax)s'%locals()
            if self.implicitnone:
                self.f90free += ' -u'
                self.f90fixed += ' -u'
            self.popt = '-O'
            self.extra_link_args += [bmax]
            self.extra_compile_args = [bmax]
            self.libs = ['xlf90', 'xlopt', 'xlf', 'xlomp_ser', 'pthread', 'essl']
            self.defines = ['PYMPI=/usr/common/homes/g/grote/pyMPI']
            self.fopt = '-O3 -qstrict -qarch=auto -qtune=auto'
            return 1

    def aix_pghpf(self):
        if self.usecompiler('pghpf', 'pghpf'):
            # --- Portland group
            self.fcompname = 'pghpf'
            self.f90free += ' -Mextend -Mdclchk'
            self.f90fixed += ' -Mextend -Mdclchk'
            self.f90free += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90fixed += ' -DFPSIZE=%s -r%s'%(realsize, realsize)
            self.f90free += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.f90fixed += ' -DISZ=%s -i%s'%(intsize, intsize)
            self.popt = '-Mcache_align'
            flibroot, b = os.path.split(self.findfile(self.fcompexec))
            self.libdirs = [flibroot + '/lib']
            self.libs = ['pghpf']  # ???
            self.fopt = '-fast -Mcache_align'
            return 1
