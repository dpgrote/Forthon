#!/usr/bin/env python
# To use:
#       python setup.py install
#
import os, sys, stat
import subprocess

try:
    import distutils
    from distutils.core import setup
    from distutils.command.install import INSTALL_SCHEMES
except:
    raise SystemExit("Distutils problem")

try:
    from distutils.command.build_py import build_py_2to3 as build_py
except ImportError:
    from distutils.command.build_py import build_py

try:
    perm644 = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR
    os.chmod('source/License.txt',perm644)
    os.chmod('source/Forthon.h',perm644)
    os.chmod('source/Forthon.c',perm644)
except:
    print('Permissions on License.txt and Src files needs to be set by hand')

# --- Write out version information to the version.py file.
version = '0.8.32'
try:
    # --- In python3, check_output or Popen returns a byte string that needs to be decoded to get the string.
    # --- The decode method is mostly harmless in python2.
    #bcommithash = subprocess.check_output(['git', 'log', '-n', '1', '--pretty=%h'], stderr=subprocess.STDOUT).strip()
    # --- Needed for Py2.6 (which doesn't have subprocess.check_output)
    bcommithash = subprocess.Popen(['git', 'log', '-n', '1', '--pretty=%h'], stderr=subprocess.PIPE, stdout=subprocess.PIPE).communicate()[0].strip()
    commithash = bcommithash.decode()
    if not commithash:
        # --- If git is not available, commithash will be an empty string.
        raise OSError('git commit hash not found')
except (subprocess.CalledProcessError, OSError):
    # --- Error returned by subprocess.check_output if git command failed
    # --- This version was obtained from a non-git distrobution. Use the
    # --- saved commit hash from the release.
    # --- This is automatically updated by version.py.
    commithash = '6a62330'

with open('source/version.py','w') as ff:
    ff.write("version = '%s'\n"%version)
    ff.write("gitversion = '%s'\n"%commithash)

# --- Get around a "bug" in disutils on 64 bit systems. When there is no
# --- extension to be installed, distutils will put the scripts in
# --- /usr/lib/... instead of /usr/lib64.
if distutils.sysconfig.get_config_vars()["LIBDEST"].find('lib64') != -1:
    for scheme in INSTALL_SCHEMES.values():
        scheme['purelib'] = scheme['platlib']

if sys.hexversion < 0x03000000:
    Forthonroot = 'Forthon'
elif sys.hexversion >= 0x03000000:
    Forthonroot = 'Forthon3'

# --- Normally, the package building script is called Forthon, but on Windows,
# --- it works better if it is called Forthon.py.
if sys.platform == 'win32':
    Forthon = Forthonroot+'.bat'
    ff = open(Forthon,'w')
    file = """\
@echo off
set sys_argv=
:Loop
if "%%1"=="" GOTO Continue
set sys_argv=%%sys_argv%% %%1
SHIFT
GOTO Loop
:Continue
%s -c "import Forthon.Forthon_builder" %%sys_argv%%
"""%(sys.executable)
    ff.write(file)
    ff.close()
else:
    Forthon = Forthonroot

# --- Force the deletion of the build directory so that a fresh install is
# --- done every time. This is needed since otherwise, after the first install,
# --- each subsequent install would use the same Forthon script and not update
# --- the python path in it appropriately.
os.system("rm -rf build")

setup (name = "Forthon",
       version = version,
       author = 'David P. Grote',
       author_email = "DPGrote@lbl.gov",
       license = "BSD-3-Clause-LLNL",
       url = "http://hifweb.lbl.gov/Forthon",
       description = "Fortran95 wrapper/code development package",
       long_description = """
Forthon provides an extensive wrapping of Fortran95 code, giving access to
routines and to any data in Fortran modules. Forthon also
provides an extensive wrapping of Fortran derived types, giving access to
derived type members, allowing passing of derived types into Fortran routines,
and creation of instances at the Python level. A mechanism for automatic
building of extension modules is also included. Versions using Numeric and
Numpy are available.""",
       platforms = 'any',
       classifiers = [
        'Programming Language :: Python',
        'Development Status :: 5 - Production/Stable',
        'Natural Language :: English',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'License :: Free To Use But Restricted',
        'Operating System :: OS Independent',
        'Topic :: Software Development :: Build Tools',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        ],
       packages = ['Forthon'],
       package_dir = {'Forthon': 'source'},
       package_data = {'Forthon': ['License.txt','Forthon.h','Forthon.c']},
       scripts = [Forthon],
       cmdclass = {'build_py':build_py}
       )

# --- Clean up the extra file created on win32.
if sys.platform == 'win32':
    os.system("rm -f Forthon.py")

