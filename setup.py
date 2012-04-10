#!/usr/bin/env python
# To use:
#       python setup.py install
#
import os, sys, stat

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
    os.chmod('License.txt',perm644)
    os.chmod('Src/Forthon.h',perm644)
    os.chmod('Src/Forthon.c',perm644)
except:
    print('Permissions on License.txt and Src files needs to be set by hand')

# --- Get around a "bug" in disutils on 64 bit systems. When there is no
# --- extension to be installed, distutils will put the scripts in
# --- /usr/lib/... instead of /usr/lib64.
if distutils.sysconfig.get_config_vars()["LIBDEST"].find('lib64') != -1:
    import distutils.command.install
    INSTALL_SCHEMES['unix_prefix']['purelib'] = '$base/lib64/python$py_version_short/site-packages'
    INSTALL_SCHEMES['unix_home']['purelib'] = '$base/lib64/python$py_version_short/site-packages'

# --- With this, the data_files listed in setup will be installed in
# --- the usual place in site-packages.
for scheme in INSTALL_SCHEMES.values():
    scheme['data'] = scheme['purelib']

# --- Normally, the package building script is called Forthon, but on Windows,
# --- it works better if it is called Forthon.py.
if sys.platform == 'win32':
    Forthon = 'Forthon.bat'
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
    Forthon = 'Forthon'

# --- Force the deletion of the build directory so that a fresh install is
# --- done every time. This is needed since otherwise, after the first install,
# --- each subsequent install would use the same Forthon script and not update
# --- the python path in it appropriately.
os.system("rm -rf build")

setup (name = "Forthon",
       version = '0.8.8',
       author = 'David P. Grote',
       author_email = "DPGrote@lbl.gov",
       url = "http://hifweb.lbl.gov/Forthon",
       download_url = "http://hifweb.lbl.gov/Forthon/Forthon.tgz",
       description = "Fortran95 wrapper/code development package",
       long_description = """
Forthon provides an extensive wrapping of Fortran95 code, giving access to
routines and to any data in Fortran modules. Forthon also
provides an extensive wrapping of Fortran derived types, giving access to
derived type members, allowing passing of derived types into Fortran routines,
and creation of instances at the Python level. A mechanism for automatic
building of extension modules is also included. Versions using Numeric and
Numpy are available.""",
       platforms = "Linux, Unix, Windows (cygwin), Mac OSX",
       packages = ['Forthon'],
       package_dir = {'Forthon': 'Lib'},
       data_files = [('Forthon', ['License.txt','Src/Forthon.h','Src/Forthon.c'])],
       scripts = [Forthon],
       cmdclass = {'build_py':build_py}
       )

# --- Clean up the extra file created on win32.
if sys.platform == 'win32':
    os.system("rm -f Forthon.py")

