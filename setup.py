#!/usr/bin/env python
# To use:
#       python setup.py install
#
import os, sys, string, re
from glob import glob

try:
    import distutils
    from distutils.command.install import install
    from distutils.core import setup, Extension
    from distutils.sysconfig import get_python_lib
except:
    raise SystemExit, "Distutils problem"

data_files_home = os.path.join(get_python_lib(),'Forthon')

# --- Get around a "bug" in disutils on 64 big systems. When there is no extension to be installed, distutils
# --- will put the scripts in /usr/lib/... instead of /usr/lib64. This fixes it.
if get_python_lib().find('lib64') != -1:
  import distutils.command.install
  distutils.command.install.INSTALL_SCHEMES['unix_prefix']['purelib'] = '$base/lib64/python$py_version_short/site-packages'
  distutils.command.install.INSTALL_SCHEMES['unix_home']['purelib'] = '$base/lib64/python$py_version_short/site-packages'

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

setup (name = "Forthon",
       version = '0.7.5',
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
       extra_path = 'Forthon',
       packages = [''],
       package_dir = {'': 'Lib'},
       data_files = [(data_files_home,['Notice','Src/Forthon.h','Src/Forthon.c'])],
       scripts = [Forthon]
       )

# --- This is probably the worst possible way to do this, but here goes...
if sys.platform in ["linux2","hp","darwin","SP"]:
  os.system('chmod -R go+r '+data_files_home)
# os.system('chmod go+x '+os.path.join(data_files_home,'preprocess.py'))
# os.system('chmod go+x '+os.path.join(data_files_home,'wrappergenerator.py'))

# --- Clean up the extra file created on win32.
if sys.platform == 'win32':
  os.system("rm -f Forthon.py")
