#!/usr/bin/env python
# To use:
#       python setup.py install
#
import os, os.path, sys, string, re
from glob import glob

try:
    import distutils
    from distutils.command.install import install
    from distutils.core import setup, Extension
    from distutils.command.install_data import install_data
    from distutils.sysconfig import get_python_lib
except:
    raise SystemExit, "Distutils problem"

data_files_home = os.path.join(get_python_lib(),'Forthon')

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
       version = '2.0',
       author = 'David P. Grote',
       author_email = "DPGrote@lbl.gov",
       description = "Fortran wrapper/code development package",
       platforms = "Unix, Windows (cygwin), Mac OSX",
       packages = ['Forthon'],
       package_dir = {'Forthon': 'Lib'},
       data_files = [(data_files_home,['Src/Forthon.h','Src/Forthon.c'])],
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
