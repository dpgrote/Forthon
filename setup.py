#!/usr/bin/env python
# To use:
#       python setup.py install
#
import os, sys, stat
import subprocess

from setuptools import setup

try:
    perm644 = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH | stat.S_IWUSR
    os.chmod('source/License.txt',perm644)
    os.chmod('source/Forthon.h',perm644)
    os.chmod('source/Forthon.c',perm644)
except:
    print('Permissions on License.txt and Src files needs to be set by hand')

# --- Write out version information to the version.py file.
version = '0.10.1'
try:
    commithash = subprocess.check_output(['git', 'log', '-n', '1', '--pretty=%h'], stderr=subprocess.STDOUT, text=True).strip()
    if not commithash:
        # --- If git is not available, commithash will be an empty string.
        raise OSError('git commit hash not found')
except (subprocess.CalledProcessError, OSError):
    # --- Error returned by subprocess.check_output if git command failed
    # --- This version was obtained from a non-git distrobution. Use the
    # --- saved commit hash from the release.
    # --- This is automatically updated by version.py.
    commithash = '9d19e48'

with open('source/version.py','w') as ff:
    ff.write("version = '%s'\n"%version)
    ff.write("gitversion = '%s'\n"%commithash)

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
os.system("rm -rf build dist")

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
        'Programming Language :: Python :: 3.7',
        ],
       packages = ['Forthon'],
       package_dir = {'Forthon': 'source'},
       package_data = {'Forthon': ['License.txt','Forthon.h','Forthon.c']},
       scripts = [Forthon],
       )

# --- Clean up the extra file created on win32.
if sys.platform == 'win32':
    os.system("rm -f Forthon.py")

