# Forthon

[![Supported Python versions](https://img.shields.io/pypi/pyversions/Forthon.svg)](https://pypi.python.org/pypi/Forthon)
[![PyPI version](https://img.shields.io/pypi/v/Forthon.svg)](https://pypi.python.org/pypi/Forthon)
[![PyPI downloads](https://img.shields.io/pypi/dm/Forthon.svg)](https://pypi.python.org/pypi/Forthon)
[![PyPI license](https://img.shields.io/pypi/l/Forthon.svg)](License.txt)

Forthon generates links between Fortran95 and Python. Python is a high level,
object oriented, interactive and scripting language that allows a flexible
and versatile interface to computational tools. The Forthon package generates
the necessary wrapping code which allows access to the Fortran database and
to the Fortran subroutines and functions. This provides a development package
where the computationally intensive parts of a code can be written in
efficient Fortran, and the high level controlling code can be written in the
much more versatile Python language.

## Installing

Requires python versions 3.8 or higher and the numpy package.

To install from the source,

python -m pip install .

Write permission is required for the python library site-packages directory.

Alternatively, it can be installed from PyPI, as in "pip install Forthon".

See the examples, which act as the documentation.

## Example

An example of how to use Forthon can be found in the simpleexample subdirectory. Go into
that directory and run "make". Forthon needs to have been installed.
This will build and install the example. Then run

python run_forthon_example.py

This will produce output like this:

Before setsqrt, x = 0.0

After setsqrt, x = 3.1622776601683795

A more extensive example can be found in the example subdirectory. It can be built and run with the "make" command.

## Release notes

See [Release_Notes](Release_Notes)

## License

See [License.txt](License.txt) for license information.

