#!/usr/bin/env python
# Created by David P. Grote, March 6, 1998

from cfinterface import *

# --- This is a bit of a hack. The preprocessor takes many of the same arguments as Forthon, but not all.
# --- It uses Forthon_options to avoid duplication.
# --- It uses the first two positional arguments as the input and output filenames.
from Forthon_options import args

def py_ifelse(m, v, t, f=''):
    if m==v:
        return t
    else:
        return f

def main():
    infile = args.pkgname
    outfile = args.remainder[0]

    with open(infile, 'r') as fin:
        text = fin.readlines()

    with open(outfile, 'w') as fout:

        for line in text:
            if line.startswith('%'):
                fout.write(eval(line[1:], globals()))
            else:
                fout.write(line)

if __name__ == '__main__':
    main()

