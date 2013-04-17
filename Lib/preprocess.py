#!/usr/bin/env python
# Created by David P. Grote, March 6, 1998
# $Id: preprocess.py,v 1.9 2009/09/08 18:01:56 dave Exp $

from cfinterface import *
import sys
from Forthon_options import args

def py_ifelse(m,v,t,f=''):
    if m==v:
        return t
    else:
        return f

def main():
    file = open(args[0],'r')
    text = file.readlines()
    sys.stdout = open(args[1],'w')

    for line in text:
        if line[0] == '%':
            print eval(line[1:],globals())
        else:
            print line[:-1]

if __name__ == '__main__':
    main()

