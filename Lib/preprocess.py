#!/usr/bin/env python
# Created by David P. Grote, March 6, 1998
# $Id: preprocess.py,v 1.1 2004/01/06 18:35:13 dave Exp $

from cfinterface import *
import sys
import getopt

def py_ifelse(m,v,t,f=''):
  if m==v:
    return t
  else:
    return f

def main():
  optlist,args=getopt.getopt(sys.argv[1:],"at:",["f90","f90f","2underscores"])

  file = open(args[0],'r')
  text = file.readlines()
  sys.stdout = open(args[1],'w')

  for line in text:
    if line[0] == '%':
      print eval(line[1:],globals())
    else:
      print line[:-1]

if __name__ == "__main__":
  main()
