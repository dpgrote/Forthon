"0.7.7"

import string
import version

def update():
  vv = string.split(version.__doc__,'.')
  vv[2] = str(int(vv[2])+1)
  ff = open('version.py','r')
  lines = ff.readlines()
  ff.close()
  ff = open('version.py','w')
  ff.write('"'+string.join(vv,'.')+'"\n')
  for line in lines[1:]:
    ff.write(line)

if __name__ == "__main__":
  print version.__doc__
