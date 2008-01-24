"0.7.7"

import string
import version

def update():
  vvold = version.__doc__
  vv = string.split(vvold,'.')
  vv[2] = str(int(vv[2])+1)
  vvnew = string.join(vv,'.')

  # --- Update the version number in this file.
  ff = open('version.py','r')
  lines = ff.readlines()
  ff.close()
  ff = open('version.py','w')
  ff.write('"'+vvnew+'"\n')
  for line in lines[1:]:
    ff.write(line)

  # --- Update the version number in the html file.
  ff = open('docs/index.html','r')
  text = ff.read()
  ff.close()
  text = text.replace(vvold,vvnew)
  ff = open('docs/index.html','w')
  ff.write(text)
  ff.close()

if __name__ == "__main__":
  print version.__doc__
