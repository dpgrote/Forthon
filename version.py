"0.8.10"

import string
import version

def updatefile(filename,vvold,vvnew):
    # --- Update the version number in the given file.
    ff = open(filename,'r')
    text = ff.read()
    ff.close()
    text = text.replace(vvold,vvnew)
    ff = open(filename,'w')
    ff.write(text)
    ff.close()

def update():
    vvold = version.__doc__
    vv = string.split(vvold,'.')
    vv[2] = str(int(vv[2])+1)
    vvnew = string.join(vv,'.')

    # --- Update the version number in the files.
    updatefile('version.py',vvold,vvnew)
    updatefile('docs/index.html',vvold,vvnew)
    updatefile('Lib/__init__.py',vvold,vvnew)
    updatefile('setup.py',vvold,vvnew)

if __name__ == "__main__":
    print version.__doc__
