"0.8.17"
commithash = "aa336d1"

import string
import version
import subprocess

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
    updatefile('setup.py',vvold,vvnew)

    # --- Update the commithash of the release.
    # --- This line is the same as the line in setup.py.
    commithash = subprocess.check_output('git log -n 1 --pretty=%h',stderr=subprocess.STDOUT,shell=True).strip()
    updatefile('version.py',version.commithash,commithash)
    updatefile('setup.py',version.commithash,commithash)

if __name__ == "__main__":
    print version.__doc__
