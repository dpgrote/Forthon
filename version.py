"0.10.0"
commithash = "9f48ac4"

import sys
import version
import subprocess

def updatefile(filename, vvold, vvnew):
    # --- Update the version number in the given file.
    ff = open(filename, 'r')
    text = ff.read()
    ff.close()
    text = text.replace(vvold, vvnew)
    ff = open(filename, 'w')
    ff.write(text)
    ff.close()

def update(major=False, release=False):
    vvold = version.__doc__
    vv = vvold.split('.')
    if release:
        vv[0] = str(int(vv[0])+1)
        vv[1] = '0'
        vv[2] = '0'
    elif major:
        vv[1] = str(int(vv[1])+1)
        vv[2] = '0'
    else:
        vv[2] = str(int(vv[2])+1)

    vvnew = '.'.join(vv)

    # --- Update the version number in the files.
    updatefile('version.py', vvold, vvnew)
    updatefile('docs/index.html', vvold, vvnew)
    updatefile('setup.py', vvold, vvnew)

    # --- Update the commithash of the release.
    # --- This line is the same as the line in setup.py.
    commithash = subprocess.check_output('git log -n 1 --pretty=%h', stderr=subprocess.STDOUT, shell=True, text=True).strip()
    updatefile('version.py', version.commithash, commithash)
    updatefile('setup.py', version.commithash, commithash)

if __name__ == "__main__":
    print(version.__doc__)
