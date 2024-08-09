import sys
import subprocess

sys.path.insert(0, 'Forthon')
import version

def updatefile(filename, vvold, vvnew):
    # --- Update the version number in the given file.
    with open(filename, 'r') as ff:
        text = ff.read()
    text = text.replace(vvold, vvnew)
    with open(filename, 'w') as ff:
        ff.write(text)

def update(major=False, release=False):
    vvold = version.version
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
    updatefile('Forthon/version.py', vvold, vvnew)
    updatefile('docs/index.html', vvold, vvnew)

    # --- Update the commithash of the release.
    commithash = subprocess.check_output('git log -n 1 --pretty=%h', stderr=subprocess.STDOUT, shell=True, text=True).strip()
    updatefile('Forthon/version.py', version.commithash, commithash)

if __name__ == "__main__":
    print(version.version)
