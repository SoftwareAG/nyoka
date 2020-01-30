import re, os, sys, platform
def sed(find, replace, filename):
    if 'windows' in platform.platform().lower():
        replace = os.path.splitdrive(replace)[-1]
        replace = replace.replace('\\','/')
    readfile = open(filename, 'r')
    data = readfile.read()
    readfile.close()
    writefile = open(filename, 'w+')
    writefile.write(re.sub(find, replace, data))

sed('\?\?\?', os.path.join(os.path.abspath(os.curdir), 'behaviors') + os.path.sep, 'behaviorsDir.xml')
