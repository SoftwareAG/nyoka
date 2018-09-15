#!/usr/bin/env python
 
import os
import sys
import errno
import fileinput
import re
from shutil import copy, copyfileobj

def sed(find, replace, filename):
    readfile = open(filename, 'r')
    data = readfile.read()
    readfile.close()

    writefile = open(filename, 'w+')
    writefile.write(re.sub(find, replace, data))

def sed_n(find, filename):
    lines = open(filename, 'r').readlines()
    pattern = re.compile(find)
    for line in lines:
        if pattern.match(line):
            for match in pattern.findall(line):
                yield lines.index(line)

def insert(data, index, filename):
    f = open(filename, 'r')
    contents = f.readlines()
    f.close()

    contents.insert(index, data)

    f = open(filename, 'w+')
    contents = "".join(contents)
    f.write(contents)
    f.close()

quotes = '"""'
curdir = os.path.abspath(os.curdir)
pmml43Extdoc = os.path.join(curdir, 'doc', 'resources', 'PMML43Ext')
pmml43ExtSuperdoc = os.path.join(curdir, 'doc', 'resources', 'PMML43ExtSuper')
directories = [pmml43Extdoc, pmml43ExtSuperdoc]

pmml43ExtPy = os.path.join(curdir, '..', 'PMML43Ext.py')
pmml43ExtSuperPy = os.path.join(curdir, '..', 'PMML43ExtSuper.py')
pyfiles = [pmml43ExtPy, pmml43ExtSuperPy]

for (index, directory) in enumerate(directories):
    pyname = os.path.splitext(os.path.basename(pyfiles[index]))[0]
    for file in [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.txt')]:
        doc = '"""' + open(file, 'r').read().rstrip() + '"""\n'
        lineCount = len(doc.splitlines())
        classname = os.path.splitext(os.path.basename(file))[0]
        match = list(sed_n(r'^class ' + classname, pyfiles[index]))

        if len(match) == 1:
            range_ = match[0] + lineCount
            insert(doc, match[0] + 1, pyfiles[index])
            insert('    ', match[0] + 1, pyfiles[index])
            if range_ > match[0] + 2:
                for i in range(match[0] + 2, range_ + 1):
                    insert('    ', i, pyfiles[index])
            
            print("\033[36;1mDOCUMENTED " + classname + " IN " + pyname + "\033[0m")

if "doc" in sys.argv:
    import webbrowser
    import subprocess
    import site

    def install(package):
        subprocess.call([sys.executable, "-m", "pip", "install", "--user", package])
        site = __import__("site")

    install("sphinx")
    subprocess.call(["make", "-C", os.path.join(curdir, "doc"), "html"])

    if "open" in sys.argv:
        webbrowser.open('file://' + os.path.join(curdir, "doc", "build", "html", "index.html"))
