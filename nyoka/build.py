#!/usr/bin/env python
 
import os
import re
import sys
import errno
import fileinput
import subprocess
from shutil import copy, copyfileobj

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def sed(find, replace, filename):
    readfile = open(filename, 'r')
    data = readfile.read()
    readfile.close()

    writefile = open(filename, 'w+')
    writefile.write(re.sub(find, replace, data))

curdir = os.path.abspath(os.curdir)
def gen(name, folder, clean=""):
    text = "CLEANING " if clean is "clean" else "BUILDING "
    print("\033[33;1m" + ("-" * 10) + text + name + ("-" * 10) + "\033[0m")
    if subprocess.call([sys.executable, os.path.join(curdir, "PMML43Ext", "gen.py"), name, clean], cwd=folder) != 0:
        exit(1)

pmml43ExtFolderPath = os.path.join(curdir, "PMML43Ext")

if "clean" in sys.argv:
    find = subprocess.Popen(["find", ".", "-type", "f"], stdout=subprocess.PIPE)
    try:
        grep = subprocess.check_output(["grep", ".pyc"], stdin=find.stdout)
    except:
        grep = b''
    if sys.version_info[0] == 3:
        grep = grep.decode('utf-8')
    files = list(filter(None, grep.split("\n")))
    for file in files:
        print("\033[36;1mCLEANING " + file + "\033[0m")
        if os.path.exists(file): os.remove(file)
    gen("PMML43Ext", curdir, "clean")
    subprocess.call(["make", "-C", os.path.join(curdir, "PMML43Ext", "doc"), "clean"])
    exit(0)

pmml43ExtSuperPy = os.path.join(pmml43ExtFolderPath, "pmml43ExtSuper.py")
pmml43ExtPy = os.path.join(pmml43ExtFolderPath, "pmml43Ext.py")
nyoka_pmml43ExtSuperPy = os.path.join(curdir, "PMML43ExtSuper.py")
nyoka_pmml43ExtPy = os.path.join(curdir, "PMML43Ext.py")
wrapper43Ext = open(os.path.join(pmml43ExtFolderPath, "wrapper43Ext.py"), 'r')

gen("PMML43Ext", pmml43ExtFolderPath)
copy(pmml43ExtSuperPy, nyoka_pmml43ExtSuperPy)
copy(pmml43ExtPy, nyoka_pmml43ExtPy)
sed(r"pmml43Ext\b", "nyoka.PMML43Ext", nyoka_pmml43ExtPy)
sed(r"pmml43ExtSuper\b", "nyoka.PMML43ExtSuper", nyoka_pmml43ExtPy)
sed(r"def parse\(", "def parseSub(", nyoka_pmml43ExtPy)
with open(nyoka_pmml43ExtPy, 'a') as f: f.write(wrapper43Ext.read())

subprocess.call([sys.executable,
                 os.path.join(curdir, "PMML43Ext", "doc.py"),
                 "doc" if "doc" in sys.argv else "",
                 "open" if "open" in sys.argv else ""],
                 cwd=pmml43ExtFolderPath)

if "test" in sys.argv:
    subprocess.call([sys.executable, os.path.join(curdir, "PMML43Ext", "test.py")])
