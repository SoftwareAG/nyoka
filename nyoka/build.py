#!/usr/bin/env python
 
import os
import sys
import errno
import fileinput
import re
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
pmml43FolderPath = os.path.join(curdir, "PMML43")
pmml43ExtFolderPath = os.path.join(curdir, "PMML43Ext")
nyoka_pmmlFolderPath = os.path.join(curdir, "nyoka/pmml")

pmml43SuperPy = os.path.join(pmml43FolderPath, "pmml43.py")
pmml43Py = os.path.join(pmml43FolderPath, "nyoka_local.py")
pmml43ExtSuperPy = os.path.join(pmml43ExtFolderPath, "pmml43Ext.py")
pmml43ExtPy = os.path.join(pmml43ExtFolderPath, "nyoka_local.py")
Base64Py = os.path.join(pmml43FolderPath, "Base64.py")

nyoka_pmml43SuperPy = os.path.join(nyoka_pmmlFolderPath, "PMML43Super.py")
nyoka_pmml43Py = os.path.join(nyoka_pmmlFolderPath, "PMML43.py")
nyoka_pmml43ExtSuperPy = os.path.join(nyoka_pmmlFolderPath, "PMML43ExtSuper.py")
nyoka_pmml43ExtPy = os.path.join(nyoka_pmmlFolderPath, "PMML43Ext.py")
nyoka_pmmlBase64Py = os.path.join(nyoka_pmmlFolderPath, "Base64.py")

wrapper43 = open(os.path.join(curdir, "wrapper43.py"), 'r')
wrapper43Ext = open(os.path.join(curdir, "wrapper43Ext.py"), 'r')

mkdir_p("nyoka")
mkdir_p("nyoka/pmml")
os.system("make -C " + pmml43FolderPath)
os.system("make -C " + pmml43ExtFolderPath)
copy(pmml43SuperPy, nyoka_pmml43SuperPy)
copy(pmml43Py, nyoka_pmml43Py)
copy(pmml43ExtSuperPy, nyoka_pmml43ExtSuperPy)
copy(pmml43ExtPy, nyoka_pmml43ExtPy)
copy(Base64Py, nyoka_pmmlBase64Py)
sed(r"pmml43", "nyoka.pmml.PMML43Super", nyoka_pmml43Py)
sed(r"pmml43Ext", "nyoka.pmml.PMML43ExtSuper", nyoka_pmml43ExtPy)
sed(r"def parse\(", "def parseSub(", nyoka_pmml43Py)
sed(r"def parse\(", "def parseSub(", nyoka_pmml43ExtPy)
open(nyoka_pmml43Py, 'a').write(wrapper43.read())
open(nyoka_pmml43ExtPy, 'a').write(wrapper43Ext.read())

wrapper43.close()
wrapper43Ext.close()

import doc
