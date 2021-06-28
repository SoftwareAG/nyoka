#!/usr/bin/env python
"""
 Copyright (c) 2004-2016 Zementis, Inc.
 Copyright (c) 2016-2021 Software AG, Darmstadt, Germany and/or Software AG USA Inc., Reston, VA, USA, and/or its

 SPDX-License-Identifier: Apache-2.0

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
 """
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
    if subprocess.call([sys.executable, os.path.join(curdir, "PMML44", "gen.py"), name, clean], cwd=folder) != 0:
        exit(1)

pmml44FolderPath = os.path.join(curdir, "PMML44")

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
    gen("PMML44", curdir, "clean")
    subprocess.call(["make", "-C", os.path.join(curdir, "PMML44", "doc"), "clean"])
    exit(0)

pmml44SuperPy = os.path.join(pmml44FolderPath, "pmml44Super.py")
pmml44Py = os.path.join(pmml44FolderPath, "pmml44.py")
nyoka_pmml44SuperPy = os.path.join(curdir, "PMML44Super.py")
nyoka_pmml44Py = os.path.join(curdir, "PMML44.py")
wrapper44 = open(os.path.join(pmml44FolderPath, "wrapper44.py"), 'r')

gen("PMML44", pmml44FolderPath)
copy(pmml44SuperPy, nyoka_pmml44SuperPy)
copy(pmml44Py, nyoka_pmml44Py)
sed(r"pmml44\b", "nyoka.PMML44", nyoka_pmml44Py)
sed(r"pmml44Super\b", "nyoka.PMML44Super", nyoka_pmml44Py)
sed(r"def parse\(", "def parseSub(", nyoka_pmml44Py)
with open(nyoka_pmml44Py, 'a') as f: f.write(wrapper44.read())

# subprocess.call([sys.executable,
#                  os.path.join(curdir, "PMML44", "doc.py"),
#                  "doc" if "doc" in sys.argv else "",
#                  "open" if "open" in sys.argv else ""],
#                  cwd=pmml44FolderPath)

# if "test" in sys.argv:
#     subprocess.call([sys.executable, os.path.join(curdir, "PMML44", "test.py")])
