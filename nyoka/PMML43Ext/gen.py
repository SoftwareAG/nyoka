import os
import re
import sys
import errno
import traceback
import fileinput
import subprocess
from shutil import copy, copyfileobj

subname = 'pmml' + sys.argv[1][4:]
name = subname + "Super"
xsd = subname + ".xsd"
pmmlpy = name + ".py"
pmmlpysub = subname + ".py"
behaviorFile = "behaviors.xml"
behaviorFileWithDir = "behaviorsDir.xml"

def call(popenargs):
    try:
        subprocess.call(popenargs)
    except Exception as exc:
        print("\n\033[31;1mFAILED TO GENERATE CLASSES\033[0m")
        type_, value_, traceback_ = sys.exc_info()
        print("\033[31;1m" + str(exc) + "\033[0m")
        for line in traceback.format_tb(traceback_):
            print("\033[31m" + line.rstrip("\n") + "\033[0m")
        exit(1)

def check_output(popenargs):
    try:
        return subprocess.check_output(popenargs)
    except Exception as exc:
        print("\033[31;1mFAILED TO GENERATE CLASSES\033[0m")
        type_, value_, traceback_ = sys.exc_info()
        print("\033[31;1m" + str(exc) + "\033[0m")
        for line in traceback.format_tb(traceback_):
            print("\033[31m" + line.rstrip("\n") + "\033[0m")
        exit(1)
        
def install(package):
    call([sys.executable, "-m", "pip", "install", "--user", package])
    site = __import__("site")

curdir = os.path.abspath(os.curdir)

if "clean" in sys.argv:
    files = [pmmlpy, pmmlpysub, behaviorFileWithDir]
    for file in [os.path.join(curdir, "PMML43Ext", x) for x in files]:
        print("\033[36;1mCLEANING " + file + "\033[0m")
        if os.path.exists(file): os.remove(file)
    print("\033[32;1mSUCCESSFULLY CLEANED BUILD FILES\033[0m")
    exit(0)

install("lxml")
copy(behaviorFile, behaviorFileWithDir)
call([sys.executable, os.path.join(curdir, "behavior_script.py")])

print("\033[33;1mGENERATING CLASSES\033[0m")
try:
    call(["python", os.path.join(curdir, "gds_local.py"), "--no-warnings", "--export=write literal etree", "--super=" + name,
    "--subclass-suffix=", "-o", pmmlpy, "-s", pmmlpysub, "-b", behaviorFileWithDir, "-f", os.path.join("..", xsd)])
except Exception as e:
    print("\033[31;mCLASSES MUST BE GENERATED WITH PYTHON 3\nPLEASE INSTALL PYTHON 3 AND TRY AGAIN\033[0m")
print("\n\033[32;1mSUCCESSFULLY GENERATED CLASSES\033[0m")
