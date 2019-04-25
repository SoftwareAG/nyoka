#!/usr/bin/env python

import subprocess
import site
import sys
import os

score = 0
count = 0
curdir = os.path.abspath(os.curdir)
os.environ["PYTHONPATH"] = os.path.join(curdir, '..')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", "--user", package])
    site = __import__("site")

def run(script, *args):
    global count, score
    count += 1
    print('\033[33;1m' + ('-' * 10) + 'RUNNING SCRIPT: ' + script + ('-' * 10) + '\033[0m')
    cmd_args = [sys.executable, os.path.join(curdir, "tests", script)]
    [cmd_args.append(os.path.join(curdir, "tests", x)) for x in args]
    if subprocess.call(cmd_args, env=os.environ) == 0:
        print('\033[32;1mTEST PASSED\033[0m')
        score += 1
    else:
        print('\033[31;1mTEST FAILED\033[0m')

def notebook(ipynb):
    global count, score
    count += 1
    print('\033[33;1m' + ('-' * 10) + 'RUNNING JUPYTER NOTEBOOK: ' + ipynb + ('-' * 10) + '\033[0m')

    import nbformat
    from nbconvert.preprocessors import ExecutePreprocessor

    with open(os.path.join(curdir, "tests", ipynb)) as f:
        nb = nbformat.read(f, as_version=4)
        ep = ExecutePreprocessor(timeout=600, kernel_name='python')
        ep.preprocess(nb, {'metadata': {'path': './tests/'}})

    with open(os.path.join(curdir, "tests", 'executed_' + ipynb), 'wt') as f:
        nbformat.write(nb, f)

    with open(os.path.join(curdir, "tests", 'executed_' + ipynb)) as f:
        import json
        nbdict = json.loads(f.read())
        text = ""

        for cell in nbdict["cells"]:
            if "outputs" in cell and len(cell["outputs"]) > 0:
                for output in cell["outputs"]:
                    if "text" in output:
                        text += "Out[" + str(cell["execution_count"]) + "]:\n"
                        for line in output["text"]:
                            text += ' ' * 4 + line
        
        print(text)

    print('\033[32;1mTEST PASSED\033[0m')
    score += 1

install("numpy")
install("jupyter")
install("pillow")
install("tensorflow")
install("keras")
install("keras-applications")
run("test_script.py", "script_test.pmml", "script_tested.pmml")
run("test2.py")
run("testbase64.py")
notebook("Nyoka_String_Export.ipynb")
notebook("Nyoka_Script_Tag.ipynb")
notebook("1_SVM.ipynb")
notebook("Nyoka_Output_Field_Threshold.ipynb")

percent = str(int(float(score)/float(count) * 100))
print("\033[36;1mTESTS PASSED: " + str(score) + "/" + str(count) + ", " + percent + "%\033[0m")
