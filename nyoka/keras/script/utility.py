import re

def update_pmml(fileName):

    with open(fileName,'r') as ff:
        fileObj = ff.read()

    fileObj=re.sub(r'architectureName=\"[A-Za-z\s]+\"','architectureName="mobilenet"',fileObj)
    fileObj=re.sub(r'max_value=\"[0-9\.]+\"','',fileObj)
    fileObj=fileObj.replace('paddingType','pad')
    fileObj=re.sub(r'trainable=\"(true|false)\"','',fileObj)
    fileObj=re.sub(r'units=\"[0-9]+\"','',fileObj)

    with open(fileName,'w') as ff:
        ff.write(fileObj)