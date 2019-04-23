import re

def update_pmml(fileName):

    with open(fileName,'r') as fp:
        lines = fp.readlines()
    
    updatedLines = []
    for line in lines:
        if 'max_value="6.0"' in line:
            line = line.replace(' max_value="6.0"',"")
            line = line.replace('activationFunction="rectifier"','activationFunction="reLU6"')
        line = line.replace('paddingType','pad')
        line=re.sub(r' trainable=\"(true|false)\"',"",line)
        line=re.sub(r' units=\"[0-9]+\"',"",line)
        line=re.sub(r'architectureName=\"[A-Za-z\s]+\"','architectureName="mobilenet"',line)
        updatedLines.append(line)

    with open(fileName,'w') as fp:
        fp.writelines(updatedLines)