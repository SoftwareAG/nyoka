import sys
from nyoka.PMML43Ext import *

net1_target_class1 = .8873
net1_target_class2 = .0644
net1_target_class3 = .0209
net1_target_class4 = .0055
net1_target_class5 = .0013

net2_target_class1 = .9446
net2_target_class2 = .0223
net2_target_class3 = .0012
net2_target_class4 = .0067
net2_target_class5 = .0005

net3_target_class1 = .9292
net3_target_class2 = .0198
net3_target_class3 = .0501
net3_target_class4 = .0001
net3_target_class5 = .0002

net1 = { "African Elephant": net1_target_class1,
         "Trunker": net1_target_class2,
         "Indian Elephant": net1_target_class3,
         "Wolf": net1_target_class4,
         "Dust Bunny": net1_target_class5
         }
net2 = { "African Elephant": net2_target_class1,
         "Indian Elephant": net2_target_class2,
         "Trunker": net2_target_class3,
         "Dog": net2_target_class4,
         "Wolf": net2_target_class5
         }
net3 = { "African Elephant": net3_target_class1,
         "Indian Elephant": net3_target_class2,
         "Trunker": net3_target_class3,
         "Camel": net3_target_class4,
         "Bat": net3_target_class5
         }

if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) == 2:
        pmml = parse(args[0])
        pmml.export(open(args[1], "w"), 0, "")
