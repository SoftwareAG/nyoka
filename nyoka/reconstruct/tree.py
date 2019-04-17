import numpy as np
from sklearn.tree import DecisionTreeClassifier


class Tree():
    # all_node_list = list()
    root = None
    fields = list()
    classes = list()

    class Node():
        def __init__(self):
            self.field = ''
            self.value = -2
            self.score = -2
            self.left = None
            self.right = None
            self.parent = None
            self.recordCount = None

    def __init__(self, features, classes=None):
        self.root = self.Node()
        self.fields = features
        self.classes = classes
        self.all_node_list = list()

    def get_node_info(self, all_node):
        for node in all_node:
            if not node.get_score():
                score = -2
            else:
                score = node.get_score()
            sp = node.get_SimplePredicate()
            value = sp.get_value()
            field = sp.get_field()
            operator = sp.get_operator()
            record_count = node.get_recordCount()
            self.all_node_list.append((score, field, value, operator, record_count))
            if len(node.get_Node()) > 0:
                self.get_node_info(node.get_Node())
        self.root.field = self.all_node_list[0][1]
        self.root.value = self.all_node_list[0][2]

    def predict(self, sample):
        prob = list()
        for rec in sample:
            prob.append(self._predict(self.root, rec))
        return np.array(prob)

    def _predict(self, root, sample):
        if root.value == -2:
            if len(self.classes) > 0:
                prob = [0] * len(self.classes)
                if self.classes[0].__class__.__name__ == 'int':
                    prob[self.classes.index(int(root.score))] = float(root.recordCount)
                else:
                    prob[self.classes.index(root.score)] = float(root.recordCount)
            else:
                prob = [float(root.score)]
            return prob
        idx = self.fields.index(root.field)
        if sample[idx] <= float(root.value):
            result = self._predict(root.left, sample)
        else:
            result = self._predict(root.right, sample)
        return result

    def build_tree(self):
        cur_node = self.root
        for nd in self.all_node_list:
            if nd[2] == cur_node.value:
                if nd[3] == 'lessOrEqual':
                    cur_node.left = self.Node()
                    if nd[0] != -2:
                        cur_node.left.score = nd[0]
                    else:
                        cur_node.left.parent = cur_node
                        cur_node = cur_node.left
                else:
                    cur_node.right = self.Node()
                    if nd[0] != -2:
                        cur_node.right.score = nd[0]
                        cur_node = cur_node.parent
                        while cur_node and cur_node.right:
                            cur_node = cur_node.parent
                            if not cur_node:
                                break
                    else:
                        cur_node.right.parent = cur_node
                        cur_node = cur_node.right
            else:
                cur_node.field = nd[1]
                cur_node.value = nd[2]
                cur_node.recordCount = nd[4]
                if nd[3] == 'lessOrEqual':
                    cur_node.left = self.Node()
                    if nd[0] != -2:
                        cur_node.left.score = nd[0]
                    else:
                        cur_node.left.parent = cur_node
                        cur_node = cur_node.left
                else:
                    cur_node.right = self.Node()
                    if nd[0] != -2:
                        cur_node.right.score = nd[0]
                        cur_node = cur_node.parent
                        while cur_node and cur_node.right:
                            cur_node = cur_node.parent
                            if not cur_node:
                                break
                    else:
                        cur_node.right.parent = cur_node
                        cur_node = cur_node.right





