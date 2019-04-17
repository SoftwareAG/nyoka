        if not hasattr(self, "elementobjs_"):
            self.elementobjs_ = []
        if hasattr(self, nodeName_) and nodeName_ not in self.elementobjs_:
            nodeName_ += '_'
        if nodeName_ not in self.elementobjs_:
            self.elementobjs_.append(nodeName_)
        if not eval("hasattr(self, '" + nodeName_ + "')"):
            nodeVal = list(filter(None, [obj_.lstrip(' ') for obj_ in child_.text.split('\n')]))[0]
            try:
                setattr(self, nodeName_,eval(nodeVal))
            except:
                setattr(self, nodeName_,nodeVal)
        else:
            if getattr(self,nodeName_).__class__.__name__ == 'str':
                setattr(self,nodeName_,[getattr(self,nodeName_)])
            else:
                setattr(self,nodeName_,list(getattr(self,nodeName_)))
            nodeVal = list(filter(None, [obj_.lstrip(' ') for obj_ in child_.text.split('\n')]))[0]
            try:
                getattr(self, nodeName_).append(eval(nodeVal))
            except:
                getattr(self, nodeName_).append(nodeVal)
                